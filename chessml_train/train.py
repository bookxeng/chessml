"""Train the Supervised model on Lichess shards, with checkpoints that survive Colab resets.

    python -m chessml_train.train --data data/lichess-2026-08 --out runs/sl-10x128

Rerunning the same command resumes from <out>/latest.pt. <out>/best.pt holds the weights
with the lowest validation loss so far, and <out>/metrics.jsonl has one line per log event.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import ShardStream, make_batch, prefetch
from .model import ChessNet, NetConfig, count_parameters


@dataclass
class TrainConfig:
    data: str
    out: str
    blocks: int = 10
    filters: int = 128
    batch_size: int = 1024
    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    value_weight: float = 1.0
    log_every: int = 200
    eval_every: int = 2000
    val_positions: int = 200_000
    max_steps: int | None = None  # stop early (smoke tests); the LR schedule still uses the full length
    seed: int = 0


def _lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


def _losses(model, planes, action, wdl, value_weight):
    policy, value = model(planes)
    policy_loss = F.cross_entropy(policy.float(), action)
    value_loss = F.cross_entropy(value.float(), wdl)
    accuracy = (policy.argmax(1) == action).float().mean()
    return policy_loss + value_weight * value_loss, policy_loss, value_loss, accuracy


def _to_device(batch, device):
    planes, action, wdl = (torch.from_numpy(a) for a in batch)
    if device.type == "cuda":
        planes, action, wdl = planes.pin_memory(), action.pin_memory(), wdl.pin_memory()
    return (planes.to(device, non_blocking=True), action.to(device, non_blocking=True),
            wdl.to(device, non_blocking=True))


@torch.no_grad()
def evaluate(model, val_records: np.ndarray, cfg: TrainConfig, device, amp: bool) -> dict:
    model.eval()
    totals = np.zeros(3)
    n = 0
    for start in range(0, len(val_records), cfg.batch_size):
        batch = _to_device(make_batch(val_records[start:start + cfg.batch_size]), device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            _, p, v, acc = _losses(model, *batch, cfg.value_weight)
        k = len(batch[1])
        totals += np.array([p.item(), v.item(), acc.item()]) * k
        n += k
    model.train()
    p, v, acc = totals / n
    return {"val_policy_loss": p, "val_value_loss": v, "val_loss": p + cfg.value_weight * v, "val_accuracy": acc}


def _save(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg: TrainConfig, log=print) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.manual_seed(cfg.seed)
    data_dir, out = Path(cfg.data), Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data_dir / "manifest.json").read_text())
    stream = ShardStream(list(data_dir.glob("train-*.npy")), cfg.batch_size, seed=cfg.seed)
    val_records = np.concatenate([np.load(p) for p in sorted(data_dir.glob("val-*.npy"))])[:cfg.val_positions]
    steps_per_epoch = manifest["train"]["positions"] // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs

    net_config = NetConfig(blocks=cfg.blocks, filters=cfg.filters)
    model = ChessNet(net_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg.warmup_steps, total_steps))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    step, best = 0, float("inf")

    latest = out / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step, best = ckpt["step"], ckpt["best_val_loss"]
        log(f"resumed from step {step:,}")
    else:
        log(f"new run: {count_parameters(model):,} parameters, {total_steps:,} steps "
            f"({cfg.epochs} epochs x {steps_per_epoch:,}), device {device}")

    def checkpoint():
        _save(latest, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                       "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                       "step": step, "best_val_loss": best,
                       "train_config": asdict(cfg), "net_config": asdict(net_config)})

    metrics_file = (out / "metrics.jsonl").open("a")
    def record(entry: dict):
        metrics_file.write(json.dumps(entry) + "\n")
        metrics_file.flush()

    stop_at = total_steps if cfg.max_steps is None else min(total_steps, cfg.max_steps)
    running = np.zeros(4)
    t0, seen = time.time(), 0
    model.train()
    while step < stop_at:
        epoch, skip = divmod(step, steps_per_epoch)
        for batch in prefetch(stream.batches(epoch, skip)):
            planes, action, wdl = _to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss, p, v, acc = _losses(model, planes, action, wdl, cfg.value_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            seen += len(action)
            running += np.array([loss.item(), p.item(), v.item(), acc.item()])

            if step % cfg.log_every == 0 or step == stop_at:
                n = (step - 1) % cfg.log_every + 1
                l, pl, vl, a = running / n
                running[:] = 0
                rate = seen / (time.time() - t0)
                entry = {"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": l,
                         "policy_loss": pl, "value_loss": vl, "accuracy": a,
                         "lr": sched.get_last_lr()[0], "positions_per_s": rate}
                record(entry)
                log(f"step {step:,}/{total_steps:,}  loss {l:.3f} (policy {pl:.3f}, value {vl:.3f})  "
                    f"move acc {a:.1%}  {rate:,.0f} pos/s")

            if step % cfg.eval_every == 0 or step == stop_at:
                val = evaluate(model, val_records, cfg, device, amp)
                record({"step": step, **val})
                if val["val_loss"] < best:
                    best = val["val_loss"]
                    _save(out / "best.pt", {"model": model.state_dict(), "net_config": asdict(net_config),
                                            "step": step, **val})
                log(f"  val loss {val['val_loss']:.3f}  val move acc {val['val_accuracy']:.1%}"
                    f"  (best {best:.3f})")
                checkpoint()

            if step >= stop_at or step % steps_per_epoch == 0:
                break
    metrics_file.close()
    return {"step": step, "best_val_loss": best}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for f in fields(TrainConfig):
        kind = int if f.type in ("int", "int | None") else float if f.type == "float" else str
        required = f.name in ("data", "out")
        ap.add_argument("--" + f.name.replace("_", "-"), type=kind, required=required,
                        default=None if required else f.default)
    train(TrainConfig(**vars(ap.parse_args(argv))))


if __name__ == "__main__":
    main()
