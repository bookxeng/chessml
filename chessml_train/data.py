"""Batches of training positions from the .npy shards written by `chessml_train.lichess`.

Shards are loaded a few at a time, shuffled together, and encoded in batches with
`encode_batch`, so memory stays at a few GB however large the dataset is. The order is a
pure function of (seed, epoch), so a resumed run can skip exactly the batches it has
already seen.
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .positions import encode_batch

Batch = tuple[np.ndarray, np.ndarray, np.ndarray]  # planes float32 (N,19,8,8), action int64, wdl int64


def wdl_targets(result: np.ndarray) -> np.ndarray:
    """Value class for each position: 0 win, 1 draw, 2 loss (result is +1/0/-1 for the side to move)."""
    return (1 - result.astype(np.int64))


def make_batch(records: np.ndarray) -> Batch:
    return encode_batch(records), records["action"].astype(np.int64), wdl_targets(records["result"])


class ShardStream:
    def __init__(self, paths: list[Path], batch_size: int, shards_in_memory: int = 6, seed: int = 0):
        if not paths:
            raise ValueError("no shards given")
        self.paths = sorted(paths)
        self.batch_size = batch_size
        self.shards_in_memory = shards_in_memory
        self.seed = seed

    def batches(self, epoch: int, skip: int = 0) -> Iterator[Batch]:
        """Full batches for one epoch, skipping the first `skip` of them (for resuming)."""
        rng = np.random.default_rng((self.seed, epoch))
        order = rng.permutation(len(self.paths))
        leftover = None
        index = 0
        for start in range(0, len(order), self.shards_in_memory):
            group = [np.load(self.paths[i]) for i in order[start:start + self.shards_in_memory]]
            if leftover is not None:
                group.append(leftover)
            records = np.concatenate(group)
            records = records[rng.permutation(len(records))]
            n_full = len(records) // self.batch_size
            for b in range(n_full):
                if index >= skip:
                    yield make_batch(records[b * self.batch_size:(b + 1) * self.batch_size])
                index += 1
            leftover = records[n_full * self.batch_size:]


def prefetch(it: Iterator[Batch], depth: int = 4) -> Iterator[Batch]:
    """Run `it` in a background thread so batch preparation overlaps with GPU work."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    errors: list[BaseException] = []

    def worker():
        try:
            for item in it:
                q.put(item)
        except BaseException as e:  # surfaced in the consumer
            errors.append(e)
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not done:
        yield item
    if errors:
        raise errors[0]
