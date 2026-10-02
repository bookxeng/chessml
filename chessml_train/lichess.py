"""Stream a Lichess monthly database, filter games, and write compact position shards.

    python -m chessml_train.lichess --month 2026-08 --games 500000 --out data/lichess-2026-08

The download is decompressed as it streams and stops once enough games are accepted, so
only a fraction of the monthly file is read. Moves are parsed by the chessml engine in
worker processes. Output: train-NNNNN.npy / val-NNNNN.npy shards of POSITION_DTYPE records
(val = every 50th game, so no game is split between the two) plus manifest.json.

Needs Python 3.14 (compression.zstd); this runs on the local PC, not on Colab.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .pgn import RESULTS, game_positions, movetext_sans
from .positions import POSITION_DTYPE

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
VAL_EVERY = 50


@dataclass
class GameFilter:
    min_elo: int = 2000
    events: tuple[str, ...] = ("Rated Blitz", "Rated Rapid")
    terminations: tuple[str, ...] = ("Normal", "Time forfeit", "Insufficient material")

    def accepts(self, headers: dict[str, str]) -> bool:
        if not headers.get("Event", "").startswith(self.events):
            return False
        if headers.get("Termination") not in self.terminations:
            return False
        if headers.get("Result") not in RESULTS or "FEN" in headers:
            return False
        try:
            return min(int(headers["WhiteElo"]), int(headers["BlackElo"])) >= self.min_elo
        except (KeyError, ValueError):
            return False


def iter_games(lines: Iterable[str]) -> Iterator[tuple[dict[str, str], str]]:
    """(headers, movetext) for each game in a PGN text stream."""
    headers: dict[str, str] = {}
    movetext: list[str] = []
    for line in lines:
        if line.startswith("["):
            if movetext:
                yield headers, " ".join(movetext)
                headers, movetext = {}, []
            key, _, value = line[1:].rstrip().rstrip("]").partition(" ")
            headers[key] = value.strip('"')
        elif line.strip():
            movetext.append(line.strip())
    if movetext:
        yield headers, " ".join(movetext)


def _parse_chunk(chunk: list[tuple[str, str]]) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    for movetext, result in chunk:
        try:
            out.append(game_positions(movetext_sans(movetext), result))
        except (ValueError, KeyError):
            out.append(None)
    return out


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, games_per_shard: int):
        self.out_dir, self.prefix, self.games_per_shard = out_dir, prefix, games_per_shard
        self.pending: list[np.ndarray] = []
        self.shards = 0
        self.games = 0
        self.positions = 0

    def add(self, game: np.ndarray) -> None:
        self.pending.append(game)
        self.games += 1
        self.positions += len(game)
        if len(self.pending) >= self.games_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        path = self.out_dir / f"{self.prefix}-{self.shards:05d}.npy"
        np.save(path, np.concatenate(self.pending))
        self.pending = []
        self.shards += 1


@dataclass
class Stats:
    scanned: int = 0
    accepted: int = 0
    parse_errors: int = 0
    events: Counter = field(default_factory=Counter)


def _accepted_chunks(lines, game_filter: GameFilter, stats: Stats, chunk_size: int):
    chunk: list[tuple[str, str]] = []
    for headers, movetext in iter_games(lines):
        stats.scanned += 1
        if game_filter.accepts(headers):
            stats.accepted += 1
            stats.events[headers["Event"]] += 1
            chunk.append((movetext, headers["Result"]))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def build_dataset(lines: Iterable[str], out_dir: Path, target_games: int, game_filter: GameFilter,
                  workers: int, games_per_shard: int = 10_000, chunk_size: int = 200,
                  log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = ShardWriter(out_dir, "train", games_per_shard)
    val = ShardWriter(out_dir, "val", games_per_shard)
    stats = Stats()
    written = 0
    start = time.time()
    next_log = games_per_shard

    chunks = _accepted_chunks(lines, game_filter, stats, chunk_size)
    with mp.Pool(workers) as pool:
        for games in pool.imap(_parse_chunk, chunks):
            for game in games:
                if game is None:
                    stats.parse_errors += 1
                    continue
                (val if written % VAL_EVERY == 0 else train).add(game)
                written += 1
                if written >= target_games:
                    break
            if written >= next_log:
                rate = stats.scanned / (time.time() - start)
                log(f"{written:,} games written, {stats.scanned:,} scanned ({rate:,.0f}/s), "
                    f"{stats.parse_errors} parse errors")
                next_log += games_per_shard
            if written >= target_games:
                pool.terminate()
                break
    train.flush()
    val.flush()

    manifest = {
        "games": written,
        "train": {"games": train.games, "positions": train.positions, "shards": train.shards},
        "val": {"games": val.games, "positions": val.positions, "shards": val.shards},
        "scanned": stats.scanned,
        "parse_errors": stats.parse_errors,
        "events": dict(stats.events.most_common()),
        "filter": asdict(game_filter),
        "dtype": str(POSITION_DTYPE.descr),
        "seconds": round(time.time() - start, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def stream_lines(month: str) -> Iterator[str]:
    from compression import zstd  # Python 3.14+

    with urllib.request.urlopen(URL.format(month=month)) as resp:
        with zstd.open(resp, "rt", encoding="utf-8") as f:
            yield from f


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--month", required=True, help="YYYY-MM of the Lichess standard rated dump")
    ap.add_argument("--games", type=int, default=500_000, help="games to keep")
    ap.add_argument("--out", type=Path, required=True, help="output directory for shards")
    ap.add_argument("--min-elo", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)

    manifest = build_dataset(stream_lines(args.month), args.out, args.games,
                             GameFilter(min_elo=args.min_elo), args.workers)
    print(json.dumps({k: manifest[k] for k in ("games", "train", "val", "scanned", "parse_errors", "seconds")},
                     indent=2))


if __name__ == "__main__":
    main()
