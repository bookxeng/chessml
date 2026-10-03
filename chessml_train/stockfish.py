"""Stockfish as an Opponent ladder rung, over the UCI protocol.

Only ever an opponent: games against it are never training data (see docs/adr/0001).
Download Stockfish from https://stockfishchess.org/download/ and pass its path, or put it on
PATH, or set the STOCKFISH environment variable.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from chessml import Board, Move

MIN_UCI_ELO = 1320  # Stockfish's lowest calibrated UCI_Elo


def find_stockfish(path: str | None = None) -> str | None:
    return path or os.environ.get("STOCKFISH") or shutil.which("stockfish")


class StockfishPlayer:
    """Stockfish limited to `elo` (UCI_LimitStrength), thinking `movetime_ms` per move."""

    def __init__(self, path: str, elo: int | None = None, movetime_ms: int = 100):
        self.movetime_ms = movetime_ms
        self.proc = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._wait_for("uciok")
        if elo is not None:
            if elo < MIN_UCI_ELO:
                raise ValueError(f"Stockfish UCI_Elo must be >= {MIN_UCI_ELO}")
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {elo}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, prefix: str) -> str:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish exited unexpectedly")
            if line.startswith(prefix):
                return line.strip()

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def choose(self, board: Board) -> Move:
        self._send(f"position fen {board.fen()}")
        self._send(f"go movetime {self.movetime_ms}")
        best = self._wait_for("bestmove").split()[1]
        move = Move.from_uci(best)
        if move not in board.legal_moves():
            raise RuntimeError(f"Stockfish returned illegal move {best} in {board.fen()}")
        return move

    def close(self) -> None:
        if self.proc.poll() is None:
            self._send("quit")
            self.proc.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
