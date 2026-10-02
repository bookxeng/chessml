"""Turn PGN games into compact training positions using the chessml engine."""
from __future__ import annotations

import re

import numpy as np

from chessml import Board, parse_san

from .positions import POSITION_DTYPE, position_record

RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}

_COMMENT = re.compile(r"\{[^}]*\}|;[^\n]*")
_SKIP = re.compile(r"^(\d+\.+|\$\d+|1-0|0-1|1/2-1/2|\*)$")


def movetext_sans(movetext: str) -> list[str]:
    """SAN tokens from PGN movetext, without comments, move numbers, NAGs or the result."""
    tokens = _COMMENT.sub(" ", movetext).split()
    return [t for t in tokens if not _SKIP.match(t)]


def game_positions(sans: list[str], result: str) -> np.ndarray:
    """One POSITION_DTYPE record per move of the game. Raises ValueError on a bad move."""
    result_for_white = RESULTS[result]
    board = Board()
    rows = []
    for san in sans:
        move = parse_san(board, san)
        rows.append(position_record(board, move, result_for_white))
        board.push(move)
    return np.array(rows, dtype=POSITION_DTYPE)
