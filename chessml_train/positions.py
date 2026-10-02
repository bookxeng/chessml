"""Compact training positions and their batched encoding into network planes.

A position is stored as one record of POSITION_DTYPE (71 bytes) instead of the 19x8x8
float32 planes the network consumes (4864 bytes); see docs/adr/0003. `encode_batch`
turns a whole array of records into planes at once and must match
`chessml.encode_board` exactly.

Numpy only (no torch, no Python 3.14 features) so it runs on Colab and the local PC.
"""
from __future__ import annotations

import numpy as np

from chessml import Board, Move, WHITE, move_to_action
from chessml.constants import WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE
from chessml.encoding import NUM_PLANES

POSITION_DTYPE = np.dtype([
    ("squares", np.int8, 64),   # board.squares: signed piece codes, a1 = 0
    ("turn", np.int8),          # +1 white to move, -1 black to move
    ("castling", np.uint8),     # castling-right bits
    ("ep", np.int8),            # en passant square, -1 for none
    ("halfmove", np.uint8),     # halfmove clock, clipped to 255
    ("action", np.int16),       # move played, as an action index for the side to move
    ("result", np.int8),        # game result for the side to move: +1 win, 0 draw, -1 loss
])

_MIRROR = np.arange(64) ^ 56


def position_record(board: Board, move: Move, result_for_white: int) -> tuple:
    """One POSITION_DTYPE row for `move` played in `board` (before the move is pushed)."""
    return (
        tuple(board.squares),  # snapshot: board.squares is mutated by later pushes
        board.turn,
        board.castling,
        -1 if board.ep_square is None else board.ep_square,
        min(board.halfmove_clock, 255),
        move_to_action(move, board),
        result_for_white * board.turn,
    )


def encode_batch(records: np.ndarray) -> np.ndarray:
    """Planes for an array of POSITION_DTYPE records: float32 (N, 19, 8, 8)."""
    n = len(records)
    turn = records["turn"].astype(np.int8)
    black = turn != WHITE
    squares = records["squares"]
    oriented = np.where(black[:, None], squares[:, _MIRROR], squares)
    rel = oriented * turn[:, None]  # positive = our pieces

    obs = np.zeros((n, NUM_PLANES, 64), dtype=np.float32)
    for kind in range(1, 7):
        obs[:, kind - 1] = rel == kind
        obs[:, kind + 5] = rel == -kind
    obs[:, 12] = (~black)[:, None]

    # Planes 13-16: our kingside, our queenside, their kingside, their queenside.
    castling = records["castling"]
    white_order = np.array([WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE], np.uint8)
    black_order = np.array([BLACK_KINGSIDE, BLACK_QUEENSIDE, WHITE_KINGSIDE, WHITE_QUEENSIDE], np.uint8)
    bits = np.where(black[:, None], black_order, white_order)
    obs[:, 13:17] = ((castling[:, None] & bits) != 0)[:, :, None]

    ep = records["ep"].astype(np.int64)
    has_ep = ep >= 0
    ep_oriented = np.where(black, ep ^ 56, ep)
    rows = np.flatnonzero(has_ep)
    obs[rows, 17, ep_oriented[rows]] = 1.0

    obs[:, 18] = (records["halfmove"] / 100.0).astype(np.float32)[:, None]
    return obs.reshape(n, NUM_PLANES, 8, 8)
