"""Tensor encodings of positions and moves for neural networks.

Everything is from the perspective of the side to move: when black is to move, the
board is mirrored vertically (square ^ 56) so "our" pawns always advance up the board.
This lets one network play both colors.

Observation: float32 array of shape (19, 8, 8), indexed [plane, rank, file].
    0-5    our pieces      P N B R Q K
    6-11   their pieces    P N B R Q K
    12     side to move is white (all ones) / black (all zeros)
    13-14  our castling rights       kingside, queenside
    15-16  their castling rights     kingside, queenside
    17     en passant target square
    18     halfmove clock / 100

Action: an int in [0, 4672), AlphaZero-style 8x8x73 = from_square * 73 + plane, where
the from square is also oriented to the side to move.
    planes 0-55   queen-like moves: direction * 7 + (distance - 1),
                  directions N, NE, E, SE, S, SW, W, NW (queen promotions use these)
    planes 56-63  knight moves
    planes 64-72  underpromotions: piece * 3 + direction,
                  piece in (knight, bishop, rook), direction in (capture left, push, capture right)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import (
    WHITE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN,
    WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE,
    KNIGHT_DELTAS,
)
from .move import Move
from .movegen import legal_moves

if TYPE_CHECKING:
    from .board import Board

NUM_PLANES = 19
OBS_SHAPE = (NUM_PLANES, 8, 8)
NUM_ACTIONS = 8 * 8 * 73

_QUEEN_DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_UNDERPROMOTIONS = (KNIGHT, BISHOP, ROOK)

# Precomputed (df, dr) -> plane lookups and the reverse.
_DELTA_TO_PLANE: dict[tuple[int, int], int] = {}
_PLANE_TO_DELTA: list[tuple[int, int]] = []
for _d, (_df, _dr) in enumerate(_QUEEN_DIRECTIONS):
    for _dist in range(1, 8):
        _DELTA_TO_PLANE[(_df * _dist, _dr * _dist)] = len(_PLANE_TO_DELTA)
        _PLANE_TO_DELTA.append((_df * _dist, _dr * _dist))
for _df, _dr in KNIGHT_DELTAS:
    _DELTA_TO_PLANE[(_df, _dr)] = len(_PLANE_TO_DELTA)
    _PLANE_TO_DELTA.append((_df, _dr))
assert len(_PLANE_TO_DELTA) == 64


def _orient(sq: int, color: int) -> int:
    return sq if color == WHITE else sq ^ 56


def encode_board(board: Board) -> np.ndarray:
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    us = board.turn
    for sq, p in enumerate(board.squares):
        if p:
            plane = abs(p) - 1 + (0 if p * us > 0 else 6)
            o = _orient(sq, us)
            obs[plane, o >> 3, o & 7] = 1.0
    if us == WHITE:
        obs[12] = 1.0
        ours, theirs = (WHITE_KINGSIDE, WHITE_QUEENSIDE), (BLACK_KINGSIDE, BLACK_QUEENSIDE)
    else:
        ours, theirs = (BLACK_KINGSIDE, BLACK_QUEENSIDE), (WHITE_KINGSIDE, WHITE_QUEENSIDE)
    for plane, bit in zip((13, 14, 15, 16), ours + theirs):
        if board.castling & bit:
            obs[plane] = 1.0
    if board.ep_square is not None:
        o = _orient(board.ep_square, us)
        obs[17, o >> 3, o & 7] = 1.0
    obs[18] = board.halfmove_clock / 100.0
    return obs


def move_to_action(move: Move, board: Board) -> int:
    """Action index for a move in the given position (board supplies the side to move)."""
    us = board.turn
    frm, to = _orient(move.from_sq, us), _orient(move.to_sq, us)
    df, dr = (to & 7) - (frm & 7), (to >> 3) - (frm >> 3)
    if move.promotion and move.promotion != QUEEN:
        plane = 64 + _UNDERPROMOTIONS.index(move.promotion) * 3 + (df + 1)
    else:
        plane = _DELTA_TO_PLANE[(df, dr)]
    return frm * 73 + plane


def action_to_move(action: int, board: Board) -> Move | None:
    """Decode an action in the given position. Returns None if it points off the board.

    The result is not necessarily legal; check against `legal_moves` / `legal_action_mask`.
    """
    if not 0 <= action < NUM_ACTIONS:
        return None
    us = board.turn
    frm, plane = divmod(action, 73)
    promotion = 0
    if plane >= 64:
        piece_idx, dir_idx = divmod(plane - 64, 3)
        df, dr = dir_idx - 1, 1
        promotion = _UNDERPROMOTIONS[piece_idx]
    else:
        df, dr = _PLANE_TO_DELTA[plane]
    f, r = (frm & 7) + df, (frm >> 3) + dr
    if not (0 <= f < 8 and 0 <= r < 8):
        return None
    from_sq, to_sq = _orient(frm, us), _orient(r * 8 + f, us)
    if not promotion and r == 7 and abs(board.squares[from_sq]) == PAWN:
        promotion = QUEEN
    return Move(from_sq, to_sq, promotion)


def legal_action_mask(board: Board, moves: list[Move] | None = None) -> np.ndarray:
    """Boolean array of shape (4672,), True for each legal action."""
    if moves is None:
        moves = legal_moves(board)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for m in moves:
        mask[move_to_action(m, board)] = True
    return mask
