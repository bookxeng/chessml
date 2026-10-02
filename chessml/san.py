"""Standard Algebraic Notation (e.g. "Nf3", "exd5", "O-O", "e8=Q+")."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import PAWN, KING, PIECE_SYMBOLS, FILE_NAMES, RANK_NAMES, square_name
from .movegen import legal_moves

if TYPE_CHECKING:
    from .board import Board
    from .move import Move


def _san_without_suffix(board: Board, move: Move, moves: list[Move]) -> str:
    sq = board.squares
    piece = sq[move.from_sq]
    kind = abs(piece)
    frm, to = move.from_sq, move.to_sq

    if kind == KING and abs(to - frm) == 2:
        return "O-O" if to > frm else "O-O-O"

    capture = sq[to] != 0 or (kind == PAWN and to == board.ep_square)
    if kind == PAWN:
        s = (FILE_NAMES[frm & 7] + "x" if capture else "") + square_name(to)
        if move.promotion:
            s += "=" + PIECE_SYMBOLS[move.promotion].upper()
        return s

    s = PIECE_SYMBOLS[kind].upper()
    rivals = [m.from_sq for m in moves
              if m.to_sq == to and m.from_sq != frm and sq[m.from_sq] == piece]
    if rivals:
        if all((r & 7) != (frm & 7) for r in rivals):
            s += FILE_NAMES[frm & 7]
        elif all((r >> 3) != (frm >> 3) for r in rivals):
            s += RANK_NAMES[frm >> 3]
        else:
            s += square_name(frm)
    return s + ("x" if capture else "") + square_name(to)


def move_to_san(board: Board, move: Move, moves: list[Move] | None = None) -> str:
    """SAN for a legal move in the current position, including the +/# suffix."""
    if moves is None:
        moves = legal_moves(board)
    s = _san_without_suffix(board, move, moves)
    board.push(move)
    try:
        if board.in_check():
            s += "#" if not legal_moves(board) else "+"
    finally:
        board.pop()
    return s


def _normalize(text: str) -> str:
    text = text.strip().replace("0", "O")
    for ch in "+#!?=x":
        text = text.replace(ch, "")
    return text


def parse_san(board: Board, text: str) -> Move:
    """Parse SAN into a legal move. Lenient about +, #, x, = and 0-0 vs O-O."""
    wanted = _normalize(text)
    if not wanted:
        raise ValueError("empty move")
    moves = legal_moves(board)
    matches = [m for m in moves if _normalize(_san_without_suffix(board, m, moves)) == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"illegal or unrecognized move: {text!r}")
    raise ValueError(f"ambiguous move: {text!r}")
