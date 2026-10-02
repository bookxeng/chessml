from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constants import WHITE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, square_file, square_rank
from .movegen import legal_moves

if TYPE_CHECKING:
    from .board import Board
    from .move import Move


@dataclass(frozen=True)
class Outcome:
    result: str          # "1-0", "0-1" or "1/2-1/2"
    reason: str          # "checkmate", "stalemate", "insufficient_material", "fifty_moves", "threefold_repetition"
    winner: int | None   # WHITE, BLACK, or None for a draw


def is_insufficient_material(board: Board) -> bool:
    """Neither side can possibly checkmate: K v K, K+minor v K, or only same-colored bishops."""
    minors = []
    for sq, p in enumerate(board.squares):
        kind = abs(p)
        if kind in (PAWN, ROOK, QUEEN):
            return False
        if kind in (KNIGHT, BISHOP):
            minors.append((kind, sq))
    if len(minors) <= 1:
        return True
    if all(kind == BISHOP for kind, _ in minors):
        colors = {(square_file(sq) + square_rank(sq)) % 2 for _, sq in minors}
        return len(colors) == 1
    return False


def outcome(board: Board, moves: list[Move] | None = None) -> Outcome | None:
    """The game result if the game is over, else None.

    Draws by the fifty-move rule and threefold repetition are applied automatically
    (rather than on claim), which is what you want for self-play training.
    Pass `moves` if the legal moves are already known, to avoid regenerating them.
    """
    if moves is None:
        moves = legal_moves(board)
    if not moves:
        if board.in_check():
            winner = -board.turn
            return Outcome("1-0" if winner == WHITE else "0-1", "checkmate", winner)
        return Outcome("1/2-1/2", "stalemate", None)
    if is_insufficient_material(board):
        return Outcome("1/2-1/2", "insufficient_material", None)
    if board.halfmove_clock >= 100:
        return Outcome("1/2-1/2", "fifty_moves", None)
    if board.repetition_count() >= 3:
        return Outcome("1/2-1/2", "threefold_repetition", None)
    return None

