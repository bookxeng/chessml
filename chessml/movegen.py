from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import (
    WHITE, PAWN, KNIGHT, BISHOP, ROOK, KING, PROMOTION_PIECES,
    WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE,
    KNIGHT_TARGETS, KING_TARGETS, ROOK_RAYS, BISHOP_RAYS, PAWN_ATTACKS,
)
from .move import Move

if TYPE_CHECKING:
    from .board import Board


def _add_pawn_move(moves: list[Move], frm: int, to: int, promo_rank: int) -> None:
    if to >> 3 == promo_rank:
        for p in PROMOTION_PIECES:
            moves.append(Move(frm, to, p))
    else:
        moves.append(Move(frm, to))


def pseudo_legal_moves(board: Board) -> list[Move]:
    """All moves that obey piece movement rules, ignoring whether the own king is left in check.

    Castling is fully checked here (including attacked squares) since that can't be caught later.
    """
    sq = board.squares
    us = board.turn
    moves: list[Move] = []
    forward = 8 * us
    start_rank = 1 if us == WHITE else 6
    promo_rank = 7 if us == WHITE else 0

    for frm in range(64):
        p = sq[frm] * us
        if p <= 0:
            continue

        if p == PAWN:
            one = frm + forward
            if sq[one] == 0:
                _add_pawn_move(moves, frm, one, promo_rank)
                if frm >> 3 == start_rank and sq[one + forward] == 0:
                    moves.append(Move(frm, one + forward))
            for to in PAWN_ATTACKS[us][frm]:
                if sq[to] * us < 0:
                    _add_pawn_move(moves, frm, to, promo_rank)
                elif to == board.ep_square:
                    moves.append(Move(frm, to))

        elif p == KNIGHT or p == KING:
            for to in (KNIGHT_TARGETS if p == KNIGHT else KING_TARGETS)[frm]:
                if sq[to] * us <= 0:
                    moves.append(Move(frm, to))

        else:
            if p == ROOK:
                rays = ROOK_RAYS[frm]
            elif p == BISHOP:
                rays = BISHOP_RAYS[frm]
            else:
                rays = ROOK_RAYS[frm] + BISHOP_RAYS[frm]
            for ray in rays:
                for to in ray:
                    target = sq[to] * us
                    if target <= 0:
                        moves.append(Move(frm, to))
                    if target != 0:
                        break

    _add_castling_moves(board, moves)
    return moves


def _add_castling_moves(board: Board, moves: list[Move]) -> None:
    us = board.turn
    if us == WHITE:
        king_rights, queen_rights, base = WHITE_KINGSIDE, WHITE_QUEENSIDE, 0
    else:
        king_rights, queen_rights, base = BLACK_KINGSIDE, BLACK_QUEENSIDE, 56
    if not board.castling & (king_rights | queen_rights):
        return
    sq = board.squares
    e = base + 4
    if sq[e] != KING * us or board.is_attacked(e, -us):
        return
    rook = ROOK * us
    if (board.castling & king_rights and sq[base + 7] == rook
            and sq[base + 5] == 0 and sq[base + 6] == 0
            and not board.is_attacked(base + 5, -us) and not board.is_attacked(base + 6, -us)):
        moves.append(Move(e, base + 6))
    if (board.castling & queen_rights and sq[base] == rook
            and sq[base + 1] == 0 and sq[base + 2] == 0 and sq[base + 3] == 0
            and not board.is_attacked(base + 3, -us) and not board.is_attacked(base + 2, -us)):
        moves.append(Move(e, base + 2))


def legal_moves(board: Board) -> list[Move]:
    us = board.turn
    legal = []
    for m in pseudo_legal_moves(board):
        board.push(m)
        if not board.is_attacked(board.king_sq[us], -us):
            legal.append(m)
        board.pop()
    return legal


def perft(board: Board, depth: int) -> int:
    """Count leaf nodes of the legal move tree; the standard correctness check for move generators."""
    moves = legal_moves(board)
    if depth <= 1:
        return len(moves) if depth == 1 else 1
    total = 0
    for m in moves:
        board.push(m)
        total += perft(board, depth - 1)
        board.pop()
    return total
