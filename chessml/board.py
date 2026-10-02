from __future__ import annotations

import random

from .constants import (
    WHITE, BLACK, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    WHITE_KINGSIDE, WHITE_QUEENSIDE, BLACK_KINGSIDE, BLACK_QUEENSIDE,
    KNIGHT_TARGETS, KING_TARGETS, ROOK_RAYS, BISHOP_RAYS, PAWN_ATTACKS, CASTLING_MASK,
    STARTING_FEN, PIECE_SYMBOLS, parse_square, square_name, piece_symbol,
)
from .move import Move

# Zobrist keys, fixed seed so hashes are stable across runs.
_rng = random.Random(0xC0FFEE)
_PIECE_KEYS = [[_rng.getrandbits(64) for _ in range(64)] for _ in range(13)]  # index: piece + 6
_CASTLING_KEYS = [_rng.getrandbits(64) for _ in range(16)]
_EP_KEYS = [_rng.getrandbits(64) for _ in range(8)]
_TURN_KEY = _rng.getrandbits(64)

_CASTLING_FEN = ((WHITE_KINGSIDE, "K"), (WHITE_QUEENSIDE, "Q"), (BLACK_KINGSIDE, "k"), (BLACK_QUEENSIDE, "q"))


class Board:
    """Full chess position plus the move history needed for undo and repetition detection."""

    def __init__(self, fen: str = STARTING_FEN):
        self.set_fen(fen)

    # ------------------------------------------------------------------ FEN

    def set_fen(self, fen: str) -> None:
        parts = fen.split()
        if len(parts) < 4:
            raise ValueError(f"FEN needs at least 4 fields: {fen!r}")
        placement, turn, castling, ep = parts[:4]
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1

        rows = placement.split("/")
        if len(rows) != 8:
            raise ValueError(f"FEN placement needs 8 ranks: {placement!r}")
        squares = [EMPTY] * 64
        for i, row in enumerate(rows):
            rank = 7 - i
            file = 0
            for ch in row:
                if ch.isdigit():
                    file += int(ch)
                else:
                    kind = PIECE_SYMBOLS.find(ch.lower())
                    if kind <= 0 or file > 7:
                        raise ValueError(f"bad FEN rank {row!r}")
                    squares[rank * 8 + file] = kind if ch.isupper() else -kind
                    file += 1
            if file != 8:
                raise ValueError(f"FEN rank {row!r} does not have 8 files")

        if turn not in ("w", "b"):
            raise ValueError(f"bad side to move: {turn!r}")
        rights = 0
        if castling != "-":
            for bit, ch in _CASTLING_FEN:
                if ch in castling:
                    rights |= bit

        kings = {WHITE: None, BLACK: None}
        for sq, p in enumerate(squares):
            if abs(p) == KING:
                color = WHITE if p > 0 else BLACK
                if kings[color] is not None:
                    raise ValueError("more than one king per side")
                kings[color] = sq
        if kings[WHITE] is None or kings[BLACK] is None:
            raise ValueError("each side needs exactly one king")

        self.squares = squares
        self.turn = WHITE if turn == "w" else BLACK
        self.castling = rights
        self.ep_square = None if ep == "-" else parse_square(ep)
        self.halfmove_clock = halfmove
        self.fullmove_number = fullmove
        self.king_sq = kings
        self._stack = []
        self.hash = self._compute_hash()
        self.history = [self.hash]  # position hashes, one per position reached

    def fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row, empty = "", 0
            for file in range(8):
                p = self.squares[rank * 8 + file]
                if p == EMPTY:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += piece_symbol(p)
            if empty:
                row += str(empty)
            rows.append(row)
        castling = "".join(ch for bit, ch in _CASTLING_FEN if self.castling & bit) or "-"
        ep = square_name(self.ep_square) if self.ep_square is not None else "-"
        turn = "w" if self.turn == WHITE else "b"
        return f"{'/'.join(rows)} {turn} {castling} {ep} {self.halfmove_clock} {self.fullmove_number}"

    # --------------------------------------------------------------- hashing

    def _ep_hash(self) -> int:
        """En passant only counts toward position identity if a capture is actually possible."""
        ep = self.ep_square
        if ep is None:
            return 0
        pawn = PAWN * self.turn
        for s in PAWN_ATTACKS[-self.turn][ep]:
            if self.squares[s] == pawn:
                return _EP_KEYS[ep & 7]
        return 0

    def _compute_hash(self) -> int:
        h = 0
        for sq, p in enumerate(self.squares):
            if p:
                h ^= _PIECE_KEYS[p + 6][sq]
        h ^= _CASTLING_KEYS[self.castling]
        h ^= self._ep_hash()
        if self.turn == BLACK:
            h ^= _TURN_KEY
        return h

    # ------------------------------------------------------------ make/unmake

    def push(self, move: Move) -> None:
        """Play a move. The move is assumed legal (see `legal_moves`)."""
        sq = self.squares
        frm, to = move.from_sq, move.to_sq
        us = self.turn
        piece = sq[frm]
        captured = sq[to]
        kind = abs(piece)
        ep_capture = kind == PAWN and to == self.ep_square

        self._stack.append((move, piece, captured, ep_capture, self.castling,
                            self.ep_square, self.halfmove_clock, self.hash))

        h = self.hash ^ self._ep_hash() ^ _CASTLING_KEYS[self.castling] ^ _TURN_KEY

        h ^= _PIECE_KEYS[piece + 6][frm]
        sq[frm] = EMPTY
        if captured:
            h ^= _PIECE_KEYS[captured + 6][to]

        placed = piece
        new_ep = None
        if kind == PAWN:
            if ep_capture:
                cap_sq = to - 8 * us
                h ^= _PIECE_KEYS[sq[cap_sq] + 6][cap_sq]
                sq[cap_sq] = EMPTY
            elif abs(to - frm) == 16:
                new_ep = (frm + to) // 2
            if move.promotion:
                placed = move.promotion * us
        elif kind == KING:
            self.king_sq[us] = to
            if abs(to - frm) == 2:  # castling: move the rook too
                rook_from, rook_to = (frm + 3, frm + 1) if to > frm else (frm - 4, frm - 1)
                rook = sq[rook_from]
                sq[rook_from] = EMPTY
                sq[rook_to] = rook
                h ^= _PIECE_KEYS[rook + 6][rook_from] ^ _PIECE_KEYS[rook + 6][rook_to]

        sq[to] = placed
        h ^= _PIECE_KEYS[placed + 6][to]

        self.castling &= CASTLING_MASK[frm] & CASTLING_MASK[to]
        self.ep_square = new_ep
        self.halfmove_clock = 0 if (kind == PAWN or captured) else self.halfmove_clock + 1
        if us == BLACK:
            self.fullmove_number += 1
        self.turn = -us

        h ^= _CASTLING_KEYS[self.castling] ^ self._ep_hash()
        self.hash = h
        self.history.append(h)

    def pop(self) -> Move:
        """Undo the last move and return it."""
        move, piece, captured, ep_capture, castling, ep, halfmove, h = self._stack.pop()
        sq = self.squares
        us = -self.turn
        frm, to = move.from_sq, move.to_sq

        sq[frm] = piece
        sq[to] = captured
        if ep_capture:
            sq[to - 8 * us] = -us * PAWN
        if abs(piece) == KING:
            self.king_sq[us] = frm
            if abs(to - frm) == 2:
                rook_from, rook_to = (frm + 3, frm + 1) if to > frm else (frm - 4, frm - 1)
                sq[rook_from] = sq[rook_to]
                sq[rook_to] = EMPTY

        self.turn = us
        self.castling = castling
        self.ep_square = ep
        self.halfmove_clock = halfmove
        if us == BLACK:
            self.fullmove_number -= 1
        self.hash = h
        self.history.pop()
        return move

    @property
    def move_stack(self) -> list[Move]:
        return [entry[0] for entry in self._stack]

    def peek(self) -> Move | None:
        return self._stack[-1][0] if self._stack else None

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b.squares = self.squares[:]
        b.turn = self.turn
        b.castling = self.castling
        b.ep_square = self.ep_square
        b.halfmove_clock = self.halfmove_clock
        b.fullmove_number = self.fullmove_number
        b.king_sq = dict(self.king_sq)
        b._stack = self._stack[:]
        b.hash = self.hash
        b.history = self.history[:]
        return b

    # -------------------------------------------------------------- attacks

    def is_attacked(self, target: int, by: int) -> bool:
        """True if any piece of color `by` attacks square `target`."""
        sq = self.squares
        # A pawn of `by` attacks target iff a pawn of the other color on target would attack it back.
        pawn = PAWN * by
        for s in PAWN_ATTACKS[-by][target]:
            if sq[s] == pawn:
                return True
        knight = KNIGHT * by
        for s in KNIGHT_TARGETS[target]:
            if sq[s] == knight:
                return True
        king = KING * by
        for s in KING_TARGETS[target]:
            if sq[s] == king:
                return True
        rook, queen, bishop = ROOK * by, QUEEN * by, BISHOP * by
        for ray in ROOK_RAYS[target]:
            for s in ray:
                p = sq[s]
                if p:
                    if p == rook or p == queen:
                        return True
                    break
        for ray in BISHOP_RAYS[target]:
            for s in ray:
                p = sq[s]
                if p:
                    if p == bishop or p == queen:
                        return True
                    break
        return False

    def in_check(self) -> bool:
        return self.is_attacked(self.king_sq[self.turn], -self.turn)

    # ------------------------------------------------------ convenience API

    def legal_moves(self) -> list[Move]:
        from .movegen import legal_moves
        return legal_moves(self)

    def outcome(self):
        from .rules import outcome
        return outcome(self)

    def is_game_over(self) -> bool:
        return self.outcome() is not None

    def repetition_count(self) -> int:
        """How many times the current position has occurred (only since the last irreversible move)."""
        window = self.history[-(self.halfmove_clock + 1):]
        return window.count(self.hash)

    def piece_at(self, sq: int) -> int:
        return self.squares[sq]

    def __str__(self) -> str:
        lines = []
        for rank in range(7, -1, -1):
            row = " ".join(piece_symbol(self.squares[rank * 8 + f]) for f in range(8))
            lines.append(f"{rank + 1} {row}")
        lines.append("  a b c d e f g h")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board({self.fen()!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Board) and self.fen() == other.fen()
