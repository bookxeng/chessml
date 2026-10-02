from __future__ import annotations

from dataclasses import dataclass

from .constants import PIECE_SYMBOLS, parse_square, square_name


@dataclass(frozen=True, slots=True)
class Move:
    """A move from one square to another, with an optional promotion piece type."""

    from_sq: int
    to_sq: int
    promotion: int = 0  # 0, or KNIGHT/BISHOP/ROOK/QUEEN

    def uci(self) -> str:
        s = square_name(self.from_sq) + square_name(self.to_sq)
        if self.promotion:
            s += PIECE_SYMBOLS[self.promotion]
        return s

    @classmethod
    def from_uci(cls, text: str) -> Move:
        text = text.strip().lower()
        if len(text) not in (4, 5):
            raise ValueError(f"invalid UCI move: {text!r}")
        promotion = 0
        if len(text) == 5:
            if text[4] not in "nbrq":
                raise ValueError(f"invalid promotion piece in {text!r}")
            promotion = PIECE_SYMBOLS.index(text[4])
        return cls(parse_square(text[:2]), parse_square(text[2:4]), promotion)

    def __str__(self) -> str:
        return self.uci()
