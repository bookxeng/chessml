"""Piece codes, colors, square helpers and precomputed move tables.

Squares are integers 0..63 with a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
Pieces are signed ints: positive for white, negative for black, 0 for empty.
"""

WHITE = 1
BLACK = -1

EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

PIECE_SYMBOLS = ".pnbrqk"  # index by piece type
PROMOTION_PIECES = (QUEEN, ROOK, BISHOP, KNIGHT)

# Castling-right bits.
WHITE_KINGSIDE = 1
WHITE_QUEENSIDE = 2
BLACK_KINGSIDE = 4
BLACK_QUEENSIDE = 8
ALL_CASTLING = 15

FILE_NAMES = "abcdefgh"
RANK_NAMES = "12345678"


def square(file: int, rank: int) -> int:
    return rank * 8 + file


def square_file(sq: int) -> int:
    return sq & 7


def square_rank(sq: int) -> int:
    return sq >> 3


def square_name(sq: int) -> str:
    return FILE_NAMES[sq & 7] + RANK_NAMES[sq >> 3]


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in FILE_NAMES or name[1] not in RANK_NAMES:
        raise ValueError(f"invalid square: {name!r}")
    return square(FILE_NAMES.index(name[0]), RANK_NAMES.index(name[1]))


def piece_symbol(piece: int) -> str:
    """'P' for a white pawn, 'n' for a black knight, '.' for empty."""
    s = PIECE_SYMBOLS[abs(piece)]
    return s.upper() if piece > 0 else s


def _on_board(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def _step_targets(deltas):
    table = []
    for sq in range(64):
        f, r = square_file(sq), square_rank(sq)
        table.append(tuple(square(f + df, r + dr) for df, dr in deltas if _on_board(f + df, r + dr)))
    return tuple(table)


def _rays(directions):
    table = []
    for sq in range(64):
        f, r = square_file(sq), square_rank(sq)
        rays = []
        for df, dr in directions:
            ray = []
            nf, nr = f + df, r + dr
            while _on_board(nf, nr):
                ray.append(square(nf, nr))
                nf, nr = nf + df, nr + dr
            if ray:
                rays.append(tuple(ray))
        table.append(tuple(rays))
    return tuple(table)


KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_DELTAS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
ROOK_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
BISHOP_DIRECTIONS = ((1, 1), (1, -1), (-1, -1), (-1, 1))

KNIGHT_TARGETS = _step_targets(KNIGHT_DELTAS)
KING_TARGETS = _step_targets(KING_DELTAS)
ROOK_RAYS = _rays(ROOK_DIRECTIONS)
BISHOP_RAYS = _rays(BISHOP_DIRECTIONS)

# PAWN_ATTACKS[color][sq]: squares a pawn of `color` standing on `sq` attacks.
PAWN_ATTACKS = {
    WHITE: _step_targets(((-1, 1), (1, 1))),
    BLACK: _step_targets(((-1, -1), (1, -1))),
}

# Castling rights that survive a move touching the given square (from or to).
CASTLING_MASK = [ALL_CASTLING] * 64
CASTLING_MASK[0] &= ~WHITE_QUEENSIDE   # a1
CASTLING_MASK[7] &= ~WHITE_KINGSIDE    # h1
CASTLING_MASK[4] &= ~(WHITE_KINGSIDE | WHITE_QUEENSIDE)   # e1
CASTLING_MASK[56] &= ~BLACK_QUEENSIDE  # a8
CASTLING_MASK[63] &= ~BLACK_KINGSIDE   # h8
CASTLING_MASK[60] &= ~(BLACK_KINGSIDE | BLACK_QUEENSIDE)  # e8
CASTLING_MASK = tuple(CASTLING_MASK)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
