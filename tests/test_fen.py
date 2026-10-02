import random

import pytest

from chessml import Board, STARTING_FEN

FENS = [
    STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
    "8/8/8/8/8/8/8/K6k b - - 42 100",
]


@pytest.mark.parametrize("fen", FENS)
def test_round_trip(fen):
    assert Board(fen).fen() == fen


@pytest.mark.parametrize("fen", [
    "",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",          # 7 ranks
    "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # 9 files
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w KQkq - 0 1",  # no white king
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",  # bad side
])
def test_invalid_fen(fen):
    with pytest.raises(ValueError):
        Board(fen)


def test_push_pop_restores_everything():
    rng = random.Random(1)
    board = Board()
    snapshots = []
    for _ in range(200):
        moves = board.legal_moves()
        if not moves:
            break
        snapshots.append((board.fen(), board.hash))
        board.push(rng.choice(moves))
    while snapshots:
        board.pop()
        assert (board.fen(), board.hash) == snapshots.pop()


def test_incremental_hash_matches_full_recompute():
    rng = random.Random(2)
    board = Board()
    for _ in range(300):
        moves = board.legal_moves()
        if not moves:
            break
        board.push(rng.choice(moves))
        assert board.hash == Board(board.fen()).hash


def test_copy_is_independent():
    a = Board()
    b = a.copy()
    b.push(b.legal_moves()[0])
    assert a.fen() == STARTING_FEN
    b.pop()
    assert b == a
