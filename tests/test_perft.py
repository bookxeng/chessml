import pytest

from chessml import Board, perft

# Reference node counts from https://www.chessprogramming.org/Perft_Results
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
POS3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
POS4 = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
POS4_MIRRORED = "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"
POS5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POS6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

FAST = [
    (START, [20, 400, 8902, 197281]),
    (KIWIPETE, [48, 2039, 97862]),
    (POS3, [14, 191, 2812, 43238]),
    (POS4, [6, 264, 9467]),
    (POS4_MIRRORED, [6, 264, 9467]),
    (POS5, [44, 1486, 62379]),
    (POS6, [46, 2079, 89890]),
]

SLOW = [
    (START, 5, 4865609),
    (KIWIPETE, 4, 4085603),
    (POS3, 5, 674624),
    (POS4, 4, 422333),
    (POS5, 4, 2103487),
]


@pytest.mark.parametrize("fen,counts", FAST)
def test_perft(fen, counts):
    board = Board(fen)
    for depth, expected in enumerate(counts, start=1):
        assert perft(board, depth) == expected, f"depth {depth}"
    assert board.fen() == fen, "push/pop must restore the position exactly"


@pytest.mark.slow
@pytest.mark.parametrize("fen,depth,expected", SLOW)
def test_perft_deep(fen, depth, expected):
    assert perft(Board(fen), depth) == expected
