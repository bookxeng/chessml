import random

import numpy as np

from chessml import Board, encode_board
from chessml_train.positions import POSITION_DTYPE, encode_batch, position_record

FENS = [
    "r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R b Kq - 4 9",  # black to move, mixed rights
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",    # white ep capture
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",    # black ep capture
    "8/P6k/8/8/8/8/8/K7 w - - 87 120",                                   # high halfmove clock
]


def boards():
    for fen in FENS:
        yield Board(fen)
    rng = random.Random(0)
    for _ in range(20):
        board = Board()
        for _ in range(rng.randrange(1, 120)):
            moves = board.legal_moves()
            if not moves:
                break
            yield board.copy()
            board.push(rng.choice(moves))


def test_encode_batch_matches_encode_board():
    positions = [b for b in boards() if b.legal_moves()]
    records = np.array([position_record(b, b.legal_moves()[0], 1) for b in positions], dtype=POSITION_DTYPE)
    batch = encode_batch(records)
    assert batch.shape == (len(positions), 19, 8, 8) and batch.dtype == np.float32
    for board, planes in zip(positions, batch):
        np.testing.assert_array_equal(planes, encode_board(board), err_msg=board.fen())


def test_result_is_from_side_to_move():
    white = Board()
    black = Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    move_w, move_b = white.legal_moves()[0], black.legal_moves()[0]
    assert position_record(white, move_w, 1)[-1] == 1
    assert position_record(black, move_b, 1)[-1] == -1
    assert position_record(black, move_b, 0)[-1] == 0
