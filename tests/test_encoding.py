import random

import numpy as np

from chessml import (
    Board, Move, NUM_ACTIONS, OBS_SHAPE, encode_board, move_to_action, action_to_move, legal_action_mask,
)


def random_positions(n_games=30, max_plies=150, seed=0):
    rng = random.Random(seed)
    for _ in range(n_games):
        board = Board()
        for _ in range(max_plies):
            moves = board.legal_moves()
            if not moves:
                break
            yield board, moves
            board.push(rng.choice(moves))


def test_action_round_trip_and_uniqueness():
    for board, moves in random_positions():
        actions = [move_to_action(m, board) for m in moves]
        assert len(set(actions)) == len(moves)
        for m, a in zip(moves, actions):
            assert 0 <= a < NUM_ACTIONS
            assert action_to_move(a, board) == m


def test_action_round_trip_with_promotions_and_castling():
    for fen in [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "1n2k3/P7/8/8/8/8/7p/4K1N1 w - - 0 1",
        "1n2k3/P7/8/8/8/8/7p/4K1N1 b - - 0 1",
    ]:
        board = Board(fen)
        moves = board.legal_moves()
        for m in moves:
            assert action_to_move(move_to_action(m, board), board) == m
        mask = legal_action_mask(board)
        assert mask.sum() == len(moves)


def test_mask_matches_legal_moves():
    for board, moves in random_positions(n_games=10):
        mask = legal_action_mask(board)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.sum() == len(moves)


def test_black_perspective_is_mirrored():
    # The position after 1.e4 e5 is color-symmetric, so white-to-move and black-to-move
    # must look identical to the network (apart from the side-to-move plane).
    placement = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
    white_view = encode_board(Board(f"{placement} w KQkq - 0 2"))
    black_view = encode_board(Board(f"{placement} b KQkq - 0 2"))
    assert white_view.shape == black_view.shape == OBS_SHAPE
    np.testing.assert_array_equal(np.delete(white_view, 12, axis=0), np.delete(black_view, 12, axis=0))
    assert white_view[12].all() and not black_view[12].any()


def test_same_move_same_action_for_both_colors():
    # 1.e4 for white and 1...e5 for black are the same move from each side's perspective.
    white = Board()
    black = Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    assert move_to_action(Move.from_uci("e2e4"), white) == move_to_action(Move.from_uci("e7e5"), black)


def test_observation_contents():
    obs = encode_board(Board())
    assert obs.dtype == np.float32
    assert obs[0].sum() == 8          # our pawns
    assert obs[0, 1].sum() == 8       # on our second rank
    assert obs[6, 6].sum() == 8       # their pawns on rank 7
    assert obs[5, 0, 4] == 1          # our king on e1
    assert obs[13:17].all()           # all castling rights
    assert not obs[17].any()          # no en passant
