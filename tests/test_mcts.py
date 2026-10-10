import numpy as np
import pytest

from chessml import Board, Move, WHITE
from chessml_train.mcts import MCTS

PIECE_VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def uniform(obs, actions):
    return [np.full(len(a), 1 / len(a)) for a in actions], np.zeros(len(actions))


def material(obs, actions):
    """Uniform priors; value = squashed material balance for the side to move (planes 0-5 ours, 6-11 theirs)."""
    weights = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)
    ours = (obs[:, 0:6].sum(axis=(2, 3)) * weights).sum(1)
    theirs = (obs[:, 6:12].sum(axis=(2, 3)) * weights).sum(1)
    return [np.full(len(a), 1 / len(a)) for a in actions], np.tanh((ours - theirs) / 5)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("fen, mate", [
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8"),   # back-rank mate, white
    ("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", "a8a1"),   # mirrored, black to move
])
def test_finds_mate_in_one(fen, mate, batch_size):
    result = MCTS(uniform, batch_size=batch_size).search(Board(fen), 200)
    assert result.best == Move.from_uci(mate)
    assert result.q[result.moves.index(Move.from_uci(mate))] == pytest.approx(1.0)


def test_does_not_hang_the_queen_for_a_defended_pawn():
    # Qxd5 wins a pawn but exd5 wins the queen; a material Value sees that two plies deep.
    board = Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")
    result = MCTS(material, batch_size=8).search(board, 400)
    qxd5 = result.moves.index(Move.from_uci("d1d5"))
    assert result.best != Move.from_uci("d1d5")
    assert result.visits[qxd5] < result.visits.mean() / 2  # search stops spending effort on it


def test_policy_prior_steers_visits():
    board = Board()
    favourite = board.legal_moves()[3]

    def biased(obs, actions):
        priors = []
        for a in actions:
            p = np.full(len(a), 0.1 / (len(a) - 1))
            p[3] = 0.9
            priors.append(p)
        return priors, np.zeros(len(actions))

    result = MCTS(biased).search(board, 100)
    assert result.best == favourite


def test_visits_add_up_and_board_is_unchanged():
    board = Board()
    fen = board.fen()
    result = MCTS(uniform, batch_size=16).search(board, 300)
    assert result.simulations == 300
    assert result.visits.sum() == 300  # expanding the root itself is not a simulation
    assert board.fen() == fen and len(board.move_stack) == 0


def test_rejects_finished_game():
    with pytest.raises(ValueError):
        MCTS(uniform).search(Board("R5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"), 10)


def test_greedy_player_takes_free_material_and_avoids_hanging():
    from chessml.players import GreedyPlayer

    greedy = GreedyPlayer(0)
    assert greedy.choose(Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")) == Move.from_uci("d1d5")  # free queen
    assert greedy.choose(Board("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1")) != Move.from_uci("d1d5")  # defended pawn
    assert greedy.choose(Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")) == Move.from_uci("a1a8")  # mate in one


def test_match_pairs_openings_and_records_endings():
    from chessml.players import RandomPlayer
    from chessml_train.arena import match

    result = match(RandomPlayer(1), RandomPlayer(2), games=4, max_plies=30, opening_plies=4)
    assert result.games == 4 and sum(result.reasons.values()) == 4


def test_stockfish_player_plays_legal_moves():
    from chessml_train.stockfish import StockfishPlayer, find_stockfish

    path = find_stockfish()
    if path is None:
        pytest.skip("Stockfish not installed")
    with StockfishPlayer(path, elo=1400, movetime_ms=20) as sf:
        board = Board()
        assert sf.choose(board) in board.legal_moves()


def test_main_line_is_a_legal_continuation_and_finds_the_mate():
    result = MCTS(uniform, batch_size=8).search(Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"), 300)
    assert result.main_line[0] == Move.from_uci("a1a8")
    board = Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    result = MCTS(material, batch_size=8).search(board, 400)
    assert len(result.lines) == len(result.moves)
    for line in result.lines:
        b = board.copy()
        for move in line:
            assert move in b.legal_moves()
            b.push(move)
    assert len(result.main_line) >= 2
