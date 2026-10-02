import numpy as np
import pytest

from chessml import Board, action_to_move
from chessml_train.lichess import GameFilter, build_dataset, iter_games
from chessml_train.pgn import game_positions, movetext_sans

GAME = """[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "2201"]
[BlackElo "2174"]
[Termination "Normal"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Qh5 $2 2... Nc6?! 3. Bc4 Nf6?? 4. Qxf7# 1-0

"""


def test_movetext_sans_strips_comments_numbers_nags_and_result():
    assert movetext_sans("1. e4 { [%clk 0:03:00] } 1... e5 2. Qh5 $2 2... Nc6?! 1-0") == ["e4", "e5", "Qh5", "Nc6?!"]


def test_game_positions_replay_the_game():
    headers, movetext = next(iter_games(GAME.splitlines(True)))
    records = game_positions(movetext_sans(movetext), headers["Result"])
    assert len(records) == 7
    assert list(records["result"]) == [1, -1, 1, -1, 1, -1, 1]
    board = Board()
    for rec in records:
        np.testing.assert_array_equal(rec["squares"], board.squares)
        assert (rec["turn"], rec["castling"]) == (board.turn, board.castling)
        move = action_to_move(int(rec["action"]), board)
        assert move in board.legal_moves()
        board.push(move)
    assert board.outcome().winner == 1


def test_game_positions_rejects_illegal_moves():
    with pytest.raises(ValueError):
        game_positions(["e4", "e4"], "1-0")


@pytest.mark.parametrize("headers, accepted", [
    ({}, True),
    ({"Event": "Rated Bullet game"}, False),
    ({"Event": "Rated Rapid tournament https://lichess.org/tournament/x"}, True),
    ({"BlackElo": "1999"}, False),
    ({"WhiteElo": "?"}, False),
    ({"Termination": "Abandoned"}, False),
    ({"Result": "*"}, False),
    ({"FEN": "8/8/8/8/8/8/8/K6k w - - 0 1"}, False),
])
def test_filter(headers, accepted):
    base = {"Event": "Rated Blitz game", "Result": "1-0", "WhiteElo": "2201",
            "BlackElo": "2174", "Termination": "Normal"}
    assert GameFilter().accepts(base | headers) is accepted


def test_build_dataset_splits_by_game(tmp_path):
    bullet = GAME.replace("Rated Blitz", "Rated Bullet")
    broken = GAME.replace("Qxf7#", "Qxf8")
    lines = (GAME * 60 + bullet + broken + GAME * 60).splitlines(True)
    manifest = build_dataset(lines, tmp_path, target_games=100, game_filter=GameFilter(),
                             workers=1, games_per_shard=30, chunk_size=7, log=lambda _: None)
    assert manifest["games"] == 100 and manifest["parse_errors"] == 1
    assert manifest["val"]["games"] == 2 and manifest["train"]["games"] == 98
    train = np.concatenate([np.load(p) for p in sorted(tmp_path.glob("train-*.npy"))])
    val = np.concatenate([np.load(p) for p in sorted(tmp_path.glob("val-*.npy"))])
    assert len(train) == 98 * 7 and len(val) == 2 * 7
