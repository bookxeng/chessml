import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chessml import Board, NUM_ACTIONS
from chessml.players import RandomPlayer
from chessml_train.arena import match
from chessml_train.data import ShardStream, wdl_targets
from chessml_train.lichess import GameFilter, build_dataset
from chessml_train.model import ChessNet, NetConfig, flatten_policy
from chessml_train.player import PolicyPlayer, load_model
from chessml_train.train import TrainConfig, train

GAME = """[Event "Rated Blitz game"]
[Result "1-0"]
[WhiteElo "2201"]
[BlackElo "2174"]
[Termination "Normal"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0

"""
TINY = NetConfig(blocks=1, filters=8, value_channels=2, value_hidden=8)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    build_dataset((GAME * 120).splitlines(True), out, target_games=120, game_filter=GameFilter(),
                  workers=1, games_per_shard=25, log=lambda _: None)
    return out


def test_flatten_policy_matches_action_layout():
    planes = torch.arange(73 * 64, dtype=torch.float32).reshape(1, 73, 8, 8)
    flat = flatten_policy(planes)[0]
    for square, plane in [(0, 0), (12, 5), (63, 72), (28, 64)]:
        assert flat[square * 73 + plane] == planes[0, plane, square // 8, square % 8]


def test_network_output_shapes():
    policy, value = ChessNet(TINY)(torch.zeros(5, 19, 8, 8))
    assert policy.shape == (5, NUM_ACTIONS) and value.shape == (5, 3)


def test_wdl_targets():
    assert list(wdl_targets(np.array([1, 0, -1], dtype=np.int8))) == [0, 1, 2]


def test_shard_stream_covers_each_position_once_and_resumes(dataset):
    stream = ShardStream(list(dataset.glob("train-*.npy")), batch_size=10, shards_in_memory=2, seed=3)
    total = sum(len(np.load(p)) for p in dataset.glob("train-*.npy"))
    batches = list(stream.batches(epoch=0))
    assert len(batches) == total // 10
    resumed = list(stream.batches(epoch=0, skip=5))
    np.testing.assert_array_equal(resumed[0][1], batches[5][1])
    assert not np.array_equal(list(stream.batches(epoch=1))[0][1], batches[0][1])


def test_train_resumes_and_player_plays_legal_moves(dataset, tmp_path):
    cfg = dict(data=str(dataset), out=str(tmp_path), blocks=1, filters=8, batch_size=16,
               epochs=1, warmup_steps=2, log_every=2, eval_every=3, max_steps=3)
    assert train(TrainConfig(**cfg), log=lambda _: None)["step"] == 3
    assert train(TrainConfig(**{**cfg, "max_steps": 5}), log=lambda _: None)["step"] == 5
    steps = [json.loads(l)["step"] for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert steps == sorted(steps) and steps[-1] == 5

    player = PolicyPlayer(load_model(tmp_path / "best.pt"))
    board = Board()
    probs, wdl = player.evaluate(board, board.legal_moves())
    assert probs.sum() == pytest.approx(1, abs=1e-4) and wdl.sum() == pytest.approx(1, abs=1e-4)
    assert player.choose(board) in board.legal_moves()
    assert match(player, RandomPlayer(0), games=2, max_plies=20).games == 2


def test_play_session_handles_commands(dataset, tmp_path):
    from chessml_train.play import play

    train(TrainConfig(data=str(dataset), out=str(tmp_path), blocks=1, filters=8, batch_size=16,
                      epochs=1, eval_every=2, max_steps=2), log=lambda _: None)
    inputs = iter(["e4", "hint", "nonsense", "moves", "flip", "undo", "d4", "quit"])
    lines = []
    assert play(str(tmp_path / "best.pt"), input_fn=lambda _: next(inputs), color=False,
                output=lines.append) is None
    text = "\n".join(lines)
    assert "The model would play" in text and "unrecognized move" in text and "Model plays" in text


def test_play_session_with_search(dataset, tmp_path):
    from chessml_train.play import play

    train(TrainConfig(data=str(dataset), out=str(tmp_path), blocks=1, filters=8, batch_size=16,
                      epochs=1, eval_every=2, max_steps=2), log=lambda _: None)
    inputs = iter(["e4", "hint", "quit"])
    lines = []
    play(str(tmp_path / "best.pt"), simulations=16, input_fn=lambda _: next(inputs), color=False,
         output=lines.append)
    text = "\n".join(lines)
    assert "Search simulations" in text and "expected score" in text
