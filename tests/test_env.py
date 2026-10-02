import numpy as np
import pytest

from chessml import ChessEnv, NUM_ACTIONS, OBS_SHAPE, Move


def test_reset():
    env = ChessEnv()
    obs, info = env.reset()
    assert obs.shape == OBS_SHAPE
    assert info["legal_mask"].shape == (NUM_ACTIONS,)
    assert info["legal_mask"].sum() == 20
    assert info["outcome"] is None


def test_random_games_terminate():
    rng = np.random.default_rng(0)
    env = ChessEnv(max_plies=400)
    results = {}
    for _ in range(200):
        obs, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = rng.choice(np.flatnonzero(info["legal_mask"]))
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == OBS_SHAPE
            if terminated:
                out = info["outcome"]
                assert out is not None
                assert reward == (1.0 if out.winner is not None else 0.0)
                results[out.reason] = results.get(out.reason, 0) + 1
            else:
                assert reward == 0.0
    assert sum(results.values()) > 0


def test_checkmate_reward():
    env = ChessEnv()
    env.reset("7k/8/6K1/8/8/8/8/R7 w - - 0 1")
    _, reward, terminated, truncated, info = env.step_move(Move.from_uci("a1a8"))
    assert terminated and not truncated
    assert reward == 1.0
    assert info["outcome"].reason == "checkmate"
    with pytest.raises(RuntimeError):
        env.step_move(Move.from_uci("h8g8"))


def test_illegal_action_raises():
    env = ChessEnv()
    env.reset()
    illegal = int(np.flatnonzero(~env.legal_action_mask())[0])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_truncation():
    env = ChessEnv(max_plies=4)
    env.reset()
    truncated = False
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(env.legal_actions()[0])
    assert truncated and not terminated
    assert env.done


def test_legal_actions_match_mask():
    env = ChessEnv()
    env.reset()
    np.testing.assert_array_equal(np.sort(env.legal_actions()), np.flatnonzero(env.legal_action_mask()))
