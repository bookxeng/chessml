import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from chessml import BLACK, WHITE
from chessml_train.model import ChessNet, NetConfig
from chessml_train.session import GameSession
from chessml_train.web import create_app

TINY = NetConfig(blocks=1, filters=8, value_channels=2, value_hidden=8)
PROMOTION = "8/P6k/8/8/8/8/8/K7 w - - 0 1"


class FakeLLM:
    name = "fake-llm"

    def complete(self, messages):
        return f"{json.loads(messages[1]['content'])['best_move']} is best."


@pytest.fixture
def game():
    return GameSession(ChessNet(TINY).eval(), simulations=16, llm=FakeLLM())


@pytest.fixture
def client(game):
    return TestClient(create_app(game))


def test_session_play_mode_turns_and_undo(game):
    game.new_game(WHITE)
    assert not game.model_to_move
    game.play("e4")
    assert game.model_to_move
    with pytest.raises(ValueError):
        game.play("e5")  # not the human's turn
    decision = game.model_move()
    assert decision.search is not None and decision.position.fen() != game.board.fen()
    assert len(game.sans) == 2 and not game.model_to_move
    assert game.undo() == 2 and game.sans == [] and game.last_decision is None


def test_session_analysis_mode_moves_both_sides(game):
    game.new_game(None)
    for move in ("e4", "e5", "Nf3"):
        game.play(move)
    assert game.mode == "analysis" and not game.model_to_move
    assert game.undo() == 1 and game.sans == ["e4", "e5"]


def test_session_raw_policy_and_explain_rules(game):
    game.set_simulations(0)
    game.new_game(BLACK)
    d = game.model_move()
    assert d.search is None
    with pytest.raises(ValueError):
        game.explain_last()  # raw policy has no search to explain
    with pytest.raises(ValueError):
        game.analyse()


def test_index_and_static(client):
    assert "chessml" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_state_at_start(client):
    s = client.get("/api/state").json()
    assert s["turn"] == "white" and s["mode"] == "play" and len(s["legal"]) == 20
    assert abs(sum(s["eval"].values()) - 1) < 1e-4 and s["llm"] == "fake-llm"


def test_play_engine_move_undo_flow(client):
    s = client.post("/api/move", json={"move": "e2e4"}).json()
    assert s["san"] == "e4" and s["state"]["model_to_move"]
    r = client.post("/api/engine-move").json()
    assert r["state"]["turn"] == "white" and len(r["state"]["sans"]) == 2
    assert r["search"]["candidates"][0]["engine_choice"]
    assert client.post("/api/undo", json={}).json()["sans"] == []


def test_illegal_move_is_a_400(client):
    r = client.post("/api/move", json={"move": "e2e5"})
    assert r.status_code == 400 and "illegal" in r.json()["detail"]


def test_analysis_board_with_fen_and_promotion(client):
    s = client.post("/api/new", json={"mode": "analysis", "fen": PROMOTION}).json()
    promos = [m["uci"] for m in s["legal"] if m["from"] == "a7"]
    assert sorted(promos) == ["a7a8b", "a7a8n", "a7a8q", "a7a8r"]
    r = client.post("/api/analyse", json={}).json()
    assert r["search"]["best"] in {m["uci"] for m in s["legal"]}
    s = client.post("/api/move", json={"move": "a7a8n"}).json()["state"]
    assert s["sans"] == ["a8=N"] and s["turn"] == "black"


def test_bad_fen_is_a_400(client):
    assert client.post("/api/new", json={"mode": "analysis", "fen": "not a fen"}).status_code == 400


def test_new_game_as_black_waits_for_model(client):
    s = client.post("/api/new", json={"mode": "play", "color": "black"}).json()
    assert s["human"] == "black" and s["model_to_move"]


def test_explain_last_and_position(client):
    assert client.post("/api/explain", json={"target": "last"}).status_code == 400  # model hasn't moved
    client.post("/api/move", json={"move": "d2d4"})
    client.post("/api/engine-move")
    x = client.post("/api/explain", json={"target": "last"}).json()["explanation"]
    assert x["source"] == "llm" and x["text"].endswith("is best.")
    client.post("/api/new", json={"mode": "analysis"})
    x = client.post("/api/explain", json={"target": "position"}).json()["explanation"]
    assert x["move"] == x["facts"]["best_move"]


def test_settings(client):
    s = client.post("/api/settings", json={"simulations": 0, "audience": "beginner"}).json()
    assert s["simulations"] == 0 and s["audience"] == "beginner"
    assert client.post("/api/settings", json={"audience": "grandmaster"}).status_code == 400
