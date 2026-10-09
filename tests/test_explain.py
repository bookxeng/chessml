import pytest

from chessml import Board, Move
from chessml_train.explain import (
    OpenAICompatibleBackend, TransformersBackend, engine_facts, explain, make_backend,
    mentioned_moves, strip_thinking, unknown_moves,
)
from chessml_train.mcts import MCTS
from test_mcts import material, uniform

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
QUEEN_TRAP = "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1"


class FakeBackend:
    name = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def search(fen, evaluator=uniform, sims=200):
    board = Board(fen)
    return board, MCTS(evaluator, batch_size=8).search(board, sims)


def test_facts_describe_the_mate():
    facts = engine_facts(*search(MATE_IN_ONE))
    best = facts["candidates"][0]
    assert facts["side_to_move"] == "White" and facts["best_move"] == "Ra8#"
    assert best["engine_choice"] and "checkmate" in best["notes"] and best["expected_score"] == "100%"


def test_facts_lines_are_san_and_material_is_tracked():
    facts = engine_facts(*search(QUEEN_TRAP, material, 400))
    assert facts["material_now"] == 7  # queen (9) vs two pawns (2)
    assert facts["best_move"] != "Qxd5"
    for c in facts["candidates"]:
        assert c["main_line"][0] == c["move"]


def test_mentioned_moves_ignores_plain_squares():
    text = "Nf3 controls e5 and d4; after Bxf7+ and O-O, exd5 or e8=Q are possible, but e4 is a square."
    assert mentioned_moves(text) == {"Nf3", "Bxf7", "O-O", "exd5", "e8=Q"}


def test_guardrail_flags_invented_moves():
    facts = engine_facts(*search(MATE_IN_ONE))
    assert unknown_moves("Ra8# wins at once.", facts) == []
    assert unknown_moves("Ra8# wins, but Qh5 was also good.", facts) == ["Qh5"]


def test_strip_thinking():
    assert strip_thinking("<think>hmm\nlet me see</think>\nRa8# mates.") == "Ra8# mates."


def test_explain_accepts_a_grounded_reply():
    backend = FakeBackend("<think>x</think>Ra8# is checkmate along the back rank.")
    result = explain(*search(MATE_IN_ONE), backend)
    assert result.source == "llm" and result.attempts == 1
    assert result.text == "Ra8# is checkmate along the back rank."
    system = backend.calls[0][0]["content"]
    assert "ONLY the facts" in system and '"best_move": "Ra8#"' in backend.calls[0][1]["content"]


def test_explain_retries_after_an_invented_move():
    backend = FakeBackend("Qh5 is strong here.", "Ra8# ends the game immediately.")
    result = explain(*search(MATE_IN_ONE), backend)
    assert result.source == "llm" and result.attempts == 2
    assert "Qh5" in result.rejected[0]
    assert "Rewrite it" in backend.calls[1][-1]["content"]


def test_explain_falls_back_to_the_template():
    result = explain(*search(MATE_IN_ONE), FakeBackend("Qh5!", "Still Qh5."))
    assert result.source == "template" and result.text.startswith("The engine plays Ra8#")
    result = explain(*search(MATE_IN_ONE), FakeBackend(ConnectionError("no server")))
    assert result.source == "template" and "backend error" in result.rejected[0]


def test_openai_compatible_backend_sends_the_messages():
    sent = {}

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    sent.update(kwargs)
                    msg = type("M", (), {"content": "ok"})
                    return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    backend = OpenAICompatibleBackend("qwen3:8b", base_url="http://x", client=Client)
    assert backend.complete([{"role": "user", "content": "hi"}]) == "ok"
    assert sent["model"] == "qwen3:8b" and sent["messages"][0]["content"] == "hi"


def test_make_backend():
    assert make_backend("ollama").model == "qwen3:8b"
    hf = make_backend("hf")
    assert isinstance(hf, TransformersBackend) and hf._model is None  # nothing downloaded yet
    with pytest.raises(ValueError):
        make_backend("openai")
