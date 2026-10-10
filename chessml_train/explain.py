"""Explanations: an LLM puts what Search found into words (see docs/adr/0004).

The LLM never analyses chess itself. `engine_facts` collects what Search found (candidate
moves, their expected scores and main lines, captures, checks, material); the prompt allows
only those facts; and a guardrail rejects any explanation that names a move not in them,
retrying once and then falling back to a plain template.

    python -m chessml_train.explain --model runs/sl-10x128/best.pt --llm ollama
    python -m chessml_train.explain --model runs/sl-10x128/best.pt --llm hf --fen "<FEN>"

Backends: "ollama" (any OpenAI-compatible server, default model qwen3:8b), "openai" (the
OpenAI API itself; needs OPENAI_API_KEY and --llm-model), and "hf" (Hugging Face
transformers in-process, default Qwen/Qwen3-1.7B).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from chessml import Board, Move, WHITE, move_to_san
from chessml.constants import KING, PAWN

from .mcts import SearchResult

PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
PIECE_VALUES = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
AUDIENCES = {
    "beginner": "a beginner who knows the rules but not much strategy; avoid jargon",
    "club": "a club player around 1500 Elo; chess terms are fine",
}


# ------------------------------------------------------------------ engine facts

def _material(board: Board, side: int) -> int:
    """Material balance in pawns from `side`'s point of view."""
    return sum(PIECE_VALUES[abs(p)] * (1 if p * side > 0 else -1) for p in board.squares if p)


def _move_notes(board: Board, move: Move) -> list[str]:
    """Plain facts about a move in this position (board is left unchanged)."""
    notes = []
    piece = abs(board.squares[move.from_sq])
    target = abs(board.squares[move.to_sq])
    if piece == KING and abs(move.to_sq - move.from_sq) == 2:
        notes.append("castles")
    if target:
        notes.append(f"captures a {PIECE_NAMES[target]}")
    elif piece == PAWN and move.to_sq == board.ep_square:
        notes.append("captures a pawn en passant")
    if move.promotion:
        notes.append(f"promotes to a {PIECE_NAMES[move.promotion]}")
    board.push(move)
    try:
        if board.in_check():
            notes.append("checkmate" if not board.legal_moves() else "gives check")
    finally:
        board.pop()
    return notes


def _line_san(board: Board, line: list[Move]) -> tuple[list[str], Board]:
    b = board.copy()
    sans = []
    for move in line:
        sans.append(move_to_san(b, move))
        b.push(move)
    return sans, b


def engine_facts(board: Board, result: SearchResult, top: int = 4) -> dict:
    """Everything the explanation may rely on, as JSON-ready data."""
    side = board.turn
    order = sorted(range(len(result.moves)), key=lambda i: -result.visits[i])[:top]
    total = float(result.visits.sum()) or 1.0
    candidates = []
    for rank, i in enumerate(order):
        move = result.moves[i]
        line = result.lines[i] if result.lines else [move]
        line_sans, end = _line_san(board, line)
        q = result.q[i]
        candidates.append({
            "move": line_sans[0],
            "engine_choice": rank == 0,
            "expected_score": None if q != q else f"{(q + 1) / 2:.0%}",  # q != q: unvisited (nan)
            "search_effort": f"{result.visits[i] / total:.0%}",
            "policy_prior": f"{result.priors[i]:.0%}",
            "notes": _move_notes(board, move),
            "main_line": line_sans,
            "material_after_main_line": _material(end, side),
        })
    return {
        "side_to_move": "White" if side == WHITE else "Black",
        "fen": board.fen(),
        "material_now": _material(board, side),
        "best_move": candidates[0]["move"],
        "candidates": candidates,
        "notes_on_numbers": ("expected_score is the engine's estimated score for the side to move "
                             "(100% = certain win, 50% = equal); material is in pawns from the "
                             "side to move's point of view; main_line alternates sides."),
    }


# ------------------------------------------------------------------ prompt and guardrail

def build_messages(facts: dict, audience: str = "club") -> list[dict]:
    system = (
        "You are a chess coach explaining a chess engine's move choice to "
        f"{AUDIENCES[audience]}.\n"
        "Rules:\n"
        "- Use ONLY the facts in the JSON you are given. Do not analyse the position yourself.\n"
        "- Only mention moves that appear in the facts, written exactly as given (SAN).\n"
        "- Do not invent threats, tactics or plans that the facts do not show.\n"
        "- Explain why the engine's choice is better than the alternatives, in 3 to 5 sentences.\n"
        "- Plain text, no headings or lists. /no_think"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(facts, indent=2)}]


# Moves that cannot be mistaken for a square name: piece moves, captures, castling, promotions.
# Plain pawn pushes ("e4") look exactly like squares, so they are not checked.
_SAN = re.compile(r"(?<![A-Za-z0-9])(O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x[a-h][1-8](?:=[QRBN])?"
                  r"|[a-h][18]=[QRBN])(?![A-Za-z0-9])")


def mentioned_moves(text: str) -> set[str]:
    return set(_SAN.findall(text))


def unknown_moves(text: str, facts: dict) -> list[str]:
    """Moves named in `text` that are not in the facts (the guardrail)."""
    allowed = {san.rstrip("+#") for c in facts["candidates"] for san in c["main_line"]}
    return sorted(m for m in mentioned_moves(text) if m not in allowed)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def template_explanation(facts: dict) -> str:
    best = facts["candidates"][0]
    parts = [f"The engine plays {best['move']}"]
    if best["notes"]:
        parts[0] += f", which {' and '.join(best['notes'])}"
    parts[0] += f" (expected score {best['expected_score']} for {facts['side_to_move']})."
    if len(best["main_line"]) > 1:
        parts.append(f"Its main line is {' '.join(best['main_line'])}.")
    others = [f"{c['move']} ({c['expected_score']})" for c in facts["candidates"][1:] if c["expected_score"]]
    if others:
        parts.append(f"It searched these alternatives less: {', '.join(others)}.")
    return " ".join(parts)


# ------------------------------------------------------------------ backends

class LLMBackend(Protocol):
    name: str

    def complete(self, messages: list[dict]) -> str: ...


class OpenAICompatibleBackend:
    """Any OpenAI-compatible chat API: Ollama, llama.cpp's server, or OpenAI itself."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.3, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.name = f"{model} via {base_url or 'OpenAI'}"

    def complete(self, messages: list[dict]) -> str:
        reply = self.client.chat.completions.create(model=self.model, messages=messages,
                                                    temperature=self.temperature)
        return reply.choices[0].message.content or ""


class TransformersBackend:
    """A Hugging Face model loaded in this process with transformers (downloaded on first use)."""

    def __init__(self, model: str = "Qwen/Qwen3-1.7B", device: str | None = None,
                 temperature: float = 0.3, max_new_tokens: int = 300):
        self.model_name = model
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.name = f"{model} via transformers"
        self._model = self._tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(device).eval()
        self.device = device

    def complete(self, messages: list[dict]) -> str:
        import torch

        if self._model is None:
            self._load()
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                     enable_thinking=False)  # Qwen3; ignored by others
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                                       temperature=self.temperature, top_p=0.9)
        return self._tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def make_backend(kind: str, model: str | None = None) -> LLMBackend:
    if kind == "ollama":
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
        return OpenAICompatibleBackend(model or "qwen3:8b", base_url=url, api_key="ollama")
    if kind == "openai":
        if not model:
            raise ValueError("--llm openai needs --llm-model (e.g. the OpenAI model you want to use)")
        return OpenAICompatibleBackend(model, api_key=os.environ.get("OPENAI_API_KEY"))
    if kind == "hf":
        return TransformersBackend(model or "Qwen/Qwen3-1.7B")
    raise ValueError(f"unknown LLM backend: {kind}")


# ------------------------------------------------------------------ explain

@dataclass
class Explanation:
    text: str
    facts: dict
    source: str                       # "llm" or "template"
    attempts: int
    rejected: list[str] = field(default_factory=list)  # why earlier attempts were refused


def explain(board: Board, result: SearchResult, backend: LLMBackend, audience: str = "club",
            max_attempts: int = 2) -> Explanation:
    facts = engine_facts(board, result)
    messages = build_messages(facts, audience)
    rejected: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            text = strip_thinking(backend.complete(messages))
        except Exception as e:  # unreachable server, missing model, out of memory...
            rejected.append(f"backend error: {e}")
            break
        bad = unknown_moves(text, facts)
        if text and not bad:
            return Explanation(text, facts, "llm", attempt, rejected)
        rejected.append(f"mentioned moves not in the facts: {', '.join(bad)}" if bad else "empty reply")
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"Your explanation {rejected[-1]}. Rewrite it using only moves "
                                        "that appear in the facts. /no_think"},
        ]
    return Explanation(template_explanation(facts), facts, "template", len(rejected), rejected)


def main(argv: list[str] | None = None) -> None:
    from .player import SearchPlayer, load_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="chess network checkpoint (best.pt)")
    ap.add_argument("--fen", default=None, help="position to explain (default: start position)")
    ap.add_argument("--simulations", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llm", choices=["ollama", "openai", "hf"], default="ollama")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--audience", choices=sorted(AUDIENCES), default="club")
    ap.add_argument("--show-facts", action="store_true")
    args = ap.parse_args(argv)

    board = Board(args.fen) if args.fen else Board()
    search = SearchPlayer(load_model(args.model, args.device), args.simulations).analyse(board)
    result = explain(board, search, make_backend(args.llm, args.llm_model), args.audience)
    if args.show_facts:
        print(json.dumps(result.facts, indent=2))
    print(result.text)
    if result.source == "template":
        print(f"\n(template fallback: {'; '.join(result.rejected)})")


if __name__ == "__main__":
    main()
