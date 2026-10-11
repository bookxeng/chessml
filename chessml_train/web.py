"""Web app: play the model, analyse positions, and ask the LLM to explain, in the browser.

    python -m chessml_train.web --model runs/sl-10x128/best.pt --device cuda --llm ollama

Then open http://localhost:8000. One GameSession lives in the server (a local, single-user
app); every endpoint returns the full game state so the page simply re-renders it.
"""
from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chessml import WHITE, BLACK
from chessml.constants import square_name

from .explain import AUDIENCES, Explanation, engine_facts, make_backend
from .mcts import SearchResult
from .session import GameSession

STATIC = Path(__file__).parent / "static"


class NewGame(BaseModel):
    mode: str = "play"           # "play" or "analysis"
    color: str = "white"         # the human's colour in play mode
    fen: str | None = None


class MoveIn(BaseModel):
    move: str                    # UCI ("e2e4", "e7e8q") or SAN


class ExplainIn(BaseModel):
    target: str = "last"         # "last" = the model's last move, "position" = the engine's choice here


class Settings(BaseModel):
    simulations: int | None = None
    audience: str | None = None


def _colour(c: int | None) -> str | None:
    return None if c is None else ("white" if c == WHITE else "black")


def state(game: GameSession) -> dict:
    board = game.board
    legal = board.legal_moves() if game.outcome is None else []
    last = board.move_stack[-1] if board.move_stack else None
    if game.outcome is None:
        _, _, wdl = game.evaluate()
        w, d, l = (wdl if board.turn == WHITE else wdl[::-1]).tolist()
    else:
        w, d, l = {WHITE: (1, 0, 0), BLACK: (0, 0, 1), None: (0, 1, 0)}[game.outcome.winner]
    outcome = game.outcome
    return {
        "mode": game.mode,
        "human": _colour(game.human),
        "turn": _colour(board.turn),
        "fen": board.fen(),
        "pieces": list(board.squares),               # index 0 = a1, positive = white
        "sans": game.sans,
        "start_fen": game.start_fen,
        "last_move": None if last is None else {"from": square_name(last.from_sq), "to": square_name(last.to_sq)},
        "legal": [{"from": square_name(m.from_sq), "to": square_name(m.to_sq), "uci": m.uci()} for m in legal],
        "in_check": board.in_check(),
        "outcome": None if outcome is None else {"result": outcome.result, "reason": outcome.reason},
        "model_to_move": game.model_to_move,
        "eval": {"white": w, "draw": d, "black": l},  # the network's Value, from white's point of view
        "simulations": game.simulations,
        "llm": game.llm.name if game.llm else None,
        "audience": game.audience,
    }


def search_json(board, result: SearchResult) -> dict:
    facts = engine_facts(board, result, top=5)
    return {"simulations": result.simulations, "best": result.best.uci(),
            "candidates": facts["candidates"]}


def explanation_json(e: Explanation, move: str) -> dict:
    return {"move": move, "text": e.text, "source": e.source, "attempts": e.attempts,
            "rejected": e.rejected, "facts": e.facts}


def create_app(game: GameSession) -> FastAPI:
    app = FastAPI(title="chessml")
    lock = threading.Lock()  # Search and LLM calls are slow; keep requests from interleaving

    def guarded(fn):
        with lock:
            try:
                return fn()
            except ValueError as e:
                raise HTTPException(400, str(e)) from None

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def get_state():
        return guarded(lambda: state(game))

    @app.post("/api/new")
    def new_game(body: NewGame):
        def run():
            if body.mode not in ("play", "analysis") or body.color not in ("white", "black"):
                raise ValueError("mode must be play/analysis and color white/black")
            human = None if body.mode == "analysis" else (WHITE if body.color == "white" else BLACK)
            try:
                game.new_game(human, body.fen or None)
            except Exception as e:  # malformed FEN
                game.new_game(human)
                raise ValueError(f"invalid FEN: {e}") from None
            return state(game)
        return guarded(run)

    @app.post("/api/move")
    def move(body: MoveIn):
        def run():
            san = game.play(body.move)
            return {"san": san, "state": state(game)}
        return guarded(run)

    @app.post("/api/engine-move")
    def engine_move():
        def run():
            d = game.model_move()
            return {"san": d.san, "search": None if d.search is None else search_json(d.position, d.search),
                    "state": state(game)}
        return guarded(run)

    @app.post("/api/analyse")
    def analyse():
        return guarded(lambda: {"search": search_json(game.board, game.analyse()), "state": state(game)})

    @app.post("/api/undo")
    def undo():
        def run():
            game.undo()
            return state(game)
        return guarded(run)

    @app.post("/api/explain")
    def explain_move(body: ExplainIn):
        def run():
            if body.target == "last":
                d, e = game.explain_last()
                return {"explanation": explanation_json(e, d.san),
                        "search": search_json(d.position, d.search), "state": state(game)}
            search, e = game.explain_position()
            best = e.facts["best_move"]
            return {"explanation": explanation_json(e, best), "search": search_json(game.board, search),
                    "state": state(game)}
        return guarded(run)

    @app.post("/api/settings")
    def settings(body: Settings):
        def run():
            if body.simulations is not None:
                game.set_simulations(max(0, min(body.simulations, 5000)))
            if body.audience is not None:
                if body.audience not in AUDIENCES:
                    raise ValueError(f"audience must be one of {sorted(AUDIENCES)}")
                game.audience = body.audience
            return state(game)
        return guarded(run)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from .player import load_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="chess network checkpoint (best.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--simulations", type=int, default=800)
    ap.add_argument("--llm", choices=["none", "ollama", "openai", "hf"], default="none")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--audience", choices=sorted(AUDIENCES), default="club")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    llm = None if args.llm == "none" else make_backend(args.llm, args.llm_model)
    game = GameSession(load_model(args.model, args.device), args.simulations, llm, args.audience)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, [url]).start()
    print(f"chessml web app on {url}  (Ctrl+C to stop)")
    uvicorn.run(create_app(game), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
