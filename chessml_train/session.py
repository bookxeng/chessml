"""A game against (or an analysis with) the model, independent of any user interface.

Both the terminal game (`play`) and the web app drive a GameSession: it owns the board,
the move list, the model's last decision, and the Search/Policy/LLM calls.

Modes:
    play      the human plays one colour and the model the other (`human` is WHITE or BLACK)
    analysis  no opponent: either side's moves are entered freely (`human` is None)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chessml import Board, Move, STARTING_FEN, WHITE, move_to_san
from chessml.players import parse_move

from .explain import Explanation, LLMBackend, explain
from .mcts import SearchResult
from .model import ChessNet
from .player import PolicyPlayer, SearchPlayer


@dataclass
class Decision:
    """A move the model chose: the position before it, the search behind it (None for raw
    policy), and the move in SAN."""
    position: Board
    search: SearchResult | None
    move: Move
    san: str


class GameSession:
    def __init__(self, model: ChessNet, simulations: int = 800, llm: LLMBackend | None = None,
                 audience: str = "club"):
        self.model = model
        self.policy = PolicyPlayer(model)
        self.searcher = None
        self.set_simulations(simulations)
        self.llm = llm
        self.audience = audience
        self.new_game()

    def set_simulations(self, simulations: int) -> None:
        """Search budget per model move; 0 plays the raw Policy."""
        self.searcher = SearchPlayer(self.model, simulations) if simulations > 0 else None

    @property
    def simulations(self) -> int:
        return self.searcher.simulations if self.searcher else 0

    # ---------------------------------------------------------------- game state

    def new_game(self, human: int | None = WHITE, fen: str | None = None) -> None:
        """Start a game: human = WHITE / BLACK plays the model; None is an analysis board."""
        self.start_fen = fen or STARTING_FEN
        self.board = Board(self.start_fen)
        self.human = human
        self.sans: list[str] = []
        self.last_decision: Decision | None = None

    @property
    def mode(self) -> str:
        return "analysis" if self.human is None else "play"

    @property
    def outcome(self):
        return self.board.outcome()

    @property
    def model_to_move(self) -> bool:
        return self.human is not None and self.board.turn != self.human and self.outcome is None

    @property
    def can_search(self) -> bool:
        return self.searcher is not None

    # ---------------------------------------------------------------- moves

    def play(self, move: Move | str) -> str:
        """Make a human move (a Move, or text in UCI or SAN). Returns its SAN; raises ValueError."""
        if self.outcome is not None:
            raise ValueError("the game is over")
        if self.model_to_move:
            raise ValueError("it is the model's turn")
        if isinstance(move, str):
            move = parse_move(self.board, move)
        legal = self.board.legal_moves()
        if move not in legal:
            raise ValueError(f"illegal move: {move.uci()}")
        return self._push(move, legal)

    def model_move(self) -> Decision:
        """Let the model choose and play a move (with Search if enabled, else raw Policy)."""
        if self.outcome is not None:
            raise ValueError("the game is over")
        position = self.board.copy()
        legal = self.board.legal_moves()
        if self.searcher:
            search = self.searcher.analyse(self.board)
            move = search.best
        else:
            probs, _ = self.policy.evaluate(self.board, legal)
            search, move = None, legal[int(probs.argmax())]
        san = self._push(move, legal)
        self.last_decision = Decision(position, search, move, san)
        return self.last_decision

    def undo(self) -> int:
        """Take back moves: in play mode back to the human's turn, in analysis one move. Returns count."""
        undone = 0
        if self.board.move_stack:
            self._pop()
            undone = 1
            if self.human is not None and self.board.turn != self.human and self.board.move_stack:
                self._pop()
                undone = 2
        self.last_decision = None
        return undone

    def _push(self, move: Move, legal: list[Move]) -> str:
        san = move_to_san(self.board, move, legal)
        self.board.push(move)
        self.sans.append(san)
        return san

    def _pop(self) -> None:
        self.board.pop()
        self.sans.pop()

    # ---------------------------------------------------------------- engine and LLM

    def evaluate(self) -> tuple[list[Move], np.ndarray, np.ndarray]:
        """Raw network output for the current position: (legal moves, Policy probs, W/D/L)."""
        moves = self.board.legal_moves()
        probs, wdl = self.policy.evaluate(self.board, moves)
        return moves, probs, wdl

    def analyse(self) -> SearchResult:
        """Search the current position without moving."""
        if not self.searcher:
            raise ValueError("Search is off (simulations = 0)")
        if self.outcome is not None:
            raise ValueError("the game is over")
        return self.searcher.analyse(self.board)

    def explain_last(self) -> tuple[Decision, Explanation]:
        """Explain the model's last move."""
        if self.llm is None:
            raise ValueError("no LLM configured")
        d = self.last_decision
        if d is None or d.search is None:
            raise ValueError("nothing to explain: the model has not moved with Search yet")
        return d, explain(d.position, d.search, self.llm, self.audience)

    def explain_position(self) -> tuple[SearchResult, Explanation]:
        """Search the current position and explain the engine's choice (analysis board)."""
        if self.llm is None:
            raise ValueError("no LLM configured")
        search = self.analyse()
        return search, explain(self.board, search, self.llm, self.audience)
