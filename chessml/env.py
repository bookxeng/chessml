from __future__ import annotations

import numpy as np

from .board import Board
from .constants import STARTING_FEN
from .encoding import NUM_ACTIONS, OBS_SHAPE, encode_board, legal_action_mask, move_to_action, action_to_move
from .move import Move
from .rules import Outcome, outcome


class ChessEnv:
    """Gym-style two-player chess environment (no gym dependency).

    Both players act through the same env, alternating turns. Observations and actions
    are always from the perspective of the side to move (see `encoding`).

    step() returns (obs, reward, terminated, truncated, info) where reward is from the
    perspective of the player who just moved: +1 for delivering checkmate, 0 otherwise.
    (The opponent's reward on that final step is the negation.)
    """

    action_size = NUM_ACTIONS
    observation_shape = OBS_SHAPE

    def __init__(self, max_plies: int | None = None):
        """max_plies: truncate the episode after this many half-moves (None = no limit)."""
        self.max_plies = max_plies
        self.board = Board()
        self._plies = 0
        self._refresh()

    def reset(self, fen: str | None = None) -> tuple[np.ndarray, dict]:
        self.board = Board(fen or STARTING_FEN)
        self._plies = 0
        self._refresh()
        return self.observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            raise RuntimeError("game is over; call reset()")
        move = action_to_move(int(action), self.board)
        if move is None or move not in self._moves:
            raise ValueError(f"illegal action {action} in position {self.board.fen()}")
        return self.step_move(move)

    def step_move(self, move: Move) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Same as step() but takes a Move instead of an action index."""
        if self.done:
            raise RuntimeError("game is over; call reset()")
        if move not in self._moves:
            raise ValueError(f"illegal move {move} in position {self.board.fen()}")
        mover = self.board.turn
        self.board.push(move)
        self._plies += 1
        self._refresh()

        terminated = self._outcome is not None
        truncated = not terminated and self.max_plies is not None and self._plies >= self.max_plies
        reward = 0.0
        if terminated and self._outcome.winner is not None:
            reward = 1.0 if self._outcome.winner == mover else -1.0
        self._truncated = truncated
        return self.observation(), reward, terminated, truncated, self._info()

    # ----------------------------------------------------------------- queries

    def observation(self) -> np.ndarray:
        return encode_board(self.board)

    def legal_moves(self) -> list[Move]:
        return list(self._moves)

    def legal_actions(self) -> np.ndarray:
        return np.array([move_to_action(m, self.board) for m in self._moves], dtype=np.int64)

    def legal_action_mask(self) -> np.ndarray:
        return legal_action_mask(self.board, self._moves)

    @property
    def outcome(self) -> Outcome | None:
        return self._outcome

    @property
    def done(self) -> bool:
        return self._outcome is not None or self._truncated

    def render(self) -> str:
        text = str(self.board)
        print(text)
        return text

    # ---------------------------------------------------------------- internal

    def _refresh(self) -> None:
        self._moves = self.board.legal_moves()
        self._outcome = outcome(self.board, self._moves)
        self._truncated = False

    def _info(self) -> dict:
        return {
            "legal_mask": self.legal_action_mask(),
            "outcome": self._outcome,
            "fen": self.board.fen(),
            "turn": self.board.turn,
            "ply": self._plies,
        }
