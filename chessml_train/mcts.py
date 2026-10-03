"""Search: Monte Carlo tree search guided by a network's Policy and Value (AlphaZero-style PUCT).

Each simulation selects a path by max Q + U, expands the leaf with one network call
(Policy priors + Value), and backs the value up the path, flipping sign every ply.
Leaves are collected in batches so the network evaluates many positions per call; a
virtual loss on paths already in the batch pushes later selections onto other branches.

Values are scalars in [-1, 1] from the point of view of the side to move at that node
(win prob - loss prob). Edge statistics N, W, Q at a node are from the point of view of
the side to move at that node, i.e. the player choosing among its moves.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from chessml import Board, Move, encode_board, move_to_action
from chessml.rules import outcome

# (observations (B, 19, 8, 8), legal action indices per leaf) -> (priors per leaf, values (B,))
Evaluator = Callable[[np.ndarray, list[np.ndarray]], tuple[list[np.ndarray], np.ndarray]]

VIRTUAL_LOSS = 1.0


def _actions(moves: list[Move], board: Board) -> np.ndarray:
    return np.array([move_to_action(m, board) for m in moves], dtype=np.int64)


class Node:
    __slots__ = ("moves", "priors", "N", "W", "children", "value", "terminal")

    def __init__(self):
        self.moves: list[Move] | None = None  # None until expanded
        self.priors = self.N = self.W = None
        self.children: list[Node | None] = []
        self.value = 0.0                      # network (or exact) value for the side to move
        self.terminal: float | None = None    # exact value if the game is over here

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    def expand(self, moves: list[Move], priors: np.ndarray, value: float) -> None:
        self.moves = moves
        self.priors = priors.astype(np.float64)
        self.N = np.zeros(len(moves))
        self.W = np.zeros(len(moves))
        self.children = [None] * len(moves)
        self.value = value

    def select(self, c_puct: float, fpu_reduction: float) -> int:
        total = self.N.sum()
        # Unvisited moves are assumed slightly worse than this node's own value ("first play urgency").
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), self.value - fpu_reduction)
        u = c_puct * self.priors * math.sqrt(total + 1) / (1 + self.N)
        return int(np.argmax(q + u))


@dataclass
class SearchResult:
    moves: list[Move]
    visits: np.ndarray        # N per root move
    q: np.ndarray             # mean value per root move (side to move's view); nan if unvisited
    priors: np.ndarray        # Policy priors per root move
    value: float              # network value of the root position
    simulations: int

    @property
    def best(self) -> Move:
        return self.moves[int(self.visits.argmax())]


class MCTS:
    def __init__(self, evaluator: Evaluator, c_puct: float = 2.0, fpu_reduction: float = 0.25,
                 batch_size: int = 16):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.batch_size = batch_size

    def search(self, board: Board, simulations: int) -> SearchResult:
        """Run `simulations` simulations from `board` (which is left unchanged)."""
        board = board.copy()
        root = Node()
        moves = board.legal_moves()
        if self._terminal_value(board, moves, root) is not None:
            raise ValueError("cannot search a finished game")
        self._evaluate([(root, moves, _actions(moves, board), encode_board(board))])

        done = 0
        while done < simulations:
            pending, seen = [], set()
            while len(pending) < min(self.batch_size, simulations - done):
                path, leaf, leaf_data = self._select_leaf(root, board)
                if leaf.terminal is not None:
                    self._backup(path, leaf.terminal)
                    done += 1
                    continue
                if id(leaf) in seen:  # two paths reached the same unexpanded leaf: evaluate what we have
                    self._undo_virtual(path)
                    break
                seen.add(id(leaf))
                pending.append((path, (leaf, *leaf_data)))
            if pending:
                values = self._evaluate([leaf for _, leaf in pending])
                for (path, _), v in zip(pending, values):
                    self._backup(path, v)
                done += len(pending)

        q = np.where(root.N > 0, root.W / np.maximum(root.N, 1), np.nan)
        return SearchResult(root.moves, root.N.copy(), q, root.priors, root.value, done)

    # ---------------------------------------------------------------- internals

    def _select_leaf(self, root: Node, board: Board):
        """Walk down by PUCT applying virtual loss.

        Returns (path, leaf, (moves, actions, obs)); the leaf data is computed while the board
        is at the leaf, since actions and observations depend on the side to move there.
        """
        path: list[tuple[Node, int]] = []
        node, pushed = root, 0
        while node.expanded:
            i = node.select(self.c_puct, self.fpu_reduction)
            node.N[i] += 1
            node.W[i] -= VIRTUAL_LOSS
            path.append((node, i))
            board.push(node.moves[i])
            pushed += 1
            if node.children[i] is None:
                node.children[i] = Node()
            node = node.children[i]
            if node.terminal is not None:
                break
        data = None
        if node.terminal is None:
            moves = board.legal_moves()
            if self._terminal_value(board, moves, node) is None:
                data = (moves, _actions(moves, board), encode_board(board))
        for _ in range(pushed):
            board.pop()
        return path, node, data

    @staticmethod
    def _terminal_value(board: Board, moves: list[Move], node: Node) -> float | None:
        result = outcome(board, moves)
        if result is not None:
            node.terminal = 0.0 if result.winner is None else -1.0  # side to move is mated, or a draw
        return node.terminal

    def _evaluate(self, leaves: list[tuple[Node, list[Move], np.ndarray, np.ndarray]]) -> list[float]:
        obs = np.stack([o for *_, o in leaves])
        priors, values = self.evaluator(obs, [a for _, _, a, _ in leaves])
        for (node, moves, _, _), p, v in zip(leaves, priors, values):
            node.expand(moves, p, float(v))
        return [float(v) for v in values]

    @staticmethod
    def _backup(path: list[tuple[Node, int]], leaf_value: float) -> None:
        # N was already counted during selection; replace the virtual loss with the real value.
        # The leaf value is for the side to move at the leaf; the last edge belongs to the opponent.
        v = -leaf_value
        for node, i in reversed(path):
            node.W[i] += v + VIRTUAL_LOSS
            v = -v

    @staticmethod
    def _undo_virtual(path: list[tuple[Node, int]]) -> None:
        for node, i in path:
            node.N[i] -= 1
            node.W[i] += VIRTUAL_LOSS
