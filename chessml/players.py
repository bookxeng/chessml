from __future__ import annotations

import random
from typing import Protocol

from .board import Board
from .constants import QUEEN
from .move import Move
from .san import move_to_san, parse_san


class Player(Protocol):
    def choose(self, board: Board) -> Move | str:
        """Return a legal move, or a command string ("undo", "quit") for interactive players."""
        ...


class RandomPlayer:
    """Plays a uniformly random legal move. Useful as a baseline opponent."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose(self, board: Board) -> Move:
        return self.rng.choice(board.legal_moves())


def parse_move(board: Board, text: str) -> Move:
    """Parse a move typed as UCI (e2e4, e7e8q) or SAN (Nf3, O-O, exd5)."""
    text = text.strip()
    legal = board.legal_moves()
    try:
        move = Move.from_uci(text)
    except ValueError:
        pass
    else:
        # A promotion typed without its piece (e7e8) means queen.
        if move not in legal and not move.promotion:
            queen = Move(move.from_sq, move.to_sq, QUEEN)
            if queen in legal:
                return queen
        if move in legal:
            return move
    return parse_san(board, text)


class HumanPlayer:
    """Reads moves from the terminal."""

    def __init__(self, name: str = "You", input_fn=input, output_fn=print):
        self.name = name
        self.input = input_fn
        self.output = output_fn

    def choose(self, board: Board) -> Move | str:
        while True:
            try:
                text = self.input(f"{self.name}> ").strip()
            except EOFError:
                return "quit"
            if not text:
                continue
            cmd = text.lower()
            if cmd in ("quit", "exit", "q"):
                return "quit"
            if cmd == "undo":
                return "undo"
            if cmd == "help":
                self.output("Enter a move as UCI (e2e4, e7e8q) or SAN (Nf3, O-O, exd5).\n"
                            "Commands: moves, fen, undo, quit")
                continue
            if cmd == "moves":
                legal = board.legal_moves()
                self.output(" ".join(sorted(move_to_san(board, m, legal) for m in legal)))
                continue
            if cmd == "fen":
                self.output(board.fen())
                continue
            try:
                return parse_move(board, text)
            except ValueError as e:
                self.output(f"{e}. Type 'help' for input format.")
