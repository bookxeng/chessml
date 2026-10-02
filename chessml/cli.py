from __future__ import annotations

import argparse

from .board import Board
from .constants import WHITE, STARTING_FEN
from .players import HumanPlayer, RandomPlayer
from .san import move_to_san


def _make_player(kind: str, name: str, seed: int | None):
    if kind == "human":
        return HumanPlayer(name)
    if kind == "random":
        return RandomPlayer(seed)
    raise ValueError(f"unknown player type: {kind}")


def _format_moves(sans: list[str], white_first: bool) -> str:
    parts, n = [], 1
    i = 0
    if not white_first and sans:
        parts.append(f"{n}... {sans[0]}")
        i, n = 1, 2
    while i < len(sans):
        parts.append(f"{n}. " + " ".join(sans[i:i + 2]))
        i += 2
        n += 1
    return " ".join(parts)


def play(white: str = "human", black: str = "random", fen: str = STARTING_FEN,
         seed: int | None = None) -> str | None:
    """Run a game in the terminal. Returns the result string, or None if quit."""
    board = Board(fen)
    white_first = board.turn == WHITE
    players = {WHITE: _make_player(white, "White", seed),
               -WHITE: _make_player(black, "Black", None if seed is None else seed + 1)}
    sans: list[str] = []
    print("Enter moves as UCI (e2e4) or SAN (Nf3). Type 'help' for commands.\n")

    while (result := board.outcome()) is None:
        side = "White" if board.turn == WHITE else "Black"
        print(board)
        print(f"{side} to move{' (check)' if board.in_check() else ''}\n")

        player = players[board.turn]
        choice = player.choose(board)
        if choice == "quit":
            print("Game abandoned.")
            return None
        if choice == "undo":
            # Undo back to this human's previous turn, skipping over a bot's reply.
            opponent_is_bot = not isinstance(players[-board.turn], HumanPlayer)
            count = 2 if opponent_is_bot else 1
            if len(board.move_stack) < count:
                print("Nothing to undo.\n")
                continue
            for _ in range(count):
                board.pop()
                sans.pop()
            continue

        san = move_to_san(board, choice)
        sans.append(san)
        board.push(choice)
        if not isinstance(player, HumanPlayer):
            print(f"{side} plays {san}\n")

    print(board)
    print(f"\nGame over: {result.result} ({result.reason.replace('_', ' ')})")
    print(_format_moves(sans, white_first))
    return result.result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m chessml", description="Play chess in the terminal.")
    parser.add_argument("--white", choices=("human", "random"), default="human")
    parser.add_argument("--black", choices=("human", "random"), default="random")
    parser.add_argument("--fen", default=STARTING_FEN, help="starting position")
    parser.add_argument("--seed", type=int, default=None, help="seed for random players")
    args = parser.parse_args(argv)
    try:
        play(args.white, args.black, args.fen, args.seed)
    except KeyboardInterrupt:
        print("\nGame abandoned.")
