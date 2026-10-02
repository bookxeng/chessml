"""Play matches between players, alternating colors.

    python -m chessml_train.arena --model runs/sl-10x128/best.pt --opponent random --games 100
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from chessml import Board, WHITE
from chessml.players import Player, RandomPlayer


@dataclass
class MatchResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Points per game for the first player: win 1, draw 0.5."""
        return (self.wins + 0.5 * self.draws) / max(1, self.games)

    def __str__(self) -> str:
        return f"+{self.wins} ={self.draws} -{self.losses}  (score {self.score:.1%} over {self.games} games)"


def play_game(white: Player, black: Player, max_plies: int = 400) -> int:
    """Result from white's point of view: +1, 0 or -1. Reaching max_plies counts as a draw."""
    board = Board()
    for _ in range(max_plies):
        outcome = board.outcome()
        if outcome is not None:
            return 0 if outcome.winner is None else (1 if outcome.winner == WHITE else -1)
        player = white if board.turn == WHITE else black
        board.push(player.choose(board))
    outcome = board.outcome()
    return 0 if outcome is None or outcome.winner is None else (1 if outcome.winner == WHITE else -1)


def match(player: Player, opponent: Player, games: int, max_plies: int = 400, log=None) -> MatchResult:
    result = MatchResult()
    for g in range(games):
        if g % 2 == 0:
            score = play_game(player, opponent, max_plies)
        else:
            score = -play_game(opponent, player, max_plies)
        if score > 0:
            result.wins += 1
        elif score < 0:
            result.losses += 1
        else:
            result.draws += 1
        if log:
            log(f"game {g + 1}/{games}: {result}")
    return result


def main(argv: list[str] | None = None) -> None:
    from .player import PolicyPlayer, load_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint (best.pt / latest.pt)")
    ap.add_argument("--opponent", default="random", choices=["random"])
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    player = PolicyPlayer(load_model(args.model, args.device))
    opponent = RandomPlayer(args.seed)
    result = match(player, opponent, args.games, args.max_plies,
                   log=lambda s: print(s, end="\r", flush=True))
    print(f"\n{args.model} vs {args.opponent}: {result}")


if __name__ == "__main__":
    main()
