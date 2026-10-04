"""Play matches between players on the Opponent ladder, alternating colors.

    python -m chessml_train.arena --model runs/sl-10x128/best.pt --opponent random
    python -m chessml_train.arena --model runs/sl-10x128/best.pt --simulations 800 --opponent greedy
    python -m chessml_train.arena --model runs/sl-10x128/best.pt --simulations 800 \\
        --opponent stockfish --elo 1600 --games 100
    python -m chessml_train.arena --model runs/sl-10x128/best.pt --simulations 200 --opponent raw

Games come in pairs from the same opening, played once with each color. The opening is
`--opening-plies` random legal moves, which gives deterministic players varied games.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

from chessml import Board, Move, WHITE
from chessml.players import GreedyPlayer, Player, RandomPlayer


@dataclass
class MatchResult:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    reasons: dict = field(default_factory=dict)

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Points per game for the first player: win 1, draw 0.5."""
        return (self.wins + 0.5 * self.draws) / max(1, self.games)

    def __str__(self) -> str:
        return f"+{self.wins} ={self.draws} -{self.losses}  (score {self.score:.1%} over {self.games} games)"


def random_opening(plies: int, rng: random.Random) -> list[Move]:
    board = Board()
    moves = []
    for _ in range(plies):
        legal = board.legal_moves()
        if not legal or board.outcome():
            break
        move = rng.choice(legal)
        board.push(move)
        moves.append(move)
    return moves


def play_game(white: Player, black: Player, max_plies: int = 400,
              opening: list[Move] = ()) -> tuple[int, str]:
    """(result from white's view: +1/0/-1, reason). Reaching max_plies counts as a draw."""
    board = Board()
    for player in (white, black):
        if hasattr(player, "new_game"):
            player.new_game()
    for move in opening:
        board.push(move)
    while len(board.move_stack) < max_plies:
        outcome = board.outcome()
        if outcome is not None:
            return (0 if outcome.winner is None else (1 if outcome.winner == WHITE else -1)), outcome.reason
        player = white if board.turn == WHITE else black
        board.push(player.choose(board))
    outcome = board.outcome()
    if outcome is None:
        return 0, "ply_cap"
    return (0 if outcome.winner is None else (1 if outcome.winner == WHITE else -1)), outcome.reason


def match(player: Player, opponent: Player, games: int, max_plies: int = 400,
          opening_plies: int = 0, seed: int = 0, log=None) -> MatchResult:
    result = MatchResult()
    rng = random.Random(seed)
    opening: list[Move] = []
    for g in range(games):
        if g % 2 == 0:
            opening = random_opening(opening_plies, rng)
            score, reason = play_game(player, opponent, max_plies, opening)
        else:
            score, reason = play_game(opponent, player, max_plies, opening)
            score = -score
        if score > 0:
            result.wins += 1
        elif score < 0:
            result.losses += 1
        else:
            result.draws += 1
        result.reasons[reason] = result.reasons.get(reason, 0) + 1
        if log:
            log(f"game {g + 1}/{games}: {result}")
    return result


def main(argv: list[str] | None = None) -> None:
    from .player import PolicyPlayer, SearchPlayer, load_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint (best.pt / latest.pt)")
    ap.add_argument("--simulations", type=int, default=0, help="Search simulations per move (0 = raw policy)")
    ap.add_argument("--batch-size", type=int, default=32, help="leaves per network call during Search")
    ap.add_argument("--opponent", default="random", choices=["random", "greedy", "stockfish", "raw"],
                    help="raw = the same model without Search")
    ap.add_argument("--elo", type=int, default=None, help="Stockfish UCI_Elo (>= 1320)")
    ap.add_argument("--movetime", type=int, default=100, help="Stockfish milliseconds per move")
    ap.add_argument("--stockfish", default=None, help="path to the Stockfish binary")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    model = load_model(args.model, args.device)
    player = (SearchPlayer(model, args.simulations, args.batch_size) if args.simulations
              else PolicyPlayer(model))
    name = f"search {args.simulations}" if args.simulations else "raw policy"

    if args.opponent == "stockfish":
        from .stockfish import StockfishPlayer, find_stockfish
        path = find_stockfish(args.stockfish)
        if path is None:
            raise SystemExit("Stockfish not found: pass --stockfish PATH or set STOCKFISH")
        opponent = StockfishPlayer(path, args.elo, args.movetime)
        opp_name = f"stockfish elo {args.elo}" if args.elo else "stockfish (full strength)"
    elif args.opponent == "greedy":
        opponent, opp_name = GreedyPlayer(args.seed), "greedy"
    elif args.opponent == "raw":
        opponent, opp_name = PolicyPlayer(model), "raw policy"
    else:
        opponent, opp_name = RandomPlayer(args.seed), "random"

    result = match(player, opponent, args.games, args.max_plies, args.opening_plies, args.seed,
                   log=lambda s: print(s, end="\r", flush=True))
    print(f"\n{name} vs {opp_name}: {result}")
    print("endings: " + ", ".join(f"{k} {v}" for k, v in sorted(result.reasons.items(), key=lambda kv: -kv[1])))
    if hasattr(opponent, "close"):
        opponent.close()


if __name__ == "__main__":
    main()
