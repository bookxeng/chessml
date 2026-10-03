"""Play against a trained model in the terminal, with its thinking shown after every move.

    python -m chessml_train.play --model runs/sl-10x128/best.pt              # you play white
    python -m chessml_train.play --model runs/sl-10x128/best.pt --color black
    python -m chessml_train.play --model runs/sl-10x128/best.pt --simulations 800 --device cuda

Each turn shows the board (last move highlighted), the model's Value as a win/draw/loss
bar from white's point of view, and on the model's turns its top Policy candidates.
Commands: hint (model's top moves for you), moves, undo, flip, quit.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from chessml import Board, Move, WHITE, BLACK, move_to_san
from chessml.players import parse_move

from .mcts import SearchResult
from .player import PolicyPlayer, SearchPlayer, load_model

GLYPHS = {1: "♟", 2: "♞", 3: "♝", 4: "♜", 5: "♛", 6: "♚"}
LETTERS = ".PNBRQK"

# 256-colour backgrounds: light square, dark square, and their last-move highlights.
LIGHT, DARK, LIGHT_HI, DARK_HI = 180, 137, 186, 143


def render_board(board: Board, last: Move | None = None, flip: bool = False, color: bool = True) -> str:
    ranks = range(8) if flip else range(7, -1, -1)
    files = range(7, -1, -1) if flip else range(8)
    highlight = {last.from_sq, last.to_sq} if last else set()
    lines = []
    for r in ranks:
        row = f" {r + 1} "
        for f in files:
            sq = r * 8 + f
            piece = board.squares[sq]
            if not color:
                ch = LETTERS[abs(piece)] if piece else ("·" if (r + f) % 2 else " ")
                ch = ch if piece >= 0 else ch.lower()
                row += f"[{ch}]" if sq in highlight else f" {ch} "
                continue
            light = (r + f) % 2 == 1
            bg = (LIGHT_HI if light else DARK_HI) if sq in highlight else (LIGHT if light else DARK)
            fg = "38;5;231" if piece > 0 else "38;5;16"
            ch = GLYPHS[abs(piece)] if piece else " "
            row += f"\x1b[48;5;{bg};{fg}m {ch} \x1b[0m"
        lines.append(row)
    names = "abcdefgh"
    lines.append("    " + "  ".join(names[f] for f in files))
    return "\n".join(lines)


def _bar(fraction: float, width: int) -> str:
    n = round(fraction * width)
    return "█" * n + "░" * (width - n)


def render_value(wdl: np.ndarray, board: Board) -> str:
    """The model's win/draw/loss for the side to move, shown from white's point of view."""
    w, d, l = (wdl if board.turn == WHITE else wdl[::-1])
    width = 30
    white_part = round(w * width)
    draw_part = round(d * width)
    bar = "█" * white_part + "▒" * draw_part + "░" * max(0, width - white_part - draw_part)
    return f" Model's eval  White {w:5.1%}  {bar}  Black {l:5.1%}   (draw {d:.1%})"


def render_candidates(board: Board, moves: list[Move], probs: np.ndarray, top: int = 5,
                      chosen: Move | None = None) -> str:
    order = np.argsort(-probs)[:top]
    lines = []
    for i in order:
        mark = "  ← plays" if moves[i] == chosen else ""
        lines.append(f"   {move_to_san(board, moves[i], moves):<8} {probs[i]:6.1%}  {_bar(float(probs[i]), 20)}{mark}")
    return "\n".join(lines)


def render_search(board: Board, result: SearchResult, top: int = 5, chosen: Move | None = None) -> str:
    """Top moves by visits, with each move's expected score from Search and its Policy prior."""
    total = result.visits.sum()
    lines = ["   move      visits                expected score   policy"]
    for i in np.argsort(-result.visits)[:top]:
        share = result.visits[i] / total
        score = "-" if np.isnan(result.q[i]) else f"{(result.q[i] + 1) / 2:.1%}"
        mark = "  ← plays" if result.moves[i] == chosen else ""
        lines.append(f"   {move_to_san(board, result.moves[i], result.moves):<8} {int(result.visits[i]):>5} "
                     f"{_bar(float(share), 12)}  {score:>14}   {result.priors[i]:6.1%}{mark}")
    return "\n".join(lines)


def _format_moves(sans: list[str]) -> str:
    return " ".join(f"{i // 2 + 1}. {sans[i]}" + (f" {sans[i + 1]}" if i + 1 < len(sans) else "")
                    for i in range(0, len(sans), 2))


def play(model_path: str, human: int = WHITE, device: str = "cpu", color: bool = True,
         simulations: int = 0, input_fn=input, output=print) -> str | None:
    """Run a game; returns the result string, or None if the human quits.

    simulations = 0 plays the raw Policy; otherwise the model thinks with that many Search simulations.
    """
    model = load_model(model_path, device)
    bot = PolicyPlayer(model)
    searcher = SearchPlayer(model, simulations) if simulations else None
    board = Board()
    flip = human == BLACK
    sans: list[str] = []
    output("You are " + ("white" if human == WHITE else "black")
           + ". Enter moves as SAN (Nf3) or UCI (g1f3). Commands: hint, moves, undo, flip, quit.\n")

    while (result := board.outcome()) is None:
        last = board.move_stack[-1] if board.move_stack else None
        moves = board.legal_moves()
        probs, wdl = bot.evaluate(board, moves)
        side = "White" if board.turn == WHITE else "Black"
        output(render_board(board, last, flip, color))
        output(render_value(wdl, board))

        if board.turn != human:
            if searcher:
                search = searcher.analyse(board)
                move = search.best
                output(f"\n {side} (model) to move. Top moves after {search.simulations} Search simulations:")
                output(render_search(board, search, chosen=move))
            else:
                move = moves[int(probs.argmax())]
                output(f"\n {side} (model) to move. Top candidates from its Policy:")
                output(render_candidates(board, moves, probs, chosen=move))
            san = move_to_san(board, move, moves)
            output(f"\n Model plays {san}\n" + "─" * 60)
            sans.append(san)
            board.push(move)
            continue

        output(f"\n {side} (you) to move{' — you are in check!' if board.in_check() else ''}")
        while True:
            text = input_fn(" your move> ").strip()
            cmd = text.lower()
            if cmd in ("quit", "exit", "q"):
                output("Game abandoned.")
                return None
            if cmd == "hint":
                output(" The model would play:\n" + (render_search(board, searcher.analyse(board)) if searcher
                                                     else render_candidates(board, moves, probs)))
                continue
            if cmd == "moves":
                output(" " + " ".join(sorted(move_to_san(board, m, moves) for m in moves)))
                continue
            if cmd == "flip":
                flip = not flip
                output(render_board(board, last, flip, color))
                continue
            if cmd == "undo":
                if len(board.move_stack) < 2:
                    output(" Nothing to undo.")
                    continue
                board.pop(); board.pop()
                sans[-2:] = []
                break
            try:
                move = parse_move(board, text)
            except ValueError as e:
                output(f" {e}. Type 'hint' or 'moves' for help.")
                continue
            sans.append(move_to_san(board, move, moves))
            board.push(move)
            output("─" * 60)
            break

    output(render_board(board, board.move_stack[-1] if board.move_stack else None, flip, color))
    output(f"\n Game over: {result.result} ({result.reason.replace('_', ' ')})")
    output(" " + _format_moves(sans))
    return result.result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint (best.pt / latest.pt)")
    ap.add_argument("--color", choices=["white", "black"], default="white", help="the colour you play")
    ap.add_argument("--simulations", type=int, default=0,
                    help="Search simulations per move (0 = raw policy, 800 = full strength)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-color", action="store_true", help="plain text board (for terminals without ANSI colours)")
    args = ap.parse_args(argv)
    if os.name == "nt":
        os.system("")  # enables ANSI escape codes in the classic Windows console
    sys.stdout.reconfigure(encoding="utf-8")  # chess glyphs, even when output is redirected
    try:
        play(args.model, WHITE if args.color == "white" else BLACK, args.device, not args.no_color,
             args.simulations)
    except (KeyboardInterrupt, EOFError):
        print("\nGame abandoned.")


if __name__ == "__main__":
    main()
