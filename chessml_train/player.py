"""Players backed by a trained network (they satisfy chessml.players.Player)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chessml import Board, Move, encode_board, move_to_action

from .mcts import MCTS, SearchResult
from .model import ChessNet, NetConfig


def load_model(path: str | Path, device: str | torch.device = "cpu") -> ChessNet:
    """Load a ChessNet from best.pt or latest.pt, in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ChessNet(NetConfig(**ckpt["net_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


class PolicyPlayer:
    """Plays the legal move with the highest Policy probability (raw policy, no Search)."""

    def __init__(self, model: ChessNet):
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def evaluate(self, board: Board, moves: list[Move]) -> tuple[np.ndarray, np.ndarray]:
        """(probabilities over `moves`, win/draw/loss for the side to move)."""
        obs = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        logits, wdl = self.model(obs)
        actions = torch.tensor([move_to_action(m, board) for m in moves], device=self.device)
        probs = torch.softmax(logits[0, actions].float(), 0)
        return probs.cpu().numpy(), torch.softmax(wdl[0].float(), 0).cpu().numpy()

    def choose(self, board: Board) -> Move:
        moves = board.legal_moves()
        probs, _ = self.evaluate(board, moves)
        return moves[int(probs.argmax())]


class NetEvaluator:
    """Batched network calls for Search: Policy priors over each leaf's legal moves, and a scalar
    Value (win prob - loss prob) for the side to move."""

    def __init__(self, model: ChessNet):
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, actions: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
        x = torch.from_numpy(obs).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits, wdl = self.model(x)
        wdl = torch.softmax(wdl.float(), 1)
        values = (wdl[:, 0] - wdl[:, 2]).cpu().numpy()
        logits = logits.float().cpu().numpy()
        priors = []
        for row, a in zip(logits, actions):
            z = row[a] - row[a].max()
            p = np.exp(z)
            priors.append(p / p.sum())
        return priors, values


class SearchPlayer:
    """Plays the most-visited move after `simulations` simulations of Search."""

    def __init__(self, model: ChessNet, simulations: int = 800, batch_size: int = 32, c_puct: float = 2.0):
        self.simulations = simulations
        self.mcts = MCTS(NetEvaluator(model), c_puct=c_puct, batch_size=batch_size)

    def analyse(self, board: Board) -> SearchResult:
        return self.mcts.search(board, self.simulations)

    def choose(self, board: Board) -> Move:
        return self.analyse(board).best
