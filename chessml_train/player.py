"""Players backed by a trained network (they satisfy chessml.players.Player)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chessml import Board, Move, encode_board, move_to_action

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
