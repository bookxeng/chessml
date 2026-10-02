"""AlphaZero-style residual network with a Policy head and a win/draw/loss Value head.

    input (N, 19, 8, 8)
      -> stem: conv3x3 -> BN -> ReLU
      -> `blocks` residual blocks of `filters` channels
      -> policy head: conv3x3 -> BN -> ReLU -> conv3x3 to 73 planes -> (N, 4672) logits
      -> value head:  conv1x1 -> BN -> ReLU -> dense -> ReLU -> dense -> (N, 3) W/D/L logits

The policy head is convolutional: its 73 output planes per square line up with the action
layout `from_square * 73 + plane` (see chessml.encoding), so no dense layer is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from chessml.encoding import NUM_ACTIONS, NUM_PLANES

MOVE_PLANES = 73
WDL = 3  # value classes, in this order: win, draw, loss (for the side to move)


@dataclass(frozen=True)
class NetConfig:
    blocks: int = 10
    filters: int = 128
    value_channels: int = 32
    value_hidden: int = 128


def flatten_policy(planes: torch.Tensor) -> torch.Tensor:
    """(N, 73, 8, 8) move planes -> (N, 4672) logits indexed by action = square * 73 + plane."""
    return planes.permute(0, 2, 3, 1).reshape(-1, NUM_ACTIONS)


def _conv_bn(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm2d(c_out))


class ResidualBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = _conv_bn(filters, filters)
        self.conv2 = _conv_bn(filters, filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class ChessNet(nn.Module):
    def __init__(self, config: NetConfig = NetConfig()):
        super().__init__()
        self.config = config
        f = config.filters
        self.stem = _conv_bn(NUM_PLANES, f)
        self.tower = nn.Sequential(*(ResidualBlock(f) for _ in range(config.blocks)))
        self.policy_conv = _conv_bn(f, f)
        self.policy_out = nn.Conv2d(f, MOVE_PLANES, 3, padding=1)
        self.value_conv = _conv_bn(f, config.value_channels, k=1)
        self.value_fc = nn.Linear(config.value_channels * 64, config.value_hidden)
        self.value_out = nn.Linear(config.value_hidden, WDL)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy logits (N, 4672), W/D/L logits (N, 3))."""
        x = self.tower(F.relu(self.stem(x)))
        policy = flatten_policy(self.policy_out(F.relu(self.policy_conv(x))))
        v = F.relu(self.value_conv(x)).flatten(1)
        value = self.value_out(F.relu(self.value_fc(v)))
        return policy, value


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
