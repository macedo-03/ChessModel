"""Policy/value network: ResNet trunk + conv policy head + value head.

Board input per ADR-0001, policy output per ADR-0003, value head per ADR-0004.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from chessmodel.data.encode import NUM_INPUT_PLANES, NUM_POLICY_PLANES

DEFAULT_TRUNK_CHANNELS = 128
DEFAULT_NUM_RESIDUAL_BLOCKS = 8
DEFAULT_VALUE_CHANNELS = 8
DEFAULT_VALUE_HIDDEN = 256


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection, per the standard ResNet pattern."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


def flatten_policy_logits(logits: Tensor) -> Tensor:
    """(N, 73, 8, 8) channels-first conv output -> (N, 4672), ADR-0003 index order.

    ADR-0003 fixes the flat index as (row*8 + col)*73 + plane. PyTorch's conv
    output is channels-first -- flattening it directly would give
    plane*64 + row*8 + col instead, a different (and wrong) ordering. This
    permute is what makes the two agree; see ADR-0004's Consequences for why
    this specific bug would train "successfully" while being silently wrong.
    """
    n = logits.shape[0]
    return logits.permute(0, 2, 3, 1).reshape(n, -1)


class ChessNet(nn.Module):
    def __init__(
        self,
        trunk_channels: int = DEFAULT_TRUNK_CHANNELS,
        num_residual_blocks: int = DEFAULT_NUM_RESIDUAL_BLOCKS,
        value_channels: int = DEFAULT_VALUE_CHANNELS,
        value_hidden: int = DEFAULT_VALUE_HIDDEN,
    ) -> None:
        super().__init__()

        self.stem_conv = nn.Conv2d(
            NUM_INPUT_PLANES, trunk_channels, kernel_size=3, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(trunk_channels)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(trunk_channels) for _ in range(num_residual_blocks)]
        )

        self.policy_conv1 = nn.Conv2d(
            trunk_channels, trunk_channels, kernel_size=3, padding=1, bias=False
        )
        self.policy_bn1 = nn.BatchNorm2d(trunk_channels)
        self.policy_conv2 = nn.Conv2d(trunk_channels, NUM_POLICY_PLANES, kernel_size=1)

        self.value_conv = nn.Conv2d(trunk_channels, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * 8 * 8, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: (N, 19, 8, 8) -> (policy_logits: (N, 4672), value: (N,))"""
        trunk = torch.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.residual_blocks:
            trunk = block(trunk)

        policy = torch.relu(self.policy_bn1(self.policy_conv1(trunk)))
        policy = self.policy_conv2(policy)  # (N, 73, 8, 8)
        policy_logits = flatten_policy_logits(policy)  # (N, 4672)

        value = torch.relu(self.value_bn(self.value_conv(trunk)))
        value = value.flatten(start_dim=1)  # (N, value_channels * 64)
        value = torch.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # (N, 1)

        return policy_logits, value.squeeze(-1)  # value: (N,)
