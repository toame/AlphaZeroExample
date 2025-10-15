# network.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GameConfig, NetworkConfig
from game import State

if TYPE_CHECKING:
    from config.models import BasicNetworkConfig, KataGoNetworkConfig


class Conv(nn.Module):
    """3×3/1×1 畳み込みと BatchNorm をまとめたユーティリティ。"""

    def __init__(self, filters0: int, filters1: int, kernel_size: int, bn: bool = False) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class ResidualBlock(nn.Module):
    """従来ネットワーク用の 1 層構成残差ブロック。"""

    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation によるチャネル注意機構。"""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(b, c)
        excite = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        return x * excite.view(b, c, 1, 1)


class KataResidualBlock(nn.Module):
    """KataGo 風の 2 層残差ブロック。"""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.conv1 = Conv(channels, channels, 3, True)
        self.conv2 = Conv(channels, channels, 3, True)
        self.se = SEBlock(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        h = self.se(h)
        return F.relu(x + h)


class _BasicAlphaZeroNet(nn.Module):
    """従来の簡易 AlphaZero ネットワーク。"""

    def __init__(self, input_channels: int, board_len: int, cfg: "BasicNetworkConfig") -> None:
        super().__init__()
        self.board_len = board_len
        self.board_size = board_len * board_len
        self.input_shape = (input_channels, board_len, board_len)

        num_filters = cfg.num_filters
        num_blocks = cfg.num_blocks
        self.pol_ch = cfg.policy_channels
        self.val_ch = cfg.value_channels

        self.layer0 = Conv(input_channels, num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

        self.conv_p1 = Conv(num_filters, self.pol_ch, 1, bn=True)
        self.conv_p2 = Conv(self.pol_ch, 1, 1)

        self.conv_v = Conv(num_filters, self.val_ch, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * self.val_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)

        h_p = F.relu(self.conv_p1(h))
        h_p = self.conv_p2(h_p).view(-1, self.board_size)

        h_v = F.relu(self.conv_v(h))
        h_v = self.fc_v(h_v.view(-1, self.board_size * self.val_ch))

        return F.softmax(h_p, dim=-1), torch.tanh(h_v)


class _KataGoNet(nn.Module):
    """KataGo 風改良ネットワーク。"""

    def __init__(self, input_channels: int, board_len: int, cfg: "KataGoNetworkConfig") -> None:
        super().__init__()
        self.board_len = board_len
        self.board_size = board_len * board_len
        self.input_shape = (input_channels, board_len, board_len)

        num_filters = cfg.num_filters
        num_blocks = cfg.num_blocks

        self.input_conv = Conv(input_channels, num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([KataResidualBlock(num_filters, cfg.se_reduction) for _ in range(num_blocks)])

        # 方策ヘッド: ローカル特徴 + グローバル特徴
        self.policy_conv = Conv(num_filters, cfg.policy_channels, 1, bn=True)
        self.policy_logits = nn.Conv2d(cfg.policy_channels, 1, kernel_size=1, bias=False)
        self.policy_global = nn.Sequential(
            nn.Linear(num_filters, cfg.policy_global_hidden),
            nn.ReLU(),
            nn.Linear(cfg.policy_global_hidden, self.board_size),
        )

        # 価値ヘッド: グローバル平均プーリング + 小型 MLP
        self.value_pre = nn.Linear(num_filters, cfg.value_hidden)
        self.value_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.value_hidden, cfg.value_mid_hidden),
            nn.ReLU(),
            nn.Linear(cfg.value_mid_hidden, 1),
        )

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_conv(x))
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._trunk(x)

        global_feature = F.adaptive_avg_pool2d(h, 1).view(-1, h.size(1))

        # 方策計算
        local_policy = F.relu(self.policy_conv(h))
        local_logits = self.policy_logits(local_policy).view(-1, self.board_size)
        global_logits = self.policy_global(global_feature)
        policy = F.softmax(local_logits + global_logits, dim=-1)

        # 価値計算
        value_hidden = self.value_pre(global_feature)
        value = torch.tanh(self.value_head(value_hidden))

        return policy, value


class Net(nn.Module):
    """アーキテクチャ切り替え対応の推論ネットワーク。"""

    def __init__(self, game_config: GameConfig, net_config: NetworkConfig) -> None:
        super().__init__()
        self._game_config = game_config
        self._net_config = net_config

        board_len = game_config.board_size
        input_channels = 2  # State.feature() のチャンネル数

        if net_config.architecture == "basic":
            self.model = _BasicAlphaZeroNet(input_channels, board_len, net_config.basic)
        elif net_config.architecture == "katago":
            self.model = _KataGoNet(input_channels, board_len, net_config.katago)
        else:
            raise ValueError(f"未知のアーキテクチャ: {net_config.architecture}")

        self.input_shape = self.model.input_shape
        self.board_len = board_len
        self.board_size = self.model.board_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    @torch.no_grad()
    def predict(self, state: State) -> tuple[np.ndarray, float]:
        self.eval()
        x = torch.from_numpy(state.feature()).unsqueeze(0)
        policy, value = self.forward(x)
        return policy.cpu().numpy()[0], float(value.cpu().numpy()[0][0])


if __name__ == "__main__":
    from config.loader import load_default_config
    from training import show_net

    cfg = load_default_config()
    net = Net(cfg.game, cfg.network)
    show_net(net, State(cfg.game))
