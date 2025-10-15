# network.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from game import State
from config.loader import cfg

class Conv(nn.Module):
    def __init__(self, filters0: int, filters1: int, kernel_size: int, bn: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(filters1) if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv(x))

class Net(nn.Module):
    """AlphaZero 風ネット（小型）。方策 p と価値 v を出力。"""
    def __init__(self) -> None:
        super().__init__()
        state = State()
        self.input_shape = state.feature().shape  # (2, N, N)
        self.board_size = self.input_shape[1] * self.input_shape[2]

        num_filters = cfg.network.num_filters
        num_blocks = cfg.network.num_blocks
        self.pol_ch = cfg.network.policy_channels
        self.val_ch = cfg.network.value_channels

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
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

    @torch.no_grad()
    def predict(self, state: State) -> tuple[np.ndarray, float]:
        self.eval()
        x = torch.from_numpy(state.feature()).unsqueeze(0)
        p, v = self.forward(x)
        return p.cpu().numpy()[0], float(v.cpu().numpy()[0][0])

if __name__ == "__main__":
    from training import show_net
    net = Net()
    show_net(net, State())
