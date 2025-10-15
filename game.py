# game.py
from __future__ import annotations
import numpy as np
from typing import List
from config.loader import cfg

BLACK, WHITE = 1, -1  # 先手・後手

class State:
    """N×N の○×ゲーム盤面。"""
    def __init__(self) -> None:
        g = cfg.game
        self.size: int = g.board_size
        self.X: str = g.axes.X
        self.Y: str = g.axes.Y
        self.C = {0: g.symbols.empty, BLACK: g.symbols.black, WHITE: g.symbols.white}

        self.board = np.zeros((self.size, self.size), dtype=np.int8)  # (x, y)
        self.color: int = g.first_player
        self.win_color: int = 0
        self.record: List[int] = []

    # --- 文字列表現との相互変換 ---
    def action2str(self, a: int) -> str:
        return self.X[a // self.size] + self.Y[a % self.size]

    def str2action(self, s: str) -> int:
        return self.X.find(s[0]) * self.size + self.Y.find(s[1])

    def record_string(self) -> str:
        return " ".join([self.action2str(a) for a in self.record])

    def __str__(self) -> str:
        s = "   " + " ".join(self.Y) + "\n"
        for i in range(self.size):
            row = [self.C[int(self.board[i, j])] for j in range(self.size)]
            s += self.X[i] + " " + " ".join(row) + "\n"
        s += "record = " + self.record_string()
        return s

    # --- 盤面更新 ---
    def play(self, action: int | str) -> "State":
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return self

        x, y = action // self.size, action % self.size
        self.board[x, y] = self.color

        n = self.size * self.color
        if (
            self.board[x, :].sum() == n
            or self.board[:, y].sum() == n
            or (x == y and np.diag(self.board, k=0).sum() == n)
            or (x == self.size - 1 - y and np.diag(self.board[::-1, :], k=0).sum() == n)
        ):
            self.win_color = self.color

        self.color = -self.color
        self.record.append(action)
        return self

    # --- 終局/合法手/特徴量 ---
    def terminal(self) -> bool:
        return self.win_color != 0 or len(self.record) == self.size * self.size

    def terminal_reward(self) -> int:
        return self.win_color if self.color == BLACK else -self.win_color

    def legal_actions(self) -> List[int]:
        return [a for a in range(self.size * self.size) if self.board[a // self.size, a % self.size] == 0]

    def feature(self) -> np.ndarray:
        return np.stack([self.board == self.color, self.board == -self.color]).astype(np.float32)

if __name__ == "__main__":
    s = State().play("B1")
    print(s)
    print("input feature\n", s.feature())
