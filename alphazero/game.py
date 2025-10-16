# game.py
from __future__ import annotations
import numpy as np
from typing import List
from config import GameConfig

BLACK, WHITE = 1, -1  # 先手・後手

class State:
    """N×N の○×ゲーム盤面。"""

    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self.size: int = config.board_size
        self.rule: str = config.rule
        self.X: str = config.axes.X
        self.Y: str = config.axes.Y
        self.C = {
            0: config.symbols.empty,
            BLACK: config.symbols.black,
            WHITE: config.symbols.white,
        }

        self.board = np.zeros((self.size, self.size), dtype=np.int8)  # (x, y)
        self.color: int = config.first_player
        self.win_color: int = 0
        self.record: List[int] = []
        self.turn_index: int = 0
        self._connect_length: int = self.size if self.rule == "tic_tac_toe" else 6
        if self.rule == "connect6" and self.size < self._connect_length:
            raise ValueError("connect6 ルールでは盤面サイズを 6 以上にしてください")
        self._stones_remaining: int = self._stones_required_for_turn(self.turn_index)

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
        if self.win_color != 0:
            raise ValueError("終局後に手を進めることはできません。")
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return self

        if isinstance(action, (list, tuple)):
            for a in action:
                self.play(a)
            return self

        x, y = action // self.size, action % self.size
        if self.board[x, y] != 0:
            raise ValueError("既に石が置かれている交点は選択できません。")
        self.board[x, y] = self.color

        if self._check_win(x, y):
            self.win_color = self.color

        self.record.append(action)
        self._stones_remaining -= 1
        if self.win_color != 0:
            self._end_turn()
            return self
        if self._stones_remaining == 0:
            self._end_turn()
        return self

    # --- 終局/合法手/特徴量 ---
    def terminal(self) -> bool:
        return self.win_color != 0 or len(self.record) == self.size * self.size

    def terminal_reward(self) -> int:
        return self.win_color if self.color == BLACK else -self.win_color

    def legal_actions(self) -> List[int]:
        return [a for a in range(self.size * self.size) if self.board[a // self.size, a % self.size] == 0]

    def feature(self) -> np.ndarray:
        """現在手番視点での特徴量（2×N×N、float32）を返す。"""
        return np.stack([
            self.board == self.color,
            self.board == -self.color,
        ]).astype(np.float32)

    def _stones_required_for_turn(self, turn_index: int) -> int:
        """ターンごとの配置数を返す内部ヘルパー。"""
        if self.rule == "connect6":
            return 1 if turn_index == 0 else 2
        return 1

    def _end_turn(self) -> None:
        """ターン終了時の処理。"""
        self.turn_index += 1
        self.color = -self.color
        self._stones_remaining = self._stones_required_for_turn(self.turn_index)

    def _check_win(self, x: int, y: int) -> bool:
        """直前に置いた石から勝利判定を行う。"""
        target = self.color
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dx, dy in directions:
            count = 1
            for sign in (1, -1):
                step = 1
                while True:
                    nx = x + dx * step * sign
                    ny = y + dy * step * sign
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == target:
                        count += 1
                        step += 1
                    else:
                        break
            if count >= self._connect_length:
                return True
        return False

if __name__ == "__main__":
    from config.loader import load_default_config

    config = load_default_config().game
    s = State(config).play("B1")
    print(s)
    print("input feature\n", s.feature())
