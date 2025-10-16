"""自己対戦時の温度調整ロジックをまとめたモジュール。"""

from __future__ import annotations

import math
import random

from config import TemperatureScheduleConfig


class TemperatureController:
    """自己対戦における手番ごとの温度を算出する管理クラス。"""

    def __init__(self, config: TemperatureScheduleConfig) -> None:
        # 設定を保持し、ゲームごとの状態を初期化する
        self._config = config
        self._move_index = 0
        self._last_temperature = config.initial
        self._noise_bias = 0.0
        self.reset()

    def reset(self) -> None:
        """新しい対局に備えて内部状態を初期化する。"""

        self._move_index = 0
        self._last_temperature = self._config.initial
        if self._config.noise_scale > 0.0:
            # ゲームごとに探索の揺らぎを微調整し、同じ局面の繰り返しを避ける
            self._noise_bias = random.gauss(0.0, self._config.noise_scale)
        else:
            self._noise_bias = 0.0

    @property
    def last_temperature(self) -> float:
        """直近に使用した温度を返す。"""

        return self._last_temperature

    def step(self) -> float:
        """現在の手番に適用する温度を算出して返す。"""

        temperature = self.temperature_for_move(self._move_index)
        self._last_temperature = temperature
        self._move_index += 1
        return temperature

    def temperature_for_move(self, move_index: int) -> float:
        """指定した手数における温度を計算する。"""

        base = self._config.initial
        decayed = self._apply_decay(base, move_index)
        endgame_adjusted = self._apply_endgame_transition(decayed, move_index)
        noisy = endgame_adjusted + self._noise_bias
        lower_bound = self._config.min_value
        upper_bound = self._config.initial
        if self._config.endgame_move is not None:
            upper_bound = max(upper_bound, self._config.endgame_temperature)
        clamped = min(upper_bound, max(lower_bound, noisy))
        return clamped

    def _apply_decay(self, temperature: float, move_index: int) -> float:
        """ウォームアップ終了後に指数的な減衰を適用する。"""

        if move_index < self._config.warmup_moves:
            return temperature
        decay_steps = max(0, move_index - self._config.warmup_moves + 1)
        decayed = temperature * (self._config.decay_rate ** decay_steps)
        return max(decayed, self._config.min_value)

    def _apply_endgame_transition(self, temperature: float, move_index: int) -> float:
        """終盤に向けて滑らかに目標温度へ遷移させる。"""

        if self._config.endgame_move is None:
            return temperature
        offset = move_index - self._config.endgame_move
        if offset < 0:
            return temperature
        slope = max(self._config.endgame_slope, 1e-3)
        weight = 1.0 / (1.0 + math.exp(-offset / slope))
        blended = (1.0 - weight) * temperature + weight * self._config.endgame_temperature
        return blended


__all__ = ["TemperatureController"]
