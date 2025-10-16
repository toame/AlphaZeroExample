# config/models.py
from __future__ import annotations

import string
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

class GameAxes(BaseModel):
    X: str = "ABC"
    Y: str = "123"

    @classmethod
    def auto(cls, board_size: int) -> "GameAxes":
        """盤面サイズから推奨座標軸を自動生成する。"""

        digits = "123456789"
        letters_upper = string.ascii_uppercase
        letters_lower = string.ascii_lowercase

        if board_size > len(letters_upper):
            raise ValueError(
                "board_size が 26 を超える場合は axes.X を手動で設定してください"
            )
        x_axis = letters_upper[:board_size]

        if board_size <= len(digits):
            y_axis = digits[:board_size]
        else:
            if board_size > len(letters_lower):
                raise ValueError(
                    "board_size が 26 を超える場合は axes.Y を手動で設定してください"
                )
            y_axis = letters_lower[:board_size]

        return cls(X=x_axis, Y=y_axis)

class GameSymbols(BaseModel):
    empty: str = "_"
    black: str = "O"
    white: str = "X"

class GameConfig(BaseModel):
    board_size: int | None = None
    rule: Literal["tic_tac_toe", "connect6"] = "tic_tac_toe"
    axes: GameAxes | None = None
    symbols: GameSymbols = Field(default_factory=GameSymbols)
    first_player: Literal[-1, 1] = 1

    @field_validator("board_size")
    @classmethod
    def _board_size_rule_check(cls, v: int, info):
        """ルールごとの最小盤面サイズを検証する。"""
        if v is None:
            return v
        rule = info.data.get("rule", "tic_tac_toe")
        if rule == "connect6" and v < 6:
            raise ValueError("connect6 ルールでは board_size を 6 以上に設定してください")
        return v

    @model_validator(mode="after")
    def _apply_rule_defaults(self) -> "GameConfig":
        """ルールに応じた推奨設定を適用し、座標軸を自動補完する。"""

        default_board_size = 3 if self.rule == "tic_tac_toe" else 19
        board_size = self.board_size or default_board_size

        if self.rule == "connect6" and board_size < 6:
            raise ValueError("connect6 ルールでは board_size を 6 以上に設定してください")

        axes = self.axes or GameAxes.auto(board_size)
        if len(axes.X) != board_size or len(axes.Y) != board_size:
            raise ValueError(f"axes.X/Y の長さは board_size={board_size} と一致させてください")

        self.board_size = board_size
        self.axes = axes
        return self

class BasicNetworkConfig(BaseModel):
    """従来の AlphaZero 風ネットワーク設定。"""

    num_filters: int = 16
    num_blocks: int = 6
    policy_channels: int = 4
    value_channels: int = 4


class KataGoNetworkConfig(BaseModel):
    """KataGo 風の拡張ネットワーク設定。"""

    num_filters: int = 128
    num_blocks: int = 10
    policy_channels: int = 32
    policy_global_hidden: int = 256
    value_hidden: int = 256
    value_mid_hidden: int = 64
    se_reduction: int = 4


class NetworkConfig(BaseModel):
    architecture: Literal["basic", "katago"] = "basic"
    basic: BasicNetworkConfig = Field(default_factory=BasicNetworkConfig)
    katago: KataGoNetworkConfig = Field(default_factory=KataGoNetworkConfig)

class TemperatureScheduleConfig(BaseModel):
    """自己対戦時の温度調整に関する設定。"""

    initial: float = 0.7
    warmup_moves: int = 2
    decay_rate: float = 0.85
    min_value: float = 0.05
    endgame_move: int | None = 16
    endgame_temperature: float = 0.02
    endgame_slope: float = 3.0
    noise_scale: float = 0.01

    @field_validator("warmup_moves")
    @classmethod
    def _warmup_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup_moves は 0 以上に設定してください")
        return v

    @field_validator("decay_rate")
    @classmethod
    def _decay_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("decay_rate は正の値に設定してください")
        return v

    @field_validator("min_value", "endgame_temperature", "noise_scale")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("min_value/endgame_temperature/noise_scale は 0 以上に設定してください")
        return v

    @field_validator("endgame_slope")
    @classmethod
    def _slope_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("endgame_slope は正の値に設定してください")
        return v


class MCTSConfig(BaseModel):
    dirichlet_alpha: float = 0.15
    dirichlet_weight: float = 0.25
    puct_c: float = 2.0
    temperature: TemperatureScheduleConfig = Field(default_factory=TemperatureScheduleConfig)
    num_simulations_demo: int = 1000
    num_simulations_demo_mid: int = 3000
    num_simulations_train: int = 50

class LRSchedulerConfig(BaseModel):
    type: Literal["none", "cosine", "onecycle"] = "cosine"
    cosine_min_lr: float = 1e-5
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4


class ReplayBufferConfig(BaseModel):
    capacity: int = 500
    warmup_size: int = 128
    sample_size: int = 256
    recent_ratio: float = 0.2

    @field_validator("capacity", "warmup_size", "sample_size")
    @classmethod
    def _positive(cls, v: int) -> int:
        """容量やサンプル数が正の値か検証する。"""

        if v <= 0:
            raise ValueError("replay buffer の容量やサンプル数は 1 以上に設定してください")
        return v

    @field_validator("recent_ratio")
    @classmethod
    def _ratio_range(cls, v: float) -> float:
        """割合が 0.0 から 1.0 の範囲内か検証する。"""

        if not 0.0 <= v <= 1.0:
            raise ValueError("recent_ratio は 0.0 以上 1.0 以下に設定してください")
        return v


class TrainingConfig(BaseModel):
    batch_size: int = 32
    num_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.75
    num_games: int = 500
    num_train_steps: int = 50
    vs_random_matches: int = 100
    scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)
    artifacts_dir: str = "artifacts"
    enable_tensorboard: bool = True
    metrics_filename: str = "training_metrics.csv"
    latest_checkpoint: str = "latest.pt"
    best_checkpoint: str = "best.pt"
    replay_buffer: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig)

class AppConfig(BaseModel):
    game: GameConfig = Field(default_factory=GameConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
