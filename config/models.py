# config/models.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class GameAxes(BaseModel):
    X: str = "ABC"
    Y: str = "123"

class GameSymbols(BaseModel):
    empty: str = "_"
    black: str = "O"
    white: str = "X"

class GameConfig(BaseModel):
    board_size: int = 3
    rule: Literal["tic_tac_toe", "connect6"] = "tic_tac_toe"
    axes: GameAxes = Field(default_factory=GameAxes)
    symbols: GameSymbols = Field(default_factory=GameSymbols)
    first_player: Literal[-1, 1] = 1

    @field_validator("board_size")
    @classmethod
    def _board_size_rule_check(cls, v: int, info):
        """ルールごとの最小盤面サイズを検証する。"""
        rule = info.data.get("rule", "tic_tac_toe")
        if rule == "connect6" and v < 6:
            raise ValueError("connect6 ルールでは board_size を 6 以上に設定してください")
        return v

    @field_validator("axes")
    @classmethod
    def _axes_len_match_board(cls, v: GameAxes, info):
        board_size = info.data.get("board_size", 3)
        if len(v.X) != board_size or len(v.Y) != board_size:
            raise ValueError(f"axes.X/Y の長さは board_size={board_size} と一致させてください")
        return v

class NetworkConfig(BaseModel):
    num_filters: int = 16
    num_blocks: int = 6
    policy_channels: int = 4
    value_channels: int = 4

class MCTSConfig(BaseModel):
    dirichlet_alpha: float = 0.15
    dirichlet_weight: float = 0.25
    puct_c: float = 2.0
    temperature_init: float = 0.7
    num_simulations_demo: int = 1000
    num_simulations_demo_mid: int = 3000
    num_simulations_train: int = 50

class LRSchedulerConfig(BaseModel):
    type: Literal["none", "cosine", "onecycle"] = "cosine"
    cosine_min_lr: float = 1e-5
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4


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

class AppConfig(BaseModel):
    game: GameConfig = Field(default_factory=GameConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
