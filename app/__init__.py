"""アプリケーション全体のエントリーポイント関連モジュール。"""

__all__ = [
    "demo_network_outputs",
    "demo_mcts",
    "self_play_and_train",
]

from .demo import demo_network_outputs, demo_mcts
from .training_loop import self_play_and_train
