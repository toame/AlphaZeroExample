"""MCTS のフィルタ機能に関するテスト。"""

from __future__ import annotations

import numpy as np

from config import GameConfig, MCTSConfig
from config.models import GameAxes

from alphazero import State, Tree


class _UniformNet:
    """一様分布を返すダミーネットワーク。"""

    def predict(self, state: State) -> tuple[np.ndarray, float]:
        board_len = state.size * state.size
        policy = np.full(board_len, 1.0 / board_len, dtype=np.float32)
        return policy, 0.0


def _connect6_config(board_size: int = 6) -> GameConfig:
    """テスト用の connect6 設定を生成する。"""

    axes = GameAxes.auto(board_size)
    return GameConfig(board_size=board_size, rule="connect6", axes=axes)


def test_proximity_bias_activates_on_third_move() -> None:
    """3 手目以降で近接バイアスが発動し、近い手の確率が高まることを確認する。"""

    config = _connect6_config(6)
    state = State(config)
    state.play(state.str2action("A1"))  # 黒初手
    state.play(state.str2action("B1"))  # 白の 1 手目

    tree = Tree(_UniformNet(), MCTSConfig())
    policy = np.full(config.board_size * config.board_size, 1.0, dtype=np.float32)
    filtered = tree._apply_move_filters(state, policy)

    near_action = state.str2action("A2")
    far_action = state.str2action("F6")

    assert filtered[near_action] > filtered[far_action]


def test_initial_state_remains_uniform() -> None:
    """初期盤面ではフィルタ適用前後で一様分布が保たれることを確認する。"""

    config = _connect6_config(6)
    state = State(config)

    tree = Tree(_UniformNet(), MCTSConfig())
    policy = np.full(config.board_size * config.board_size, 1.0, dtype=np.float32)
    filtered = tree._apply_move_filters(state, policy)

    legal = state.legal_actions()
    expected = 1.0 / len(legal)
    for action in legal:
        assert np.isclose(filtered[action], expected)


def test_think_prioritizes_immediate_win() -> None:
    """勝ち筋が存在する場合にその手が必ず選ばれることを確認する。"""

    config = _connect6_config(6)
    state = State(config)

    state.play(state.str2action("A1"))
    state.play(state.str2action("F1"))
    state.play(state.str2action("F3"))
    state.play(state.str2action("A2"))
    state.play(state.str2action("A3"))
    state.play(state.str2action("E1"))
    state.play(state.str2action("E3"))
    state.play(state.str2action("A4"))
    state.play(state.str2action("A5"))
    state.play(state.str2action("D1"))
    state.play(state.str2action("D3"))

    tree = Tree(_UniformNet(), MCTSConfig())
    probs = tree.think(state, num_simulations=1)

    winning_action = state.str2action("A6")
    assert np.isclose(probs[winning_action], 1.0)
