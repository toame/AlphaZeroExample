"""乱数ベース MCTS の基本的な挙動を検証する。"""

from __future__ import annotations

from app.random_match import play_random_mcts_vs_random
from config.loader import load_default_config
from game import State
from random_mcts import RandomMCTSAgent


def test_random_mcts_selects_legal_action() -> None:
    """初期局面で選択された手が合法手であることを確認する。"""

    cfg = load_default_config()
    state = State(cfg.game)
    agent = RandomMCTSAgent(cfg.game, seed=123)
    action = agent.select_action(state, num_simulations=5)
    assert action in state.legal_actions()


def test_play_random_mcts_vs_random_runs() -> None:
    """対戦関数が指定局数分の結果を返すことを確認する。"""

    cfg = load_default_config()
    results = play_random_mcts_vs_random(cfg, simulations=5, games=3, seed=42)
    assert set(results.keys()) == {"mcts_win", "random_win", "draw"}
    assert sum(results.values()) == 3
