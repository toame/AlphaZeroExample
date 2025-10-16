"""乱数ベース MCTS の基本的な挙動を検証する。"""

from __future__ import annotations

from app.random_match import play_random_mcts_vs_random
from config import GameConfig
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
    report = agent.last_report
    assert report is not None
    assert report.best_action == action
    assert 0.0 <= report.win_rate <= 1.0
    assert report.total_visits >= report.visit_count >= 0
    assert report.simulations == 5


def test_play_random_mcts_vs_random_runs() -> None:
    """対戦関数が指定局数分の結果を返すことを確認する。"""

    cfg = load_default_config()
    results = play_random_mcts_vs_random(cfg, simulations=5, games=3, seed=42)
    assert set(results.keys()) == {"mcts_win", "random_win", "draw"}
    assert sum(results.values()) == 3


def test_random_mcts_connect6_can_select_action() -> None:
    """connect6 盤面でも候補制限付き探索が動作することを確認する。"""

    game_cfg = GameConfig(rule="connect6", board_size=19)
    state = State(game_cfg)
    agent = RandomMCTSAgent(
        game_cfg,
        seed=321,
        candidate_radius=2,
        initial_radius=3,
        rollout_limit=10,
    )
    action = agent.select_action(state, num_simulations=2)
    assert action in state.legal_actions()


def test_random_mcts_detects_immediate_win() -> None:
    """必勝手を検出して勝ち切り評価を返すことを確認する。"""

    cfg = load_default_config()
    state = State(cfg.game)
    # X が二目並べの上段で 2 連を作った局面。
    for action in [0, 3, 1, 4]:
        state.play(action)

    agent = RandomMCTSAgent(cfg.game, seed=0)
    value = agent._detect_forced_outcome(state, state.color)
    assert value == 1.0


def test_random_mcts_detects_forced_loss() -> None:
    """どの応手でも相手に即勝される局面を必敗として判定する。"""

    cfg = load_default_config()
    state = State(cfg.game)
    # 白番 (X) が上段と左列で同時にリーチしており、黒番 (O) が片方しか受けられない局面を人工的に構成する。
    state.board.fill(0)
    state.board[0, 0] = -1
    state.board[0, 1] = -1
    state.board[1, 0] = -1
    state.board[1, 1] = 1
    state.board[2, 2] = 1
    state.record = [4, 0, 8, 1, 3]
    state.turn_index = len(state.record)
    state.color = 1
    state.win_color = 0
    state._stones_remaining = 1

    agent = RandomMCTSAgent(cfg.game, seed=0)
    value = agent._detect_forced_outcome(state, state.color)
    assert value == -1.0
