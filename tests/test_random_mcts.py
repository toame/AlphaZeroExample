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


def test_random_mcts_checks_forced_defense_in_sparse_area() -> None:
    """合法手が多くても局所的な危険があれば防御判定を実行する。"""

    game_cfg = GameConfig(rule="connect6", board_size=19)
    state = State(game_cfg)
    size = state.size

    # 初手から互いに石を配置しつつ、白番が横一列に 5 連を作る局面を構築する。
    opening_sequence = [
        size * 0 + 0,  # 先手の初手
        size * 9 + 4,
        size * 9 + 5,
        size * 1 + 0,
        size * 1 + 1,
        size * 9 + 6,
        size * 9 + 7,
        size * 2 + 0,
        size * 2 + 1,
        size * 9 + 8,
        size * 8 + 4,
    ]
    for action in opening_sequence:
        state.play(action)

    agent = RandomMCTSAgent(
        game_cfg,
        seed=0,
        candidate_radius=2,
        initial_radius=3,
    )
    legal = state.legal_actions()

    # 広い盤面で合法手は 300 以上残っているが、候補手は局所に集中する。
    assert len(legal) > agent._forced_loss_check_limit * 2
    analysis_candidates = agent._collect_candidate_actions(
        state, legal, use_initial_radius=False
    )
    assert len(analysis_candidates) <= agent._forced_loss_check_limit * 2

    assert agent._should_check_forced_defense(state, legal, analysis_candidates)
    defensive = agent._find_forced_defense_actions(state, legal)
    assert defensive is not None

    expected_defense = {size * 9 + 3, size * 9 + 9}
    assert expected_defense.issubset(set(defensive))


def test_random_mcts_detects_connect6_open_four_finish() -> None:
    """開放四を作ったターンで両端を詰めれば勝てることを検出する。"""

    game_cfg = GameConfig(rule="connect6", board_size=19)
    state = State(game_cfg)
    size = state.size
    row = 6
    start = 7

    stones = []
    for offset in range(4):
        idx = row * size + (start + offset)
        stones.append(idx)
        x, y = divmod(idx, size)
        state.board[x, y] = 1

    state.record = stones
    state.turn_index = len(stones)
    state.color = 1
    state.win_color = 0
    state._stones_remaining = 2

    agent = RandomMCTSAgent(game_cfg, seed=0)
    winning = agent._find_immediate_wins(state)

    expected = {row * size + (start - 1), row * size + (start + 4)}
    assert expected.issubset(set(winning))


def test_random_mcts_forced_defense_blocks_open_four() -> None:
    """相手の開放四に対し両端を受けとして提示する。"""

    game_cfg = GameConfig(rule="connect6", board_size=19)
    state = State(game_cfg)
    size = state.size
    row = 10
    start = 5

    stones = []
    for offset in range(4):
        idx = row * size + (start + offset)
        stones.append(idx)
        x, y = divmod(idx, size)
        state.board[x, y] = -1

    state.record = stones
    state.turn_index = len(stones)
    state.color = 1
    state.win_color = 0
    state._stones_remaining = 2

    agent = RandomMCTSAgent(game_cfg, seed=0)
    legal = state.legal_actions()
    defensive = agent._find_forced_defense_actions(state, legal)

    assert defensive is not None
    expected = {row * size + (start - 1), row * size + (start + 4)}
    assert expected.issubset(set(defensive))
