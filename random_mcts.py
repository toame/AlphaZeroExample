"""ニューラルネットワークを用いない乱数ベースのモンテカルロ木探索。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from config import GameConfig
from game import State


@dataclass
class RandomNode:
    """乱数モンテカルロ木探索で使用するノード。"""

    player: int
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, "RandomNode"] = field(default_factory=dict)
    untried_actions: List[int] = field(default_factory=list)

    def add_child(self, action: int, child: "RandomNode") -> None:
        """子ノードを追加する。"""

        self.children[action] = child


@dataclass
class RandomSearchReport:
    """探索結果の概要をまとめたデータ構造。"""

    best_action: int
    win_rate: float
    visit_count: int
    total_visits: int
    simulations: int


class RandomMCTSAgent:
    """乱数ロールアウトで評価するシンプルなモンテカルロ木探索プレイヤー。"""

    def __init__(
        self,
        game_config: GameConfig,
        *,
        exploration_c: float = 1.4,
        rollout_limit: Optional[int] = None,
        seed: Optional[int] = None,
        candidate_radius: Optional[int] = None,
        initial_radius: Optional[int] = None,
    ) -> None:
        self._c = exploration_c
        self._rollout_limit = rollout_limit
        self._rng = random.Random(seed)
        self._root: Optional[RandomNode] = None
        self._candidate_radius = candidate_radius
        self._initial_radius = initial_radius
        self._last_report: Optional[RandomSearchReport] = None

    def select_action(self, state: State, num_simulations: int) -> int:
        """指定回数の探索から最善手を選択する。"""

        if not state.legal_actions():
            raise ValueError("合法手が存在しない局面では探索できません。")

        self._root = self._create_node(state)
        for _ in range(num_simulations):
            self._run_simulation(state)

        assert self._root.children, "子ノードが生成されていません。"
        best_action, best_child = max(
            self._root.children.items(),
            key=lambda item: self._child_value_from_parent(self._root, item[1]),
        )

        total_visits = sum(child.visits for child in self._root.children.values())
        best_value = self._child_value_from_parent(self._root, best_child)
        win_rate = max(0.0, min(1.0, (best_value + 1.0) / 2.0))
        self._last_report = RandomSearchReport(
            best_action=best_action,
            win_rate=win_rate,
            visit_count=best_child.visits,
            total_visits=total_visits,
            simulations=num_simulations,
        )
        return best_action

    @property
    def last_report(self) -> Optional[RandomSearchReport]:
        """直近の探索結果を返す。未探索時は ``None``。"""

        return self._last_report

    def _run_simulation(self, root_state: State) -> None:
        """1 回分のシミュレーションを実行する。"""

        assert self._root is not None
        state = root_state.copy()
        node = self._root
        path: List[tuple[RandomNode, RandomNode]] = []

        while True:
            if state.terminal():
                value = self._evaluate_terminal(state, node.player)
                break

            if node.untried_actions:
                action = self._rng.choice(node.untried_actions)
                node.untried_actions.remove(action)
                state.play(action)
                child = self._create_node(state)
                node.add_child(action, child)
                path.append((node, child))
                node = child
                if state.terminal():
                    value = self._evaluate_terminal(state, node.player)
                else:
                    value = self._rollout(state, node.player)
                break

            action = self._select_child_action(node)
            state.play(action)
            child = node.children[action]
            path.append((node, child))
            node = child

        self._update_node(node, value)
        for parent, child in reversed(path):
            if parent.player != child.player:
                value = -value
            self._update_node(parent, value)

    def _create_node(self, state: State) -> RandomNode:
        """現在局面からノードを生成する。"""

        return RandomNode(
            player=state.color,
            visits=0,
            value_sum=0.0,
            children={},
            untried_actions=list(self._select_candidates(state)),
        )

    def _select_child_action(self, node: RandomNode) -> int:
        """UCT に基づいて子ノードを選択する。"""

        assert node.children, "子ノードが存在しません。"
        best_action = None
        best_score = -float("inf")
        parent_visits = max(node.visits, 1)

        for action, child in node.children.items():
            exploitation = self._child_value_from_parent(node, child)
            exploration = self._c * math.sqrt(math.log(parent_visits + 1) / (child.visits + 1))
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_action = action

        assert best_action is not None, "UCT による子選択に失敗しました。"
        return best_action

    def _child_value_from_parent(self, parent: RandomNode, child: RandomNode) -> float:
        """親ノード視点で子ノードの平均価値を返す。"""

        if child.visits == 0:
            return 0.0
        value = child.value_sum / child.visits
        if parent.player != child.player:
            value = -value
        return value

    def _rollout(self, state: State, target_player: int) -> float:
        """終局までランダムに手を進め評価値を得る。"""

        rollout_state = state.copy()
        steps = 0
        while not rollout_state.terminal():
            action = self._select_random_action(rollout_state)
            if action is None:
                break
            rollout_state.play(action)
            steps += 1
            if self._rollout_limit is not None and steps >= self._rollout_limit:
                break
        return self._evaluate_terminal(rollout_state, target_player)

    def _evaluate_terminal(self, state: State, target_player: int) -> float:
        """終局局面からターゲット視点の報酬を計算する。"""

        winner = state.win_color
        if winner == 0:
            return 0.0
        return 1.0 if winner == target_player else -1.0

    def _update_node(self, node: RandomNode, value: float) -> None:
        """訪問回数と累積価値を更新する。"""

        node.visits += 1
        node.value_sum += value

    def _select_candidates(self, state: State) -> Sequence[int]:
        """盤面に応じて探索対象の候補手を抽出する。"""

        legal = state.legal_actions()
        if not legal:
            return []
        if self._candidate_radius is None:
            return legal

        candidates = set()
        size = state.size

        if state.record:
            radius = max(self._candidate_radius, 0)
            for action in state.record:
                x = action // size
                y = action % size
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx = x + dx
                        ny = y + dy
                        if not (0 <= nx < size and 0 <= ny < size):
                            continue
                        if state.board[nx, ny] != 0:
                            continue
                        candidates.add(nx * size + ny)
        else:
            radius = self._initial_radius
            if radius is None:
                radius = self._candidate_radius
            radius = max(radius or 0, 0)
            center = size // 2
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx = center + dx
                    ny = center + dy
                    if not (0 <= nx < size and 0 <= ny < size):
                        continue
                    if state.board[nx, ny] != 0:
                        continue
                    candidates.add(nx * size + ny)

        if not candidates:
            return legal
        return sorted(candidates)

    def _select_random_action(self, state: State) -> Optional[int]:
        """ランダムロールアウト用の候補手から 1 手選択する。"""

        candidates = self._select_candidates(state)
        if not candidates:
            return None
        return self._rng.choice(list(candidates))
