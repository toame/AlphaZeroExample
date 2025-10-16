"""ニューラルネットワークを用いない乱数ベースのモンテカルロ木探索。"""

from __future__ import annotations

import itertools
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
        # 強制敗北判定は合法手が多すぎる局面で行うと計算量が跳ね上がるため、
        # ある程度盤面が進んだ局面のみに限定して実行する。
        self._forced_loss_check_limit = 30

    def select_action(self, state: State, num_simulations: int) -> int:
        """指定回数の探索から最善手を選択する。"""

        if not state.legal_actions():
            raise ValueError("合法手が存在しない局面では探索できません。")

        self._root = self._create_node(state)
        for _ in range(num_simulations):
            self._run_simulation(state)

        if not self._root.children:
            legal_actions = state.legal_actions()
            if not legal_actions:
                raise ValueError("合法手が存在しない局面では探索できません。")
            forced_value = self._detect_forced_outcome(state, self._root.player)
            winning_actions = self._find_immediate_wins(state, legal_actions)
            if winning_actions:
                best_action = winning_actions[0]
                win_rate = 1.0
            else:
                best_action = legal_actions[0]
                if forced_value is None:
                    win_rate = 0.5
                else:
                    win_rate = max(0.0, min(1.0, (forced_value + 1.0) / 2.0))
            self._last_report = RandomSearchReport(
                best_action=best_action,
                win_rate=win_rate,
                visit_count=0,
                total_visits=0,
                simulations=num_simulations,
            )
            return best_action

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

            forced_value = self._detect_forced_outcome(state, node.player)
            if forced_value is not None:
                value = forced_value
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
                    forced_value = self._detect_forced_outcome(state, node.player)
                    if forced_value is not None:
                        value = forced_value
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
            forced_value = self._detect_forced_outcome(rollout_state, target_player)
            if forced_value is not None:
                return forced_value
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

        if self._should_check_forced_defense(state, legal):
            defensive = self._find_forced_defense_actions(state, legal)
            if defensive is not None:
                return defensive

        if self._candidate_radius is None:
            return legal
        # 石数が一定以上の局面では候補手の抽出を省略し、全合法手から選ぶ。
        # 盤面が広く埋まると候補手の算出コストが大きくなるため、
        # 探索を継続することを優先して閾値を設けている。
        if len(state.record) > state.size * 2:
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

        legal = state.legal_actions()
        if not legal:
            return None
        defensive: Optional[Sequence[int]] = None
        if self._should_check_forced_defense(state, legal):
            defensive = self._find_forced_defense_actions(state, legal)

        if defensive:
            pool = list(defensive)
        elif self._candidate_radius is None:
            pool = legal
        else:
            candidates = list(self._select_candidates(state))
            pool = candidates or legal
        return self._rng.choice(pool)

    def _should_check_forced_defense(
        self, state: State, legal_actions: Sequence[int]
    ) -> bool:
        """防御判定を行うべきかを簡易に判定する。"""

        # 序盤は合法手が極端に多く、毎回の防御判定コストが高すぎるため、
        # ある程度盤面が進行したときにのみ詳細な判定を行う。
        if len(legal_actions) <= self._forced_loss_check_limit:
            return True

        if len(state.record) >= state.size * 2:
            return True

        # connect6 では同一ターンに複数手を打てるため、ターン中に残り手数が
        # 少ない場合は早めに防御判定を行い、受け漏れを防ぐ。ただし合法手が
        # 極端に多い序盤で同じ判定を行うと 1 手ごとに 300 以上の候補を精査
        # することになり、シミュレーションが大幅に遅延してしまう。そこで
        # 序盤のように合法手が多すぎる局面では防御判定をスキップし、候補手
        # がある程度に絞られている場合のみ早期判定を有効にする。
        stones_remaining = getattr(state, "_stones_remaining", 1)
        if stones_remaining <= 1:
            return len(legal_actions) <= self._forced_loss_check_limit * 2

        return False

    def _detect_forced_outcome(self, state: State, target_player: int) -> Optional[float]:
        """強制的な勝敗が決まっているかを判定し、値を返す。"""

        if state.win_color != 0:
            return 1.0 if state.win_color == target_player else -1.0

        legal_actions = state.legal_actions()
        if not legal_actions:
            return None

        if len(legal_actions) <= self._forced_loss_check_limit * 2:
            winning_actions = self._find_immediate_wins(state, legal_actions)
            if winning_actions:
                return 1.0 if state.color == target_player else -1.0

        if len(legal_actions) > self._forced_loss_check_limit:
            return None

        if self._is_forced_loss(state, target_player, legal_actions=legal_actions):
            return -1.0 if state.color == target_player else 1.0
        return None

    def _find_immediate_wins(
        self,
        state: State,
        legal_actions: Sequence[int] | None = None,
        *,
        use_candidate_filter: bool = True,
    ) -> List[int]:
        """現在手番が同一ターン内で確実に勝てる手を列挙する。"""

        actions = list(legal_actions) if legal_actions is not None else state.legal_actions()
        if (
            use_candidate_filter
            and len(actions) > self._forced_loss_check_limit
            and self._candidate_radius is not None
        ):
            # 序盤など合法手が極端に多い局面では候補手に絞って高速化する。
            candidate_actions = list(self._select_candidates(state))
            if candidate_actions:
                candidate_set = set(candidate_actions)
                filtered = [action for action in actions if action in candidate_set]
                if filtered:
                    actions = filtered
        if not actions:
            return []

        winning_actions: List[int] = []
        stones_to_place = getattr(state, "_stones_remaining", 1)

        for action in actions:
            next_state = state.copy()
            next_state.play(action)
            if next_state.win_color == state.color:
                winning_actions.append(action)

        if winning_actions or stones_to_place <= 1:
            return winning_actions

        # 合法手が多すぎる局面で多手順の即勝ち探索を行うと計算量が爆発する。
        # connect6 では序盤に同一ターンで勝てる局面は現れないため、
        # 一定以上の手数が残っている場合は探索を打ち切り、
        # それ以外の状況だけ詳細に調べる。
        if len(actions) > self._forced_loss_check_limit:
            return winning_actions

        winning_first_actions = set()
        for action in actions:
            if action in winning_first_actions:
                continue
            next_state = state.copy()
            next_state.play(action)
            remaining = stones_to_place - 1
            if remaining <= 0:
                continue
            follow_actions = next_state.legal_actions()
            if not follow_actions:
                continue
            for combo in itertools.combinations(follow_actions, remaining):
                branch_state = next_state.copy()
                for follow_action in combo:
                    branch_state.play(follow_action)
                    if branch_state.win_color == state.color:
                        winning_first_actions.add(action)
                        break
                if action in winning_first_actions:
                    break

        for action in actions:
            if action in winning_first_actions:
                winning_actions.append(action)
        return winning_actions

    def _is_forced_loss(
        self,
        state: State,
        target_player: int,
        *,
        legal_actions: Sequence[int] | None = None,
        depth: int = 0,
    ) -> bool:
        """任意の応手でも相手に即勝される場合は必敗とみなす。"""

        if state.win_color != 0:
            return state.win_color != target_player

        if state.color != target_player:
            opponent_wins = self._find_immediate_wins(state)
            return bool(opponent_wins)

        actions = list(legal_actions) if legal_actions is not None else state.legal_actions()
        if not actions:
            return True

        if depth >= 3:
            # connect6 の同一ターン内で最大 2 手までを想定し、安全側に倒す。
            return False

        for action in actions:
            next_state = state.copy()
            next_state.play(action)

            if next_state.win_color == target_player:
                return False

            if next_state.color == target_player:
                next_legal = next_state.legal_actions()
                if not next_legal:
                    return True
                if len(next_legal) > self._forced_loss_check_limit:
                    return False
                if not self._is_forced_loss(
                    next_state,
                    target_player,
                    legal_actions=next_legal,
                    depth=depth + 1,
                ):
                    return False
            else:
                opponent_legal = next_state.legal_actions()
                if not opponent_legal:
                    return False
                if not self._find_immediate_wins(next_state, opponent_legal):
                    return False

        return True

    def _find_forced_defense_actions(
        self, state: State, legal_actions: Sequence[int]
    ) -> Optional[List[int]]:
        """パスすると相手が即勝する局面では受けの手に絞り込む。"""

        if getattr(state, "_stones_remaining", 1) <= 0:
            return None

        # 序盤など合法手が極端に多い場合は全候補を精査すると計算量が膨大に
        # なるため、受け判定を諦めて通常の候補手抽出に委ねる。
        if len(legal_actions) > self._forced_loss_check_limit * 2:
            return None

        opponent_wins = self._find_opponent_immediate_wins_if_pass(state)
        if not opponent_wins:
            return None

        defensive = sorted(set(legal_actions) & set(opponent_wins))
        if defensive:
            return defensive
        return None

    def _find_opponent_immediate_wins_if_pass(self, state: State) -> List[int]:
        """このターンで何も打たなければ相手が確実に勝つ手を列挙する。"""

        pass_state = state.copy()
        if getattr(pass_state, "_stones_remaining", 0) > 0:
            pass_state._stones_remaining = 0
        pass_state._end_turn()
        return self._find_immediate_wins(pass_state, use_candidate_filter=False)
