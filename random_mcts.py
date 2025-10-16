"""ニューラルネットワークを用いない乱数ベースのモンテカルロ木探索。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        if game_config.rule == "connect6":
            self._connect_length = 6
        else:
            self._connect_length = 3
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

    def _collect_candidate_actions(
        self,
        state: State,
        legal_actions: Sequence[int],
        *,
        use_initial_radius: bool = True,
    ) -> List[int]:
        """候補抽出ロジックを共通化した内部ヘルパー。"""

        if not legal_actions:
            return []

        if self._candidate_radius is None:
            return list(legal_actions)

        if len(state.record) > state.size * 2:
            return list(legal_actions)

        candidates: set[int] = set()
        size = state.size

        if state.record:
            radius = max(int(self._candidate_radius), 0)
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
            if use_initial_radius and self._initial_radius is not None:
                radius = int(self._initial_radius)
            else:
                radius = int(self._candidate_radius or 0)
            radius = max(radius, 0)
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
            return list(legal_actions)
        return sorted(candidates)

    def _select_candidates(self, state: State) -> Sequence[int]:
        """盤面に応じて探索対象の候補手を抽出する。"""

        legal = state.legal_actions()
        if not legal:
            return []

        analysis_candidates = self._collect_candidate_actions(
            state, legal, use_initial_radius=False
        )
        if self._should_check_forced_defense(state, legal, analysis_candidates):
            defensive = self._find_forced_defense_actions(state, legal)
            if defensive is not None:
                return defensive

        candidates = self._collect_candidate_actions(state, legal)
        return candidates

    def _select_random_action(self, state: State) -> Optional[int]:
        """ランダムロールアウト用の候補手から 1 手選択する。"""

        legal = state.legal_actions()
        if not legal:
            return None
        analysis_candidates = self._collect_candidate_actions(
            state, legal, use_initial_radius=False
        )
        defensive: Optional[Sequence[int]] = None
        if self._should_check_forced_defense(state, legal, analysis_candidates):
            defensive = self._find_forced_defense_actions(state, legal)

        if defensive:
            pool = list(defensive)
        else:
            candidates = self._collect_candidate_actions(state, legal)
            pool = list(candidates) if candidates else list(legal)
        return self._rng.choice(pool)

    def _should_check_forced_defense(
        self,
        state: State,
        legal_actions: Sequence[int],
        candidate_scope: Optional[Sequence[int]] = None,
    ) -> bool:
        """防御判定を行うべきかを簡易に判定する。"""

        # 序盤は合法手が極端に多く、毎回の防御判定コストが高すぎるため、
        # ある程度盤面が進行したときにのみ詳細な判定を行う。
        if len(legal_actions) <= self._forced_loss_check_limit:
            return True

        if len(state.record) >= state.size * 2:
            return True

        if not state.record:
            return len(legal_actions) <= self._forced_loss_check_limit

        # connect6 では同一ターンに複数手を打てるため、ターン中に残り手数が
        # 少ない場合は早めに防御判定を行い、受け漏れを防ぐ。ただし合法手が
        # 極端に多い序盤で同じ判定を行うと 1 手ごとに 300 以上の候補を精査
        # することになり、シミュレーションが大幅に遅延してしまう。そこで
        # 序盤のように合法手が多すぎる局面では防御判定をスキップし、候補手
        # がある程度に絞られている場合のみ早期判定を有効にする。
        stones_remaining = getattr(state, "_stones_remaining", 1)

        scope = (
            list(candidate_scope)
            if candidate_scope is not None
            else self._collect_candidate_actions(
                state, legal_actions, use_initial_radius=False
            )
        )
        candidate_count = len(scope)

        if candidate_count <= self._forced_loss_check_limit:
            return True

        if (
            candidate_count <= self._forced_loss_check_limit * 2
            and candidate_count < len(legal_actions)
        ):
            return True

        if self._candidate_radius is None:
            relevant_empty = self._estimate_relevant_empty_points(state)
            if relevant_empty <= self._forced_loss_check_limit:
                return True
            if stones_remaining <= 1 and relevant_empty <= self._forced_loss_check_limit * 2:
                return True

        return False

    def _estimate_relevant_empty_points(self, state: State) -> int:
        """石が存在する領域を囲う矩形内の空点数を概算する。"""

        if not state.record:
            return len(state.legal_actions())

        size = state.size
        min_x = size - 1
        max_x = 0
        min_y = size - 1
        max_y = 0

        for action in state.record:
            x = action // size
            y = action % size
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

        margin = 2
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, size - 1)
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, size - 1)

        empty_points = 0
        board = state.board
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if board[x, y] == 0:
                    empty_points += 1
        return empty_points

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
        if not actions:
            return []

        stones_to_place = getattr(state, "_stones_remaining", 1)
        if stones_to_place <= 0:
            return []

        winning_pool = self._collect_line_based_winning_actions(
            state, state.color, stones_to_place
        )
        target_set = set(actions)
        return sorted(a for a in winning_pool if a in target_set)

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
        opponent_legal = pass_state.legal_actions()
        candidates = self._collect_candidate_actions(
            pass_state, opponent_legal, use_initial_radius=False
        )
        use_filter = len(candidates) < len(opponent_legal)
        target_actions: Sequence[int]
        if use_filter and candidates:
            target_actions = candidates
        else:
            target_actions = opponent_legal
            use_filter = False
        return self._find_immediate_wins(
            pass_state,
            target_actions,
            use_candidate_filter=use_filter,
        )

    # --- 線分解析ロジック ---

    def _collect_line_based_winning_actions(
        self, state: State, color: int, stones_to_place: int
    ) -> set[int]:
        """指定した色が同一ターンで勝てる交点を線分単位で列挙する。"""

        if stones_to_place <= 0:
            return set()

        size = state.size
        board = state.board
        connect = self._connect_length
        winning: set[int] = set()

        for line in self._iterate_lines(size):
            if len(line) < connect:
                continue
            values = [board[x, y] for x, y in line]
            for i in range(len(line) - connect + 1):
                segment_coords = line[i : i + connect]
                segment_values = values[i : i + connect]
                if any(v == -color for v in segment_values):
                    continue
                empties: List[Tuple[int, int]] = [
                    segment_coords[j]
                    for j, v in enumerate(segment_values)
                    if v == 0
                ]
                if not empties:
                    # 既に勝っている場合は勝ち確定として扱う。
                    for x, y in segment_coords:
                        winning.add(x * size + y)
                    continue
                if len(empties) > stones_to_place:
                    continue
                for ex, ey in empties:
                    winning.add(ex * size + ey)
        return winning

    def _iterate_lines(self, size: int) -> Iterable[List[Tuple[int, int]]]:
        """縦横斜めの各線分の座標列を生成する。"""

        for x in range(size):
            yield [(x, y) for y in range(size)]
        for y in range(size):
            yield [(x, y) for x in range(size)]

        for start in range(size):
            coords: List[Tuple[int, int]] = []
            x, y = start, 0
            while 0 <= x < size and 0 <= y < size:
                coords.append((x, y))
                x += 1
                y += 1
            yield coords
        for start in range(1, size):
            coords = []
            x, y = 0, start
            while 0 <= x < size and 0 <= y < size:
                coords.append((x, y))
                x += 1
                y += 1
            yield coords

        for start in range(size):
            coords = []
            x, y = start, size - 1
            while 0 <= x < size and 0 <= y < size:
                coords.append((x, y))
                x += 1
                y -= 1
            yield coords
        for start in range(1, size):
            coords = []
            x, y = 0, size - 1 - start
            while 0 <= x < size and 0 <= y < size:
                coords.append((x, y))
                x += 1
                y -= 1
            yield coords
