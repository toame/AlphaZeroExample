# mcts.py
from __future__ import annotations
import copy
import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List
from config import MCTSConfig
from game import State

class Node:
    """ある1状態の探索結果を保存するノード。"""
    def __init__(self, p: np.ndarray, v: float) -> None:
        self.p = p.astype(np.float64)
        self.v = float(v)
        self.n = np.zeros_like(p, dtype=np.int32)
        self.q_sum = np.zeros_like(p, dtype=np.float64)
        self.n_all = 1
        self.q_sum_all = v / 2.0

    def update(self, action: int, q_new: float) -> None:
        self.n[action] += 1
        self.q_sum[action] += q_new
        self.n_all += 1
        self.q_sum_all += q_new

@dataclass(frozen=True)
class SearchReport:
    """探索途中経過を外部に知らせるための情報。"""

    elapsed: float
    best_action: int | None
    best_q: float | None
    best_n: int | None
    total_simulations: int
    pv: List[int]

ProgressCallback = Callable[[State, SearchReport], None]

class Tree:
    """モンテカルロ木探索本体。"""

    def __init__(
        self,
        net,
        config: MCTSConfig,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.net = net
        self.nodes: Dict[str, Node] = {}
        self._config = config
        self._progress_callback = progress_callback

    def search(self, state: State, depth: int) -> float:
        """モンテカルロ木探索の 1 プレイアウトを実行する。"""

        path: List[tuple[Node, int]] = []
        current_depth = depth

        while True:
            if state.terminal():
                value = float(state.terminal_reward())
                break

            key = state.record_string()
            if key not in self.nodes:
                p, v = self.net.predict(state)
                self.nodes[key] = Node(p, v)
                value = float(v)
                break

            node = self.nodes[key]
            p = node.p.copy()
            if current_depth == 0:
                p = (1.0 - self._config.dirichlet_weight) * p + self._config.dirichlet_weight * np.random.dirichlet(
                    [self._config.dirichlet_alpha] * len(p)
                )

            best_action, best_ucb = None, -float("inf")
            for action in state.legal_actions():
                n = 1 + node.n[action]
                q_sum = node.q_sum_all / node.n_all + node.q_sum[action]
                ucb = q_sum / n + self._config.puct_c * np.sqrt(node.n_all) * p[action] / n
                if ucb > best_ucb:
                    best_action, best_ucb = action, ucb

            assert best_action is not None, "合法手が存在しない状態で探索が進行しました。"
            path.append((node, best_action))
            state.play(best_action)
            current_depth += 1

        backup_value = value
        for node, action in reversed(path):
            backup_value = -backup_value
            node.update(action, backup_value)

        return float(backup_value)

    def think(
        self,
        state: State,
        num_simulations: int,
        temperature: float | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> np.ndarray:
        if temperature is None:
            temperature = self._config.temperature_init
        callback = progress_callback or self._progress_callback
        start, prev_time = time.time(), 0.0
        for _ in range(num_simulations):
            self.search(copy.deepcopy(state), depth=0)
            if callback is not None:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root = self.nodes[state.record_string()]
                    pv = self.pv(state)
                    best_action = pv[0] if pv else None
                    best_q = None
                    best_n = None
                    if best_action is not None:
                        visits = max(root.n[best_action], 1)
                        best_q = root.q_sum[best_action] / visits
                        best_n = root.n[best_action]
                    callback(
                        state,
                        SearchReport(
                            elapsed=tmp_time,
                            best_action=best_action,
                            best_q=best_q,
                            best_n=best_n,
                            total_simulations=root.n_all,
                            pv=pv,
                        ),
                    )

        root_n = self.nodes[state.record_string()].n.astype(np.float64)
        max_visit = np.max(root_n)
        if max_visit == 0:
            probs = np.ones_like(root_n, dtype=np.float64) / len(root_n)
        else:
            scaled = (root_n / max_visit) ** (1.0 / (temperature + 1e-8))
            probs = scaled / scaled.sum()
        return probs

    def pv(self, state: State) -> List[int]:
        s = copy.deepcopy(state)
        pv_seq = []
        while True:
            key = s.record_string()
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            legal = s.legal_actions()
            best_action = sorted([(a, self.nodes[key].n[a]) for a in legal], key=lambda x: -x[1])[0][0]
            pv_seq.append(best_action)
            s.play(best_action)
        return pv_seq
