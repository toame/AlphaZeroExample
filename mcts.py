# mcts.py
from __future__ import annotations
import copy
import time
import numpy as np
from typing import Dict
from game import State
from config.loader import cfg

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

class Tree:
    """モンテカルロ木探索本体。"""
    def __init__(self, net) -> None:
        self.net = net
        self.nodes: Dict[str, Node] = {}
        self.dirichlet_alpha = cfg.mcts.dirichlet_alpha
        self.dirichlet_weight = cfg.mcts.dirichlet_weight
        self.puct_c = cfg.mcts.puct_c

    def search(self, state: State, depth: int) -> float:
        if state.terminal():
            return float(state.terminal_reward())

        key = state.record_string()
        if key not in self.nodes:
            p, v = self.net.predict(state)
            self.nodes[key] = Node(p, v)
            return float(v)

        node = self.nodes[key]
        p = node.p.copy()
        if depth == 0:
            p = (1.0 - self.dirichlet_weight) * p + self.dirichlet_weight * np.random.dirichlet(
                [self.dirichlet_alpha] * len(p)
            )

        best_action, best_ucb = None, -float("inf")
        for action in state.legal_actions():
            n = 1 + node.n[action]
            q_sum = node.q_sum_all / node.n_all + node.q_sum[action]
            ucb = q_sum / n + self.puct_c * np.sqrt(node.n_all) * p[action] / n
            if ucb > best_ucb:
                best_action, best_ucb = action, ucb

        state.play(best_action)  # type: ignore[arg-type]
        q_new = -self.search(state, depth + 1)
        node.update(best_action, q_new)  # type: ignore[arg-type]
        return float(q_new)

    def think(self, state: State, num_simulations: int, temperature: float | None = None, show: bool = False) -> np.ndarray:
        if temperature is None:
            temperature = cfg.mcts.temperature_init
        if show:
            print(state)
        start, prev_time = time.time(), 0.0
        for _ in range(num_simulations):
            self.search(copy.deepcopy(state), depth=0)
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root = self.nodes[state.record_string()]
                    pv = self.pv(state)
                    if pv:
                        print(
                            "%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s"
                            % (
                                tmp_time,
                                state.action2str(pv[0]),
                                root.q_sum[pv[0]] / max(root.n[pv[0]], 1),
                                root.n[pv[0]],
                                root.n_all,
                                " ".join([state.action2str(a) for a in pv]),
                            )
                        )

        root_n = self.nodes[state.record_string()].n
        n = root_n + 1
        n = (n / np.max(n)) ** (1.0 / (temperature + 1e-8))
        return n / n.sum()

    def pv(self, state: State):
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
