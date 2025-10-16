"""デモ用の補助関数群。"""

from __future__ import annotations

import logging
from typing import Callable

from config import AppConfig

from alphazero import SearchReport, State, Tree, Net, show_net

logger = logging.getLogger(__name__)


def _log_search_report(state: State, report: SearchReport) -> None:
    """探索過程を記録するためのコールバック。"""

    if report.best_action is None:
        return
    pv_text = " ".join([state.action2str(a) for a in report.pv])
    best = state.action2str(report.best_action)
    logger.info(
        "探索進捗: %.2f sec best=%s q=%.4f n=%d/%d pv=%s",
        report.elapsed,
        best,
        report.best_q or 0.0,
        report.best_n or 0,
        report.total_simulations,
        pv_text,
    )


def demo_network_outputs(cfg: AppConfig) -> None:
    """ネットワーク出力のデモを実行する。"""

    game_cfg = cfg.game
    net_cfg = cfg.network

    logger.info("ネットワーク推論デモを開始します")
    logger.info("初期局面")
    show_net(Net(game_cfg, net_cfg), State(game_cfg))

    logger.info("石を連打して勝利するケース")
    show_net(Net(game_cfg, net_cfg), State(game_cfg).play("A1 C1 A2 C2"))

    logger.info("相手のダブルリーチで負けるケース")
    show_net(Net(game_cfg, net_cfg), State(game_cfg).play("B2 A2 A3 C1 B3"))

    logger.info("ダブルリーチで勝利するケース")
    show_net(Net(game_cfg, net_cfg), State(game_cfg).play("B2 A2 A3 C1"))

    logger.info("フォローのダブルで勝利するケース")
    show_net(Net(game_cfg, net_cfg), State(game_cfg).play("B1 A3"))


def _run_tree_demo(cfg: AppConfig, state_builder: Callable[[State], State], simulations: int) -> None:
    """指定した局面で MCTS のデモを実行する。"""

    mcts_cfg = cfg.mcts
    game_cfg = cfg.game
    net_cfg = cfg.network

    tree = Tree(Net(game_cfg, net_cfg), mcts_cfg, progress_callback=_log_search_report)
    state = state_builder(State(game_cfg))
    print(state)
    tree.think(state, simulations)


def demo_mcts(cfg: AppConfig) -> None:
    """MCTS 探索のデモを実行する。"""

    mcts_cfg = cfg.mcts
    demo = mcts_cfg.num_simulations_demo
    demo_mid = mcts_cfg.num_simulations_demo_mid

    logger.info("MCTS デモを開始します")
    _run_tree_demo(cfg, lambda s: s, demo)
    _run_tree_demo(cfg, lambda s: s.play("A1 C1 A2 C2"), demo_mid)
    _run_tree_demo(cfg, lambda s: s.play("B2 A2 A3 C1 B3"), demo_mid)
    _run_tree_demo(cfg, lambda s: s.play("B2 A2 A3 C1"), demo_mid)
