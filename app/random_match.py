"""乱数ベース MCTS とランダムプレイヤーの対戦デモ。"""

from __future__ import annotations

import logging
import random
from typing import Dict, Optional

from config import AppConfig
from game import State
from random_mcts import RandomMCTSAgent

logger = logging.getLogger(__name__)


def play_random_mcts_vs_random(
    cfg: AppConfig,
    *,
    simulations: int = 100,
    games: int = 1,
    seed: Optional[int] = None,
    mcts_color: Optional[int] = None,
) -> Dict[str, int]:
    """乱数 MCTS エージェントとランダムプレイヤーの対戦を実行する。"""

    game_cfg = cfg.game
    controller_color = mcts_color if mcts_color is not None else game_cfg.first_player
    rng = random.Random(seed)
    results = {"mcts_win": 0, "random_win": 0, "draw": 0}

    logger.info(
        "乱数 MCTS エージェントとランダムプレイヤーの対戦を %d 局開始します (シミュレーション回数=%d)",
        games,
        simulations,
    )

    agent = RandomMCTSAgent(game_cfg, seed=rng.randrange(1 << 30))

    for game_index in range(games):
        state = State(game_cfg)
        logger.debug("ゲーム %d を開始します", game_index + 1)

        while not state.terminal():
            if state.color == controller_color:
                _play_mcts_turn(state, agent, simulations)
            else:
                _play_random_turn(state, rng)

        if state.win_color == controller_color:
            results["mcts_win"] += 1
            logger.debug("ゲーム %d: MCTS エージェントの勝利", game_index + 1)
        elif state.win_color == -controller_color:
            results["random_win"] += 1
            logger.debug("ゲーム %d: ランダムプレイヤーの勝利", game_index + 1)
        else:
            results["draw"] += 1
            logger.debug("ゲーム %d: 引き分け", game_index + 1)

    logger.info(
        "対戦結果: MCTS %d 勝 / ランダム %d 勝 / 引き分け %d",
        results["mcts_win"],
        results["random_win"],
        results["draw"],
    )
    return results


def _play_mcts_turn(state: State, agent: RandomMCTSAgent, simulations: int) -> None:
    """MCTS エージェントに手番を任せる。"""

    color = state.color
    while not state.terminal() and state.color == color:
        action = agent.select_action(state, simulations)
        state.play(action)


def _play_random_turn(state: State, rng: random.Random) -> None:
    """完全ランダムに手を選び、必要数だけ着手する。"""

    color = state.color
    while not state.terminal() and state.color == color:
        legal = state.legal_actions()
        if not legal:
            break
        state.play(rng.choice(legal))
