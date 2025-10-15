"""自己対戦と学習の制御フロー。"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from config import AppConfig
from game import State
from mcts import Tree
from network import Net
from training import Episode, Trainer, create_default_optimizer, vs_random

logger = logging.getLogger(__name__)


def _format_result_distribution(result_distribution: Dict[int, int]) -> str:
    """勝敗分布を読みやすい文字列に整形する。"""

    order = [1, 0, -1]
    return ", ".join([f"{key}:{result_distribution.get(key, 0)}" for key in order])


def self_play_and_train(cfg: AppConfig) -> Net:
    """自己対戦と学習をまとめて実行する。"""

    game_cfg = cfg.game
    net_cfg = cfg.network
    train_cfg = cfg.training
    mcts_cfg = cfg.mcts
    num_games = train_cfg.num_games
    num_train_steps = train_cfg.num_train_steps
    num_simulations = mcts_cfg.num_simulations_train

    net = Net(game_cfg, net_cfg)
    optimizer = create_default_optimizer(net, train_cfg)
    trainer = Trainer(game_cfg, train_cfg, net, optimizer)
    episodes: List[Episode] = []
    result_distribution: Dict[int, int] = {1: 0, 0: 0, -1: 0}

    logger.info(
        "自己対戦と学習を開始: total_games=%d, train_interval=%d, simulations=%d",
        num_games,
        num_train_steps,
        num_simulations,
    )
    logger.info(
        "ランダム対戦による初期評価: %s",
        sorted(vs_random(net, game_cfg, train_cfg.vs_random_matches).items()),
    )

    for game_index in range(num_games):
        record: List[int] = []
        p_targets: List[np.ndarray] = []
        state = State(game_cfg)
        tree = Tree(net, mcts_cfg)
        temperature = mcts_cfg.temperature_init

        while not state.terminal():
            p_target = tree.think(state, num_simulations, temperature)
            action = int(np.random.choice(np.arange(len(p_target)), p=p_target))
            state.play(action)
            record.append(action)
            p_targets.append(p_target)
            temperature *= 0.8

        final_temperature = temperature
        reward = state.terminal_reward() * (1 if len(record) % 2 == 0 else -1)
        result_distribution[reward] += 1
        episodes.append((record, reward, p_targets))

        logger.info(
            "ゲーム %d/%d 完了: moves=%d reward=%d temperature=%.5f",
            game_index + 1,
            num_games,
            len(record),
            reward,
            final_temperature,
        )

        should_train = (game_index + 1) % num_train_steps == 0
        is_last_game = game_index + 1 == num_games
        if should_train or is_last_game:
            logger.info(
                "学習を実行します: episodes=%d result_distribution=%s",
                len(episodes),
                _format_result_distribution(result_distribution),
            )
            result = trainer.fit(episodes)
            logger.info(
                "学習結果: policy_loss=%.6f value_loss=%.6f",
                result.policy_loss,
                result.value_loss,
            )
            logger.info(
                "ランダム対戦評価: %s",
                sorted(vs_random(net, game_cfg, train_cfg.vs_random_matches).items()),
            )

    logger.info("自己対戦と学習が完了しました")
    return net
