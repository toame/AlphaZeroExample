"""自己対戦と学習の制御フロー。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.artifacts import CheckpointManager, MetricsRecorder
from config import AppConfig
from game import State
from mcts import Tree
from network import Net
from training import (
    Episode,
    EpisodeStep,
    Trainer,
    create_default_optimizer,
    create_scheduler_factory,
    vs_random,
)
from replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


def _format_result_distribution(result_distribution: Dict[int, int]) -> str:
    """勝敗分布を読みやすい文字列に整形する。"""

    order = [1, 0, -1]
    return ", ".join([f"{key}:{result_distribution.get(key, 0)}" for key in order])


def _value_from_perspective(player_color: int, winner: int) -> float:
    """特定プレイヤー視点での勝敗値を返す。"""

    if winner == 0:
        return 0.0
    return 1.0 if player_color == winner else -1.0


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
    scheduler_factory = create_scheduler_factory(train_cfg)
    metrics_dir = Path(train_cfg.artifacts_dir)
    checkpoint_dir = metrics_dir / "checkpoints"
    recorder = MetricsRecorder(
        str(metrics_dir),
        train_cfg.metrics_filename,
        enable_tensorboard=train_cfg.enable_tensorboard,
    )
    replay_cfg = train_cfg.replay_buffer
    trainer = Trainer(
        game_cfg,
        train_cfg,
        net,
        optimizer,
        scheduler_factory=scheduler_factory,
        epoch_callback=recorder.log_epoch,
    )
    checkpoint_manager = CheckpointManager(
        str(checkpoint_dir),
        train_cfg.latest_checkpoint,
        train_cfg.best_checkpoint,
    )
    replay_buffer = ReplayBuffer(
        capacity=replay_cfg.capacity,
        recent_ratio=replay_cfg.recent_ratio,
    )
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

    best_value = None
    try:
        for game_index in range(num_games):
            steps: List[EpisodeStep] = []
            players: List[int] = []
            state = State(game_cfg)
            tree = Tree(net, mcts_cfg)
            temperature = mcts_cfg.temperature_init

            while not state.terminal():
                feature = state.feature()
                p_target = tree.think(state, num_simulations, temperature)
                action = int(np.random.choice(np.arange(len(p_target)), p=p_target))
                players.append(state.color)
                steps.append(
                    EpisodeStep(
                        feature=feature,
                        policy_target=p_target.astype(np.float32),
                        value_target=0.0,
                    )
                )
                state.play(action)
                temperature *= 0.8

            final_temperature = temperature
            winner = state.win_color
            for step, player in zip(steps, players):
                step.value_target = _value_from_perspective(player, winner)
            replay_buffer.add_episode(Episode(steps=steps, winner=winner))

            result_key = int(_value_from_perspective(game_cfg.first_player, winner))
            result_distribution[result_key] += 1

            logger.info(
                "ゲーム %d/%d 完了: moves=%d winner=%d temperature=%.5f",
                game_index + 1,
                num_games,
                len(steps),
                winner,
                final_temperature,
            )

            should_train = (game_index + 1) % num_train_steps == 0
            is_last_game = game_index + 1 == num_games
            if should_train or is_last_game:
                buffer_stats = replay_buffer.stats()
                logger.info(
                    "学習を実行します: buffer_size=%d total_added=%d result_distribution=%s",
                    buffer_stats.size,
                    buffer_stats.total_added,
                    _format_result_distribution(result_distribution),
                )
                if buffer_stats.size < replay_cfg.warmup_size:
                    if not is_last_game:
                        logger.info(
                            "リプレイバッファのウォームアップが未完了のため学習をスキップします: size=%d warmup=%d",
                            buffer_stats.size,
                            replay_cfg.warmup_size,
                        )
                        continue
                    logger.info(
                        "最終ゲームのためウォームアップ未達でも学習を実行します: size=%d warmup=%d",
                        buffer_stats.size,
                        replay_cfg.warmup_size,
                    )

                sample_size = min(replay_cfg.sample_size, buffer_stats.size)
                sampled_episodes = replay_buffer.sample(sample_size)
                logger.info(
                    "サンプリング完了: sample_size=%d recent_ratio=%.2f",
                    len(sampled_episodes),
                    replay_buffer.recent_ratio,
                )

                result = trainer.fit(sampled_episodes)
                checkpoint_manager.save(net, result.value_loss)
                best_value = checkpoint_manager.best_metric
                logger.info(
                    "学習結果: policy_loss=%.6f value_loss=%.6f",
                    result.policy_loss,
                    result.value_loss,
                )
                logger.info(
                    "モデルを保存しました: latest=%s best=%s",
                    checkpoint_manager.latest_path,
                    checkpoint_manager.best_path,
                )
                logger.info(
                    "ランダム対戦評価: %s",
                    sorted(vs_random(net, game_cfg, train_cfg.vs_random_matches).items()),
                )

    finally:
        recorder.close()

    if best_value is not None:
        logger.info("最良の value_loss=%.6f", best_value)
    logger.info("自己対戦と学習が完了しました")
    return net
