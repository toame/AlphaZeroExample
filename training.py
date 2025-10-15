# training.py
from __future__ import annotations
import dataclasses
import logging
import math
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from config import GameConfig, TrainingConfig
from game import State
from network import Net

logger = logging.getLogger(__name__)

SchedulerFactory = Callable[[optim.Optimizer, int], lr_scheduler.LRScheduler | None]

@dataclasses.dataclass
class EpisodeStep:
    """自己対戦で得られた 1 手分のデータ。"""

    feature: np.ndarray
    policy_target: np.ndarray
    value_target: float


@dataclasses.dataclass
class Episode:
    """自己対戦 1 エピソード分のデータ。"""

    steps: List[EpisodeStep]
    winner: int


@dataclasses.dataclass
class EpisodeBatchCache:
    """バッチ化しやすいように整形した自己対戦データ。"""

    features: np.ndarray
    policies: np.ndarray
    values: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

@dataclasses.dataclass
class TrainingBatch:
    """学習用に整形した特徴量とターゲット。"""

    x: torch.Tensor
    policy_target: torch.Tensor
    value_target: torch.Tensor

@dataclasses.dataclass
class TrainingResult:
    """学習 1 回分の指標。"""

    policy_loss: float
    value_loss: float

class EpisodeSampler:
    """エピソードをバッチ学習向けに整形・シャッフルする。"""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def build_cache(self, episodes: Sequence[Episode]) -> EpisodeBatchCache:
        if not episodes:
            raise ValueError("学習用エピソードが空のためキャッシュを生成できません")

        features, policies, values = [], [], []
        for episode in episodes:
            for step in episode.steps:
                features.append(step.feature.astype(np.float32, copy=False))
                policies.append(step.policy_target.astype(np.float32, copy=False))
                values.append(np.array([step.value_target], dtype=np.float32))

        if not features:
            raise ValueError("エピソードに手番データが含まれていません")

        feature_array = np.stack(features, axis=0)
        policy_array = np.stack(policies, axis=0)
        value_array = np.stack(values, axis=0)
        return EpisodeBatchCache(
            features=feature_array,
            policies=policy_array,
            values=value_array,
        )

    def iterate_batches(self, cache: EpisodeBatchCache, batch_size: int):
        indices = np.arange(cache.num_samples)
        self._rng.shuffle(indices)
        for start in range(0, cache.num_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            x = torch.from_numpy(cache.features[batch_indices])
            policy_tensor = torch.from_numpy(cache.policies[batch_indices])
            value_tensor = torch.from_numpy(cache.values[batch_indices])
            yield TrainingBatch(x=x, policy_target=policy_tensor, value_target=value_tensor)

def create_default_optimizer(net: Net, config: TrainingConfig) -> optim.Optimizer:
    """デフォルトの最適化手法（SGD）を生成する。"""

    return optim.SGD(
        net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )


def create_scheduler_factory(config: TrainingConfig) -> SchedulerFactory:
    """学習率スケジューラを生成するためのファクトリ関数を返す。"""

    scheduler_config = config.scheduler

    def factory(optimizer: optim.Optimizer, total_steps: int) -> lr_scheduler.LRScheduler | None:
        if total_steps <= 0:
            raise ValueError("スケジューラを初期化するためのステップ数が 0 以下です")

        if scheduler_config.type == "none":
            return None
        if scheduler_config.type == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=scheduler_config.cosine_min_lr,
            )
        if scheduler_config.type == "onecycle":
            return lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.lr,
                total_steps=total_steps,
                pct_start=scheduler_config.onecycle_pct_start,
                div_factor=scheduler_config.onecycle_div_factor,
                final_div_factor=scheduler_config.onecycle_final_div_factor,
            )
        raise ValueError(f"未対応のスケジューラ種別です: {scheduler_config.type}")

    return factory

class Trainer:
    """ネットワーク学習ループを管理するクラス。"""

    def __init__(
        self,
        game_config: GameConfig,
        training_config: TrainingConfig,
        net: Net,
        optimizer: optim.Optimizer,
        *,
        sampler: EpisodeSampler | None = None,
        scheduler_factory: Callable[[optim.Optimizer, int], lr_scheduler.LRScheduler | None] | None = None,
        epoch_callback: Callable[[int, Dict[str, float]], None] | None = None,
    ) -> None:
        self._training_config = training_config
        self.net = net
        self.optimizer = optimizer
        self._sampler = sampler or EpisodeSampler()
        self._scheduler_factory = scheduler_factory
        self._epoch_callback = epoch_callback
        logger.info(
            "Trainer を初期化: batch_size=%d epochs=%d lr=%.6f",
            training_config.batch_size,
            training_config.num_epochs,
            training_config.lr,
        )

    def _reset_optimizer(self) -> None:
        """学習率と内部状態を初期化して再学習時の停滞を防ぐ。"""

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self._training_config.lr
        # 以前のモーメンタム等を破棄し、学習再開時に影響を残さない。
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    value.zero_()

    def fit(self, episodes: Sequence[Episode]) -> TrainingResult:
        """保持しているネットワークを学習させる。"""

        if not episodes:
            raise ValueError("学習用エピソードが空です")

        cache = self._sampler.build_cache(episodes)
        batch_size = self._training_config.batch_size
        batches_per_epoch = max(math.ceil(cache.num_samples / batch_size), 1)
        policy_loss_sum, value_loss_sum = 0.0, 0.0

        self._reset_optimizer()
        self.net.train()
        logger.info(
            "学習を開始: episodes=%d batch_size=%d epochs=%d",
            len(episodes),
            batch_size,
            self._training_config.num_epochs,
        )
        total_steps = self._training_config.num_epochs * batches_per_epoch
        scheduler = None
        if self._scheduler_factory is not None:
            scheduler = self._scheduler_factory(self.optimizer, total_steps)
        processed_batches = 0
        for epoch in range(self._training_config.num_epochs):
            policy_loss_epoch, value_loss_epoch = 0.0, 0.0
            batch_count_epoch = 0
            for batch in self._sampler.iterate_batches(cache, batch_size):
                policy_pred, value_pred = self.net(batch.x)

                policy_loss = torch.sum(-batch.policy_target * torch.log(policy_pred + 1e-12))
                value_loss = torch.sum((batch.value_target - value_pred) ** 2)

                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())
                policy_loss_sum += policy_loss_value
                value_loss_sum += value_loss_value
                policy_loss_epoch += policy_loss_value
                value_loss_epoch += value_loss_value
                batch_count_epoch += 1
                processed_batches += 1

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            avg_policy = policy_loss_epoch / max(batch_count_epoch, 1)
            avg_value = value_loss_epoch / max(batch_count_epoch, 1)
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "エポック %d/%d 完了: policy_loss=%.6f value_loss=%.6f lr=%.6f batches=%d",
                epoch + 1,
                self._training_config.num_epochs,
                avg_policy,
                avg_value,
                current_lr,
                batch_count_epoch,
            )
            if self._epoch_callback is not None:
                self._epoch_callback(
                    epoch,
                    {
                        "policy_loss": avg_policy,
                        "value_loss": avg_value,
                        "lr": current_lr,
                        "batches": float(batch_count_epoch),
                    },
                )

        num_batches = max(processed_batches, 1)
        return TrainingResult(
            policy_loss=policy_loss_sum / num_batches,
            value_loss=value_loss_sum / num_batches,
        )

def vs_random(net: Net, game_config: GameConfig, matches: int) -> Dict[int, int]:
    """ランダムプレイヤーと対戦して戦績を計測する。"""

    results: Dict[int, int] = {}
    for i in range(matches):
        first_turn = i % 2 == 0
        turn = first_turn
        state = State(game_config)
        while not state.terminal():
            if turn:
                p, _ = net.predict(state)
                legal = state.legal_actions()
                action = sorted([(a, p[a]) for a in legal], key=lambda x: -x[1])[0][0]
            else:
                action = int(np.random.choice(state.legal_actions()))
            state.play(action)
            turn = not turn
        r = state.terminal_reward() if turn else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
    return results

def show_net(net: Net, state: State) -> None:
    """ネットワークの推論結果を表示する補助関数。"""

    print(state)
    p, v = net.predict(state)
    n = state.size
    print("p = ")
    print((p * 1000).astype(int).reshape((n, n)))
    print("v = ", v)
    print()
