# training.py
from __future__ import annotations
import dataclasses
import logging
import math
import numpy as np
import torch
import torch.optim as optim
from typing import Callable, Dict, List, Sequence
from config import GameConfig, TrainingConfig
from game import State
from network import Net

logger = logging.getLogger(__name__)

Episode = tuple[List[int], int, List[np.ndarray]]

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
    """エピソードからランダムにバッチを生成するユーティリティ。"""

    def __init__(self, game_config: GameConfig, rng: np.random.Generator | None = None) -> None:
        self._game_config = game_config
        self._rng = rng or np.random.default_rng()

    def sample_batch(self, episodes: Sequence[Episode], batch_size: int) -> TrainingBatch:
        if not episodes:
            raise ValueError("学習用エピソードが空のためバッチを生成できません")

        x_list, policy_targets, value_targets = [], [], []
        for _ in range(batch_size):
            ep = episodes[self._rng.integers(len(episodes))]
            turn_idx = int(self._rng.integers(len(ep[0])))
            state = State(self._game_config)
            for action in ep[0][:turn_idx]:
                state.play(action)

            v = ep[1]
            p_target = ep[2][turn_idx]
            # 奇数ターンでは価値の符号を反転させる。
            value = v if turn_idx % 2 == 0 else -v

            x_list.append(state.feature())
            policy_targets.append(p_target.astype(np.float32))
            value_targets.append(np.array([value], dtype=np.float32))

        x = torch.from_numpy(np.array(x_list, dtype=np.float32))
        policy_tensor = torch.from_numpy(np.array(policy_targets, dtype=np.float32))
        value_tensor = torch.from_numpy(np.array(value_targets, dtype=np.float32))
        return TrainingBatch(x=x, policy_target=policy_tensor, value_target=value_tensor)

def create_default_optimizer(net: Net, config: TrainingConfig) -> optim.Optimizer:
    """デフォルトの最適化手法（SGD）を生成する。"""

    return optim.SGD(
        net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )

def decay_learning_rate(optimizer: optim.Optimizer, decay: float) -> None:
    """単純な学習率減衰を適用する。"""

    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay

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
        scheduler: Callable[[optim.Optimizer], None] | None = None,
    ) -> None:
        self._game_config = game_config
        self._training_config = training_config
        self.net = net
        self.optimizer = optimizer
        self._sampler = sampler or EpisodeSampler(game_config)
        self._scheduler = scheduler
        logger.info(
            "Trainer を初期化: batch_size=%d epochs=%d lr=%.6f lr_decay=%.4f",
            training_config.batch_size,
            training_config.num_epochs,
            training_config.lr,
            training_config.lr_decay,
        )

    def fit(self, episodes: Sequence[Episode]) -> TrainingResult:
        """保持しているネットワークを学習させる。"""

        if not episodes:
            raise ValueError("学習用エピソードが空です")

        batch_size = self._training_config.batch_size
        batches_per_epoch = max(math.ceil(len(episodes) / batch_size), 1)
        policy_loss_sum, value_loss_sum = 0.0, 0.0

        self.net.train()
        logger.info(
            "学習を開始: episodes=%d batch_size=%d epochs=%d",
            len(episodes),
            batch_size,
            self._training_config.num_epochs,
        )
        for epoch in range(self._training_config.num_epochs):
            policy_loss_epoch, value_loss_epoch = 0.0, 0.0
            batch_count_epoch = 0
            for _ in range(0, len(episodes), batch_size):
                batch = self._sampler.sample_batch(episodes, batch_size)
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

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                self.optimizer.step()

            lr_before = self.optimizer.param_groups[0]["lr"]
            if self._scheduler is not None:
                self._scheduler(self.optimizer)
            else:
                decay_learning_rate(self.optimizer, self._training_config.lr_decay)
            lr_after = self.optimizer.param_groups[0]["lr"]

            avg_policy = policy_loss_epoch / max(batch_count_epoch, 1)
            avg_value = value_loss_epoch / max(batch_count_epoch, 1)
            logger.info(
                "エポック %d/%d 完了: policy_loss=%.6f value_loss=%.6f lr=%.6f→%.6f batches=%d",
                epoch + 1,
                self._training_config.num_epochs,
                avg_policy,
                avg_value,
                lr_before,
                lr_after,
                batch_count_epoch,
            )

        num_batches = self._training_config.num_epochs * batches_per_epoch
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
