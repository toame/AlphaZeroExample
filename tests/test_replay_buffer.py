"""リプレイバッファの基本挙動を検証するテスト。"""

from __future__ import annotations

import numpy as np

from replay_buffer import ReplayBuffer
from training import Episode, EpisodeStep


def _create_episode(value: int) -> Episode:
    """単純なエピソードを生成するヘルパー。"""

    feature = np.zeros((1,), dtype=np.float32)
    policy = np.zeros((1,), dtype=np.float32)
    step = EpisodeStep(feature=feature, policy_target=policy, value_target=float(value))
    return Episode(steps=[step], winner=value)


def test_replay_buffer_keeps_capacity() -> None:
    """容量を超えた際に最古のエピソードが破棄されることを確認する。"""

    buffer = ReplayBuffer(capacity=3, rng=np.random.default_rng(0))
    for i in range(5):
        buffer.add_episode(_create_episode(i))

    assert len(buffer) == 3
    assert buffer.stats().total_added == 5
    winners = [episode.winner for episode in buffer.iterate()]
    assert winners == [2, 3, 4]


def test_replay_buffer_sample_respects_recent_ratio() -> None:
    """recent_ratio に基づき最新エピソードがサンプルへ含まれることを確認する。"""

    buffer = ReplayBuffer(capacity=10, recent_ratio=0.5, rng=np.random.default_rng(1))
    for i in range(6):
        buffer.add_episode(_create_episode(i))

    sampled = buffer.sample(4)
    winners = {episode.winner for episode in sampled}
    # recent_ratio=0.5 かつ sample_size=4 のため、末尾 2 件が必ず含まれる想定。
    assert {4, 5}.issubset(winners)
