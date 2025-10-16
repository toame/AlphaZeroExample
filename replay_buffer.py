"""自己対戦エピソードを保持・サンプリングするリプレイバッファ。"""

from __future__ import annotations

import dataclasses
import logging
from collections import deque
from typing import Deque, Iterable, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ReplayBufferStats:
    """リプレイバッファの状態を表すメタ情報。"""

    size: int
    capacity: int
    total_added: int


class ReplayBuffer:
    """自己対戦データを蓄積し、学習用にサンプリングするためのバッファ。"""

    def __init__(
        self,
        capacity: int,
        *,
        recent_ratio: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity は 1 以上に設定してください")
        if not 0.0 <= recent_ratio <= 1.0:
            raise ValueError("recent_ratio は 0.0 以上 1.0 以下に設定してください")
        self._buffer: Deque["Episode"] = deque(maxlen=capacity)
        self._recent_ratio = recent_ratio
        self._rng = rng or np.random.default_rng()
        self._total_added = 0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        """バッファの最大保持数。"""

        return int(self._buffer.maxlen or 0)

    @property
    def recent_ratio(self) -> float:
        """サンプリング時に常に確保する最新エピソードの割合。"""

        return self._recent_ratio

    def set_recent_ratio(self, ratio: float) -> None:
        """最新エピソードを優先する割合を変更する。"""

        if not 0.0 <= ratio <= 1.0:
            raise ValueError("recent_ratio は 0.0 以上 1.0 以下に設定してください")
        self._recent_ratio = ratio

    def add_episode(self, episode: "Episode") -> None:
        """エピソードを 1 件追加する。容量超過時は最古を破棄する。"""

        if len(self._buffer) == self.capacity:
            logger.debug("リプレイバッファが満杯のため最古のエピソードを破棄します")
        self._buffer.append(episode)
        self._total_added += 1

    def extend(self, episodes: Sequence["Episode"]) -> None:
        """複数エピソードを追加するヘルパー。"""

        for episode in episodes:
            self.add_episode(episode)

    def clear(self) -> None:
        """保持している全エピソードを破棄する。"""

        self._buffer.clear()

    def stats(self) -> ReplayBufferStats:
        """現在の統計情報を返す。"""

        return ReplayBufferStats(
            size=len(self._buffer),
            capacity=self.capacity,
            total_added=self._total_added,
        )

    def sample(self, max_episodes: int, *, recent_ratio: float | None = None) -> List["Episode"]:
        """保持中のエピソードをサンプリングする。"""

        if not self._buffer:
            raise ValueError("リプレイバッファが空のためサンプリングできません")
        if max_episodes <= 0:
            raise ValueError("max_episodes は 1 以上に設定してください")

        episodes = list(self._buffer)
        available = len(episodes)
        sample_size = min(max_episodes, available)
        ratio = self._recent_ratio if recent_ratio is None else recent_ratio
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("recent_ratio は 0.0 以上 1.0 以下に設定してください")

        recent_count = min(int(round(sample_size * ratio)), available)
        if recent_count > 0:
            recent_samples = episodes[-recent_count:]
        else:
            recent_samples = []

        remaining = sample_size - len(recent_samples)
        if remaining > 0:
            older_candidates = episodes[: available - len(recent_samples)]
            if older_candidates:
                remaining = min(remaining, len(older_candidates))
                indices = self._rng.choice(len(older_candidates), size=remaining, replace=False)
                older_samples = [older_candidates[int(i)] for i in np.asarray(indices)]
            else:
                older_samples = []
        else:
            older_samples = []

        sampled = list(recent_samples) + list(older_samples)
        self._rng.shuffle(sampled)
        return sampled

    def iterate(self) -> Iterable["Episode"]:
        """現在保持しているエピソードを古い順に返す。"""

        return tuple(self._buffer)

