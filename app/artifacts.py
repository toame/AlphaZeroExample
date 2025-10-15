"""学習成果物の管理ユーティリティ。"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
try:  # pragma: no cover - tensorboard はオプション依存
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class MetricsRecorder:
    """学習指標を CSV および TensorBoard に書き出す。"""

    def __init__(
        self,
        base_dir: str,
        filename: str,
        *,
        enable_tensorboard: bool,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._base_dir / filename
        existed = self._csv_path.exists()
        self._csv_file = self._csv_path.open("a", newline="", encoding="utf-8")
        fieldnames = ["epoch", "policy_loss", "value_loss", "lr", "batches"]
        self._writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        if not existed:
            self._writer.writeheader()
        self._tensorboard: Optional[SummaryWriter] = None
        if enable_tensorboard and SummaryWriter is None:
            logger.warning("tensorboard がインストールされていないため、TensorBoard 出力を無効化します")
        if enable_tensorboard and SummaryWriter is not None:
            tb_dir = self._base_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tensorboard = SummaryWriter(str(tb_dir))
            logger.info("TensorBoard ログを %s に出力します", tb_dir)

    def log_epoch(self, epoch_index: int, metrics: Dict[str, float]) -> None:
        """エポック単位の指標を保存する。"""

        row = {
            "epoch": epoch_index + 1,
            "policy_loss": metrics.get("policy_loss", float("nan")),
            "value_loss": metrics.get("value_loss", float("nan")),
            "lr": metrics.get("lr", float("nan")),
            "batches": metrics.get("batches", float("nan")),
        }
        self._writer.writerow(row)
        self._csv_file.flush()
        if self._tensorboard is not None:
            self._tensorboard.add_scalar("loss/policy", row["policy_loss"], epoch_index)
            self._tensorboard.add_scalar("loss/value", row["value_loss"], epoch_index)
            self._tensorboard.add_scalar("lr", row["lr"], epoch_index)

    def close(self) -> None:
        """関連リソースを明示的に解放する。"""

        if self._tensorboard is not None:
            self._tensorboard.flush()
            self._tensorboard.close()
        self._csv_file.close()

    @property
    def csv_path(self) -> Path:
        return self._csv_path


class CheckpointManager:
    """モデル重みの保存を管理する。"""

    def __init__(self, base_dir: str, latest_name: str, best_name: str) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._latest_path = self._base_dir / latest_name
        self._best_path = self._base_dir / best_name
        self._best_metric: float | None = None

    def save(self, net: torch.nn.Module, metric: float) -> None:
        """最新モデルとベストモデルを保存する。"""

        torch.save(net.state_dict(), self._latest_path)
        if self._best_metric is None or metric <= self._best_metric:
            torch.save(net.state_dict(), self._best_path)
            self._best_metric = metric

    @property
    def latest_path(self) -> Path:
        return self._latest_path

    @property
    def best_path(self) -> Path:
        return self._best_path

    @property
    def best_metric(self) -> float | None:
        return self._best_metric
