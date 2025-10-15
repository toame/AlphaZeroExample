"""学習周辺機能のテスト。"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest

import numpy as np
import torch

from app.artifacts import CheckpointManager, MetricsRecorder
from app.training_loop import _value_from_perspective
from config import GameConfig, TrainingConfig
from training import Episode, EpisodeSampler, EpisodeStep, Trainer


class DummyNet(torch.nn.Module):
    """最小構成のネットワーク。"""

    def __init__(self) -> None:
        super().__init__()
        self._policy = torch.nn.Linear(2, 2)
        self._value = torch.nn.Linear(2, 1)

    def forward(self, x: torch.Tensor):
        flat = x.view(x.size(0), -1)
        policy = torch.softmax(self._policy(flat), dim=-1)
        value = torch.tanh(self._value(flat))
        return policy, value


class TrainingLoopTest(unittest.TestCase):
    def test_value_from_perspective(self) -> None:
        self.assertEqual(_value_from_perspective(1, 1), 1.0)
        self.assertEqual(_value_from_perspective(-1, 1), -1.0)
        self.assertEqual(_value_from_perspective(1, -1), -1.0)
        self.assertEqual(_value_from_perspective(-1, -1), 1.0)
        self.assertEqual(_value_from_perspective(1, 0), 0.0)

    def test_episode_sampler_iterates_each_step_once(self) -> None:
        sampler = EpisodeSampler()
        steps = [
            EpisodeStep(
                feature=np.full((2, 1, 1), fill_value=float(i), dtype=np.float32),
                policy_target=np.array([1.0, 0.0], dtype=np.float32),
                value_target=float(i % 2),
            )
            for i in range(3)
        ]
        episodes = [Episode(steps=steps[:2], winner=1), Episode(steps=steps[2:], winner=-1)]
        cache = sampler.build_cache(episodes)
        seen = []
        for batch in sampler.iterate_batches(cache, batch_size=2):
            batch_values = batch.x.numpy().reshape(batch.x.size(0), -1)[:, 0]
            seen.extend(batch_values.tolist())
        self.assertCountEqual(seen, [0.0, 1.0, 2.0])

    def test_trainer_scheduler_hook_and_callback(self) -> None:
        training_config = TrainingConfig(batch_size=2, num_epochs=2)
        net = DummyNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=training_config.lr)

        scheduler_calls: dict[str, int] = {}

        def factory(opt: torch.optim.Optimizer, total_steps: int):
            scheduler_calls["steps"] = total_steps

            class _Counter:
                def __init__(self) -> None:
                    self.calls = 0

                def step(self) -> None:
                    self.calls += 1

            counter = _Counter()
            scheduler_calls["instance"] = counter
            return counter

        callback_records: list[float] = []

        def callback(epoch: int, metrics: dict[str, float]) -> None:
            callback_records.append(metrics["policy_loss"])

        trainer = Trainer(
            GameConfig(),
            training_config,
            net,
            optimizer,
            scheduler_factory=factory,
            epoch_callback=callback,
        )

        feature = np.zeros((2, 1, 1), dtype=np.float32)
        policy = np.array([0.5, 0.5], dtype=np.float32)
        episode = Episode(
            steps=[
                EpisodeStep(feature=feature, policy_target=policy, value_target=1.0),
                EpisodeStep(feature=feature, policy_target=policy, value_target=-1.0),
            ],
            winner=1,
        )
        result = trainer.fit([episode])

        total_steps = training_config.num_epochs * 1  # ceil(2 samples / batch_size 2) = 1
        self.assertEqual(scheduler_calls["steps"], total_steps)
        self.assertEqual(scheduler_calls["instance"].calls, total_steps)
        self.assertEqual(len(callback_records), training_config.num_epochs)
        self.assertIsInstance(result.policy_loss, float)

    def test_metrics_and_checkpoint_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = MetricsRecorder(tmp_dir, "metrics.csv", enable_tensorboard=False)
            recorder.log_epoch(0, {"policy_loss": 1.0, "value_loss": 2.0, "lr": 0.1, "batches": 3})
            recorder.close()

            with open(os.path.join(tmp_dir, "metrics.csv"), encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], "1")

            manager = CheckpointManager(tmp_dir, "latest.pt", "best.pt")
            net = DummyNet()
            manager.save(net, 1.0)
            self.assertTrue(os.path.exists(manager.latest_path))
            self.assertTrue(os.path.exists(manager.best_path))
            manager.save(net, 0.5)
            self.assertEqual(manager.best_metric, 0.5)


if __name__ == "__main__":
    unittest.main()
