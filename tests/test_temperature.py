"""温度調整コントローラのテスト。"""

from __future__ import annotations

import math
import random

from app.temperature import TemperatureController
from config import TemperatureScheduleConfig


def test_temperature_controller_progression() -> None:
    config = TemperatureScheduleConfig(
        initial=1.0,
        warmup_moves=2,
        decay_rate=0.5,
        min_value=0.05,
        endgame_move=3,
        endgame_temperature=0.05,
        endgame_slope=1.0,
        noise_scale=0.0,
    )
    controller = TemperatureController(config)
    temps = [controller.step() for _ in range(6)]

    assert math.isclose(temps[0], 1.0)
    assert math.isclose(temps[1], 1.0)
    assert temps[2] < temps[1]
    assert temps[-1] >= config.endgame_temperature
    late_temp = controller.temperature_for_move(30)
    assert math.isclose(late_temp, config.endgame_temperature, rel_tol=1e-3, abs_tol=1e-4)
    assert math.isclose(controller.last_temperature, temps[-1], rel_tol=1e-6)


def test_temperature_controller_reset_noise_changes_bias() -> None:
    config = TemperatureScheduleConfig(
        initial=0.6,
        warmup_moves=0,
        decay_rate=0.9,
        min_value=0.2,
        endgame_move=None,
        endgame_temperature=0.2,
        endgame_slope=1.0,
        noise_scale=0.05,
    )
    controller = TemperatureController(config)

    random.seed(42)
    controller.reset()
    first = controller.step()

    random.seed(42)
    controller.reset()
    second = controller.step()
    assert math.isclose(first, second, rel_tol=1e-6, abs_tol=1e-6)

    random.seed(99)
    controller.reset()
    third = controller.step()
    assert not math.isclose(second, third, rel_tol=1e-3, abs_tol=1e-3)
