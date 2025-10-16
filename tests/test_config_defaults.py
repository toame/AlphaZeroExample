"""設定モデルのデフォルト動作を確認するテスト。"""

from __future__ import annotations

from config import GameConfig


def test_game_config_auto_defaults_for_tic_tac_toe() -> None:
    """三目並べでは盤面サイズと座標軸が従来通りになることを確認する。"""

    cfg = GameConfig()
    assert cfg.board_size == 3
    assert cfg.axes.X == "ABC"
    assert cfg.axes.Y == "123"


def test_game_config_auto_defaults_for_connect6() -> None:
    """connect6 で自動補完された盤面サイズと座標軸を確認する。"""

    cfg = GameConfig(rule="connect6")
    assert cfg.board_size == 19
    assert cfg.axes.X == "ABCDEFGHIJKLMNOPQRS"
    assert cfg.axes.Y == "abcdefghijklmnopqrs"
