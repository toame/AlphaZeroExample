"""棋譜保存モジュールのユニットテスト。"""
from __future__ import annotations

from pathlib import Path

from config import GameConfig
from game import State
from app.game_records import GameRecord, GameRecordSaver, build_game_record, load_game_record


def test_build_game_record_for_connect6() -> None:
    """connect6 ルールの棋譜が正しく文字列化されることを確認する。"""

    config = GameConfig(rule="connect6", board_size=6)
    state = State(config)
    moves = [f"{config.axes.X[i]}{config.axes.Y[i]}" for i in range(5)]
    for move in moves:
        state.play(move)

    record = build_game_record(game_index=3, state=state, game_config=config)

    assert record.game_index == 3
    assert record.rule == "connect6"
    assert record.board_size == 6
    assert record.first_player == config.first_player
    assert record.axis_x == config.axes.X
    assert record.axis_y == config.axes.Y
    assert record.moves == moves


def test_game_record_saver_interval_and_persistence(tmp_path: Path) -> None:
    """保存間隔判定と JSON 保存が行えることを確認する。"""

    saver = GameRecordSaver(base_dir=tmp_path, interval=5)
    assert saver.should_save(1)
    assert not saver.should_save(2)
    assert saver.should_save(6)

    record = GameRecord(
        game_index=1,
        rule="connect6",
        board_size=19,
        first_player=1,
        winner=0,
        axis_x="ABCDEFGHIKLMNOPQRST",
        axis_y="abcdefghijklmnopqrs",
        moves=["Aa", "Bb"],
    )

    saved_path = saver.save(record)
    loaded = load_game_record(saved_path)
    assert loaded == record
    assert saved_path.read_text(encoding="utf-8").strip().startswith("{")
