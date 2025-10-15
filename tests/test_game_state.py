"""`game.State` のルール別挙動を検証するテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from config import GameConfig
from config.models import GameAxes
from game import BLACK, WHITE, State


def _make_connect6_config(board_size: int = 19) -> GameConfig:
    """指定サイズの connect6 用ゲーム設定を作成する。"""
    letters_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letters_lower = letters_upper.lower()
    axes = GameAxes(X=letters_upper[:board_size], Y=letters_lower[:board_size])
    return GameConfig(board_size=board_size, rule="connect6", axes=axes)


def test_connect6_turn_flow_on_19_board() -> None:
    """19x19 盤で connect6 の手番処理が正しく進むかを確認する。"""
    config = _make_connect6_config(19)
    state = State(config)

    assert state.board.shape == (19, 19)
    assert state.color == BLACK
    assert len(state.legal_actions()) == 19 * 19
    assert state.feature().shape == (2, 19, 19)

    state.play(state.str2action("Aa"))
    assert state.board[0, 0] == BLACK
    assert state.turn_index == 1
    assert state.color == WHITE
    assert state._stones_remaining == 2

    state.play(state.str2action("Ba"))
    assert state.color == WHITE
    assert state._stones_remaining == 1

    state.play(state.str2action("Bb"))
    assert state.turn_index == 2
    assert state.color == BLACK
    assert state._stones_remaining == 2
    assert len(state.record) == 3
    assert len(state.legal_actions()) == 19 * 19 - 3


def test_connect6_win_detection_on_19_board() -> None:
    """19x19 盤で 6 連を作ったときに勝敗が確定することを確認する。"""
    config = _make_connect6_config(19)
    state = State(config)

    def place(sequence: list[str]) -> None:
        """指定した座標列を順番に打つヘルパー。"""
        for coord in sequence:
            state.play(state.str2action(coord))

    state.play(state.str2action("Aa"))
    place(["Ss", "Sr"])  # 白の初手
    place(["Ab", "Ac"])  # 黒の 2 手目
    place(["Rq", "Rr"])  # 白
    place(["Ad", "Ae"])  # 黒
    place(["Qq", "Qr"])  # 白

    state.play(state.str2action("Af"))  # 黒が 6 連を達成

    assert state.win_color == BLACK
    assert state.terminal()
    assert state.terminal_reward() == -1
    assert state.color == WHITE
    np.testing.assert_array_equal(state.board[0, :6], np.full(6, BLACK, dtype=np.int8))

    with pytest.raises(ValueError):
        state.play(state.str2action("Ag"))
