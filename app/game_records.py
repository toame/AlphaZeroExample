"""自己対戦で得られた棋譜を保存・読み込みする補助モジュール。"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from config import GameConfig
from game import State


@dataclass
class GameRecord:
    """自己対戦 1 局分の棋譜情報。"""

    game_index: int
    rule: str
    board_size: int
    first_player: int
    winner: int
    axis_x: str
    axis_y: str
    moves: List[str]

    def to_dict(self) -> dict:
        """JSON 化しやすい辞書形式に変換する。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GameRecord":
        """辞書から `GameRecord` を復元する。"""

        required_keys = {
            "game_index",
            "rule",
            "board_size",
            "first_player",
            "winner",
            "axis_x",
            "axis_y",
            "moves",
        }
        missing = required_keys - data.keys()
        if missing:
            raise ValueError(f"棋譜データに不足しているキーがあります: {sorted(missing)}")
        moves = list(data["moves"])
        if not isinstance(moves, list):
            raise ValueError("moves はリスト形式で保存してください")
        return cls(
            game_index=int(data["game_index"]),
            rule=str(data["rule"]),
            board_size=int(data["board_size"]),
            first_player=int(data["first_player"]),
            winner=int(data["winner"]),
            axis_x=str(data["axis_x"]),
            axis_y=str(data["axis_y"]),
            moves=[str(m) for m in moves],
        )


def build_game_record(game_index: int, state: State, game_config: GameConfig) -> GameRecord:
    """終局後の `State` から棋譜情報を組み立てる。"""

    moves = [state.action2str(action) for action in state.record]
    return GameRecord(
        game_index=game_index,
        rule=game_config.rule,
        board_size=game_config.board_size,
        first_player=game_config.first_player,
        winner=state.win_color,
        axis_x=game_config.axes.X,
        axis_y=game_config.axes.Y,
        moves=moves,
    )


class GameRecordSaver:
    """定期的に棋譜ファイルを保存するユーティリティ。"""

    def __init__(
        self,
        base_dir: str | Path,
        interval: int,
        *,
        filename_template: str = "game_{index:04d}.json",
    ) -> None:
        if interval <= 0:
            raise ValueError("interval は 1 以上に設定してください")
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._interval = interval
        self._filename_template = filename_template

    def should_save(self, game_index: int) -> bool:
        """与えられた対局番号を保存対象にするか判定する。"""

        return (game_index - 1) % self._interval == 0

    def save(self, record: GameRecord) -> Path:
        """棋譜ファイルを保存し、保存先パスを返す。"""

        path = self._base_dir / self._filename_template.format(index=record.game_index)
        payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=2)
        path.write_text(payload, encoding="utf-8")
        return path

    def list_records(self) -> List[Path]:
        """保存済み棋譜ファイルを更新日時順で取得する。"""

        files = sorted(self._base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files


def load_game_record(path: str | Path) -> GameRecord:
    """保存済みの JSON ファイルから棋譜を読み込む。"""

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return GameRecord.from_dict(data)


def load_records_from_dir(base_dir: str | Path) -> List[GameRecord]:
    """ディレクトリ内の棋譜ファイルをまとめて読み込む。"""

    p = Path(base_dir)
    if not p.exists():
        return []
    records: List[GameRecord] = []
    for file in sorted(p.glob("*.json")):
        try:
            records.append(load_game_record(file))
        except Exception:
            continue
    return records
