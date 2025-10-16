"""保存済みの自己対戦棋譜を GUI で閲覧するビューア。"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import List, Optional

from config import AppConfig, GameConfig
from config.loader import load_default_config
from game import State
from .game_records import GameRecord, load_game_record


class _GameRecordViewer:
    """Tkinter を用いた簡易的な棋譜ビューア。"""

    def __init__(self, records_dir: Path) -> None:
        self._records_dir = records_dir
        self._record_paths: List[Path] = []
        self._current_record: Optional[GameRecord] = None
        self._current_step: int = 0
        self._suspend_scale_callback = False

        self._root = tk.Tk()
        self._root.title("自己対戦棋譜ビューア")

        self._file_var = tk.StringVar(value="棋譜ファイルを選択してください")
        self._status_var = tk.StringVar(value="保存済みの棋譜がここに表示されます")

        main_frame = tk.Frame(self._root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        list_frame = tk.Frame(main_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)

        self._listbox = tk.Listbox(list_frame, width=28)
        self._listbox.pack(side=tk.LEFT, fill=tk.Y)
        self._listbox.bind("<<ListboxSelect>>", self._on_select_record)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.configure(yscrollcommand=scrollbar.set)

        board_frame = tk.Frame(main_frame)
        board_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._board_size_px = 640
        self._canvas = tk.Canvas(board_frame, width=self._board_size_px, height=self._board_size_px, bg="#f8f5e4")
        self._canvas.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(self._root)
        control_frame.pack(fill=tk.X, padx=4, pady=4)

        tk.Label(control_frame, textvariable=self._file_var, anchor="w").pack(fill=tk.X)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=4)
        tk.Button(button_frame, text="<< 最初", command=self._go_first).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="< 戻る", command=self._go_prev).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="進む >", command=self._go_next).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="最後 >>", command=self._go_last).pack(side=tk.LEFT, padx=2)

        self._move_scale = tk.Scale(
            control_frame,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._on_scale_changed,
            label="表示する手数",
        )
        self._move_scale.pack(fill=tk.X)

        tk.Label(control_frame, textvariable=self._status_var, anchor="w").pack(fill=tk.X)

    def run(self) -> None:
        """ビューアを起動して Tk のイベントループを開始する。"""

        self._load_records()
        self._root.mainloop()

    def _load_records(self) -> None:
        """棋譜ディレクトリを走査し、リストに反映する。"""

        if not self._records_dir.exists():
            self._status_var.set(f"棋譜ディレクトリが見つかりません: {self._records_dir}")
            return
        self._record_paths = sorted(self._records_dir.glob("*.json"))
        self._listbox.delete(0, tk.END)
        if not self._record_paths:
            self._status_var.set("保存済みの棋譜がありません")
            return
        for path in self._record_paths:
            self._listbox.insert(tk.END, path.stem)
        self._listbox.selection_set(0)
        self._on_select_record()

    def _on_select_record(self, event=None) -> None:
        """リストで選択された棋譜を読み込み表示する。"""

        if not self._record_paths:
            return
        selection = self._listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        path = self._record_paths[index]
        try:
            self._current_record = load_game_record(path)
        except Exception as exc:  # pragma: no cover - GUI 例外ハンドリング
            self._status_var.set(f"棋譜の読み込みに失敗しました: {exc}")
            return
        self._file_var.set(f"ファイル: {path.name}")
        self._current_step = 0
        self._move_scale.configure(to=len(self._current_record.moves))
        self._set_scale_value(0)
        self._update_board()

    def _update_board(self) -> None:
        """現在の手数に応じて盤面を描画する。"""

        record = self._current_record
        if record is None:
            self._canvas.delete("all")
            self._status_var.set("棋譜ファイルを選択してください")
            return

        config = GameConfig(
            rule=record.rule,
            board_size=record.board_size,
            axes={"X": record.axis_x, "Y": record.axis_y},
            first_player=record.first_player,
        )
        state = State(config)
        if self._current_step > 0:
            for move in record.moves[: self._current_step]:
                state.play(move)

        self._draw_board(state)

        total_moves = len(record.moves)
        winner_map = {1: "先手勝ち", -1: "後手勝ち", 0: "引き分け"}
        winner_text = winner_map.get(record.winner, "不明")
        if self._current_step == 0:
            move_text = "開始局面"
        else:
            move_text = f"{self._current_step} 手目: {record.moves[self._current_step - 1]}"
        self._status_var.set(
            f"{move_text} / 全 {total_moves} 手 | 結果: {winner_text}"
        )

    def _draw_board(self, state: State) -> None:
        """盤面と石を Canvas に描画する。"""

        self._canvas.delete("all")
        size = state.size
        margin = 30
        usable = self._board_size_px - margin * 2
        if size > 1:
            cell = usable / (size - 1)
        else:
            cell = 0

        # 盤の格子線を描画
        for i in range(size):
            pos = margin + cell * i
            self._canvas.create_line(margin, pos, margin + cell * (size - 1), pos, fill="#8b5a2b")
            self._canvas.create_line(pos, margin, pos, margin + cell * (size - 1), fill="#8b5a2b")

        # 座標ラベル
        for i, label in enumerate(state.Y):
            pos = margin + cell * i
            self._canvas.create_text(margin - 12, pos, text=label, fill="#333333")
        for i, label in enumerate(state.X):
            pos = margin + cell * i
            self._canvas.create_text(pos, margin - 12, text=label, fill="#333333")

        radius = cell * 0.4 if size > 1 else 12
        for x in range(size):
            for y in range(size):
                value = int(state.board[x, y])
                if value == 0:
                    continue
                cx = margin + cell * y
                cy = margin + cell * x
                fill = "#000000" if value == 1 else "#ffffff"
                outline = "#000000"
                self._canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill=fill,
                    outline=outline,
                )

    def _on_scale_changed(self, value: str) -> None:
        """スライダーの変更を反映する。"""

        if self._suspend_scale_callback:
            return
        self._current_step = int(float(value))
        self._update_board()

    def _set_scale_value(self, value: int) -> None:
        """スライダー値をプログラム側から更新する。"""

        self._suspend_scale_callback = True
        self._move_scale.set(value)
        self._suspend_scale_callback = False

    def _go_first(self) -> None:
        self._current_step = 0
        self._set_scale_value(0)
        self._update_board()

    def _go_prev(self) -> None:
        if self._current_record is None:
            return
        self._current_step = max(0, self._current_step - 1)
        self._set_scale_value(self._current_step)
        self._update_board()

    def _go_next(self) -> None:
        if self._current_record is None:
            return
        total = len(self._current_record.moves)
        self._current_step = min(total, self._current_step + 1)
        self._set_scale_value(self._current_step)
        self._update_board()

    def _go_last(self) -> None:
        if self._current_record is None:
            return
        self._current_step = len(self._current_record.moves)
        self._set_scale_value(self._current_step)
        self._update_board()


def launch_game_record_viewer(
    config: Optional[AppConfig] = None,
    records_dir: str | Path | None = None,
) -> None:
    """保存済み棋譜を閲覧する Tkinter ベースのビューアを起動する。"""

    cfg = config or load_default_config()
    base_dir = Path(records_dir) if records_dir else Path(cfg.training.artifacts_dir) / cfg.training.game_record_dirname
    viewer = _GameRecordViewer(base_dir)
    viewer.run()


if __name__ == "__main__":  # pragma: no cover - GUI の手動起動用
    launch_game_record_viewer()
