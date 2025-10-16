"""乱数 MCTS エージェントと人間が対戦できる Tkinter GUI。"""

from __future__ import annotations

import random
import tkinter as tk
from typing import Optional

from config import AppConfig, GameConfig
from config.loader import load_default_config
from game import State
from random_mcts import RandomMCTSAgent, RandomSearchReport


class _MCTSMatchGUI:
    """乱数 MCTS エージェントと人間が対戦する簡易 GUI。"""

    def __init__(
        self,
        config: AppConfig,
        *,
        simulations: int,
        human_color: int,
        seed: Optional[int],
    ) -> None:
        self._config = config
        self._simulations = max(1, simulations)
        self._seed = seed
        self._rng = random.Random(seed)

        self._state = State(config.game)
        self._human_color = human_color
        self._agent = self._create_agent()
        self._last_ai_report = None

        self._input_enabled = True

        self._root = tk.Tk()
        self._root.title("MCTS 対人戦デモ")

        self._status_var = tk.StringVar(value="対局設定を選んでください")
        self._record_var = tk.StringVar(value="棋譜: なし")
        self._color_var = tk.IntVar(value=human_color)
        self._simulation_var = tk.IntVar(value=self._simulations)
        self._sim_label_var = tk.StringVar(value=self._format_simulation_label(self._simulations))
        self._eval_var = tk.StringVar(value="AI 推定勝率: -")

        control_frame = tk.Frame(self._root)
        control_frame.pack(fill=tk.X, padx=4, pady=4)

        tk.Label(control_frame, text="あなたの手番:").pack(side=tk.LEFT)
        tk.Radiobutton(
            control_frame,
            text="先手 (黒)",
            variable=self._color_var,
            value=1,
        ).pack(side=tk.LEFT, padx=4)
        tk.Radiobutton(
            control_frame,
            text="後手 (白)",
            variable=self._color_var,
            value=-1,
        ).pack(side=tk.LEFT, padx=4)
        tk.Label(control_frame, text="シミュレーション回数:").pack(side=tk.LEFT, padx=4)
        tk.Spinbox(
            control_frame,
            from_=1,
            to=10000,
            textvariable=self._simulation_var,
            width=6,
        ).pack(side=tk.LEFT)
        tk.Button(control_frame, text="新しい対局を開始", command=self._on_restart).pack(side=tk.RIGHT)

        info_frame = tk.Frame(self._root)
        info_frame.pack(fill=tk.X, padx=4)

        tk.Label(info_frame, textvariable=self._status_var, anchor="w").pack(fill=tk.X)
        tk.Label(info_frame, textvariable=self._record_var, anchor="w").pack(fill=tk.X)
        tk.Label(info_frame, textvariable=self._sim_label_var, anchor="w").pack(fill=tk.X)
        tk.Label(info_frame, textvariable=self._eval_var, anchor="w").pack(fill=tk.X)

        self._board_size_px = 640
        self._margin = 30
        self._canvas = tk.Canvas(
            self._root,
            width=self._board_size_px,
            height=self._board_size_px,
            bg="#f8f5e4",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        self._update_board()
        self._update_status()
        self._maybe_trigger_ai()

    def run(self) -> None:
        """Tk のメインループを開始する。"""

        self._root.mainloop()

    def _create_agent(self) -> RandomMCTSAgent:
        """乱数シードを考慮して MCTS エージェントを生成する。"""

        return RandomMCTSAgent(
            self._config.game,
            seed=self._rng.randrange(1 << 30) if self._seed is not None else None,
            candidate_radius=2,
            initial_radius=3,
            rollout_limit=120,
        )

    def _on_restart(self) -> None:
        """対局をリセットし、設定を反映する。"""

        self._human_color = int(self._color_var.get())
        self._state = State(self._config.game)
        self._agent = self._create_agent()
        self._input_enabled = True
        self._last_ai_report = None
        self._refresh_simulation_setting()
        self._update_board()
        self._update_status()
        self._update_evaluation(None)
        self._maybe_trigger_ai()

    def _on_canvas_click(self, event) -> None:
        """盤面クリック時に人間の着手を処理する。"""

        if not self._input_enabled:
            return
        if self._state.terminal():
            return
        if self._state.color != self._human_color:
            return

        action = self._pixel_to_action(event.x, event.y)
        if action is None:
            return
        if action not in self._state.legal_actions():
            self._status_var.set("その交点には石を置けません")
            return

        self._state.play(action)
        self._update_board()
        self._update_status()
        self._update_evaluation(None)
        if self._state.terminal():
            return
        if self._state.color != self._human_color:
            self._maybe_trigger_ai()

    def _pixel_to_action(self, x: float, y: float) -> Optional[int]:
        """キャンバス座標から最寄りの交点を求める。"""

        size = self._state.size
        if size <= 1:
            return 0
        usable = self._board_size_px - self._margin * 2
        cell = usable / (size - 1)
        grid_x = round((y - self._margin) / cell)
        grid_y = round((x - self._margin) / cell)
        if not (0 <= grid_x < size and 0 <= grid_y < size):
            return None
        return grid_x * size + grid_y

    def _maybe_trigger_ai(self) -> None:
        """AI の手番であれば非同期に着手を進める。"""

        if self._state.terminal():
            self._update_status()
            return
        if self._state.color == self._human_color:
            self._update_status()
            return
        self._refresh_simulation_setting()
        self._input_enabled = False
        self._status_var.set("MCTS が思考中です...")
        self._root.after(150, self._run_ai_turn)

    def _run_ai_turn(self) -> None:
        """MCTS エージェントの手番を処理する。"""

        if self._state.terminal():
            self._input_enabled = True
            self._update_status()
            return
        if self._state.color == self._human_color:
            self._input_enabled = True
            self._update_status()
            return

        action = self._agent.select_action(self._state.copy(), self._simulations)
        self._last_ai_report = self._agent.last_report
        self._state.play(action)
        self._update_board()
        self._update_evaluation(self._last_ai_report)
        if self._state.terminal():
            self._input_enabled = True
            self._update_status()
            return
        if self._state.color != self._human_color:
            self._root.after(100, self._run_ai_turn)
            return
        self._input_enabled = True
        self._update_status()

    def _refresh_simulation_setting(self) -> None:
        """入力されたシミュレーション回数を正規化して保持する。"""

        try:
            value = int(self._simulation_var.get())
        except (tk.TclError, ValueError):
            value = self._simulations
        if value <= 0:
            value = 1
        self._simulation_var.set(value)
        self._simulations = value
        self._sim_label_var.set(self._format_simulation_label(value))

    def _format_simulation_label(self, value: int) -> str:
        """ラベル表示用の文字列を生成する。"""

        return f"MCTS シミュレーション回数: {value}"

    def _update_evaluation(self, report: Optional[RandomSearchReport]) -> None:
        """探索結果に基づいて勝率ラベルを更新する。"""

        if report is None:
            self._eval_var.set("AI 推定勝率: -")
            return
        percent = report.win_rate * 100
        visit_text = (
            f"訪問数 {report.visit_count}/{report.total_visits}"
            if report.total_visits > 0
            else "訪問数 0"
        )
        self._eval_var.set(f"AI 推定勝率: {percent:.1f}% ({visit_text})")

    def _update_board(self) -> None:
        """現在局面の盤面を描画する。"""

        self._canvas.delete("all")
        size = self._state.size
        usable = self._board_size_px - self._margin * 2
        cell = usable / (size - 1) if size > 1 else 0

        for i in range(size):
            pos = self._margin + cell * i
            self._canvas.create_line(
                self._margin,
                pos,
                self._margin + cell * (size - 1),
                pos,
                fill="#8b5a2b",
            )
            self._canvas.create_line(
                pos,
                self._margin,
                pos,
                self._margin + cell * (size - 1),
                fill="#8b5a2b",
            )

        for i, label in enumerate(self._state.Y):
            pos = self._margin + cell * i
            self._canvas.create_text(self._margin - 12, pos, text=label, fill="#333333")
        for i, label in enumerate(self._state.X):
            pos = self._margin + cell * i
            self._canvas.create_text(pos, self._margin - 12, text=label, fill="#333333")

        radius = cell * 0.4 if size > 1 else 12
        for x in range(size):
            for y in range(size):
                value = int(self._state.board[x, y])
                if value == 0:
                    continue
                cx = self._margin + cell * y
                cy = self._margin + cell * x
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

        if self._state.record:
            last = self._state.record[-1]
            lx = last // size
            ly = last % size
            cx = self._margin + cell * ly
            cy = self._margin + cell * lx
            highlight = cell * 0.15 if size > 1 else 6
            self._canvas.create_oval(
                cx - highlight,
                cy - highlight,
                cx + highlight,
                cy + highlight,
                outline="#ff0000",
                width=2,
            )

        record_text = self._state.record_string()
        self._record_var.set(f"棋譜: {record_text if record_text else 'なし'}")

    def _update_status(self) -> None:
        """現在の状況をラベルに表示する。"""

        if self._state.terminal():
            winner_map = {1: "先手の勝利", -1: "後手の勝利", 0: "引き分け"}
            result = winner_map.get(self._state.win_color, "不明")
            self._status_var.set(f"対局終了: {result}")
            return
        if self._state.color == self._human_color:
            self._status_var.set(
                "あなたの手番です。connect6 では初手は 1 石、以降は 2 石を続けて置きます"
            )
        else:
            self._status_var.set("MCTS が思考しています")


def launch_mcts_vs_player_gui(
    config: Optional[AppConfig] = None,
    *,
    simulations: int = 200,
    human_color: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """乱数 MCTS と人間が対局する GUI を起動する。"""

    cfg = _prepare_connect6_config(config or load_default_config())
    color = human_color or cfg.game.first_player
    if color not in (1, -1):
        raise ValueError("human_color には 1 (先手) または -1 (後手) を指定してください")
    gui = _MCTSMatchGUI(cfg, simulations=simulations, human_color=color, seed=seed)
    gui.run()


def _prepare_connect6_config(config: AppConfig) -> AppConfig:
    """GUI 用に connect6 設定へ調整した `AppConfig` を返す。"""

    game_cfg = GameConfig(
        rule="connect6",
        board_size=19,
        symbols=config.game.symbols,
        first_player=config.game.first_player,
    )
    return config.model_copy(update={"game": game_cfg}, deep=True)


if __name__ == "__main__":  # pragma: no cover - GUI の手動起動用
    launch_mcts_vs_player_gui()
