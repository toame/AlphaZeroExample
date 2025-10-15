# main.py
from __future__ import annotations

import logging
from typing import Any

from app import demo_mcts, demo_network_outputs, self_play_and_train
from config.loader import load_default_config


def configure_logging() -> None:
    """ログ出力の基本設定を行う。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def main() -> Any:
    """アプリケーションのエントリーポイント。"""

    configure_logging()
    app_config = load_default_config()
    demo_network_outputs(app_config)
    demo_mcts(app_config)
    return self_play_and_train(app_config)


if __name__ == "__main__":
    # 必要に応じてコメントアウトを外して実行してください。
    main()
