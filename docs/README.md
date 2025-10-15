# docs

このディレクトリには、プロジェクトの仕様や運用に関するドキュメントを配置します。

## リファクタリング概要

* 設定値は `config.loader.load_default_config()` から `AppConfig` を取得し、`State` や `Net`、`Tree`、`Trainer` へ明示的に渡す構成になりました。
* 学習処理は `training.Trainer` クラスへ集約され、オプティマイザやスケジューラを外部から差し替えられます。データ生成には `EpisodeSampler` を利用してください。
* MCTS の探索進捗は `SearchReport` を受け取るコールバックで通知され、標準出力への依存を排除しています。
