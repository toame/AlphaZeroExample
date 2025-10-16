# docs

このディレクトリには、プロジェクトの仕様や運用に関するドキュメントを配置します。

## リファクタリング概要

* 設定値は `config.loader.load_default_config()` から `AppConfig` を取得し、`State` や `Net`、`Tree`、`Trainer` へ明示的に渡す構成になりました。
* 学習処理は `training.Trainer` クラスへ集約され、オプティマイザや学習率スケジューラを外部から差し替えられます。データ生成には盤面特徴量をキャッシュする `Episode`／`EpisodeSampler` を利用してください。
* MCTS の探索進捗は `SearchReport` を受け取るコールバックで通知され、標準出力への依存を排除しています。
* MCTS の探索処理は再帰呼び出しを廃止し、イテレーティブなバックアップで Python のコールスタック負荷を削減しました。
* ルートの訪問数に基づく方策計算から余分な +1 バイアスを取り除き、探索結果をそのまま反映するようになりました。
* エントリーポイントの責務を `app.demo` と `app.training_loop` に切り出し、`main.py` は設定読込と関数呼び出しのみを担う構成にしました。
* 学習ログは `logging` モジュールに加え、CSV/TensorBoard へも書き出され、学習率や損失曲線を後から分析できます。
* 学習済みモデルは最新・ベストの 2 系統で自動保存され、長時間学習の途中停止に備えています。
* `Trainer.fit` は学習開始時にオプティマイザの学習率と内部状態を初期化し、繰り返し学習でも学習率が極端に低下しないようになりました。

## 自己対戦時の温度制御

* 温度調整ロジックは `app.temperature.TemperatureController` に切り出され、ウォームアップ期間・指数減衰・終盤の滑らかな遷移・ゲームごとのノイズ付与をまとめて扱います。
* `config/config.yaml` の `mcts.temperature` セクションで以下のパラメータを調整できます。
  * `initial`: ゲーム開始直後の温度。
  * `warmup_moves`: 初期温度を維持する手数。
  * `decay_rate`: ウォームアップ後に掛ける減衰率。
  * `min_value`: 温度の下限。
  * `endgame_move`: 終盤への遷移を開始する手数（`null` で無効化）。
  * `endgame_temperature`: 終盤で目標とする温度。
  * `endgame_slope`: 終盤遷移の滑らかさ（大きいほど緩やか）。
  * `noise_scale`: ゲームごとに付与するガウスノイズの標準偏差。
* 自己対戦ループからは `TemperatureController.step()` を呼び出すだけで適切な温度が得られ、探索の多様性と収束性を両立できます。

## MCTS の探索強化

* 3 手目以降は既存の石に近い交点へ確率を寄せるフィルタを導入し、初期自己対戦でも自然な布石が現れやすくなりました。
* 即勝できる合法手を検出した場合は探索結果を上書きし、必ず勝ち筋を選択するようにしました。

## 乱数モンテカルロ木探索のデモ

* `random_mcts.RandomMCTSAgent` はニューラルネットワークを用いず、乱数ロールアウトで評価する軽量な探索エージェントです。
* `app.random_match.play_random_mcts_vs_random()` を呼び出すと、乱数 MCTS エージェントと完全ランダムプレイヤーとの対戦結果を取得できます。
* 使い方の一例: `python -c "from config.loader import load_default_config; from app.random_match import play_random_mcts_vs_random; cfg = load_default_config(); print(play_random_mcts_vs_random(cfg, simulations=50, games=10))"`
* UCT の評価値を手番視点で正規化し、高シミュレーション時でも探索が安定するよう調整しました。`simulations=2000`, `games=30` の条件で乱数プレイヤーに対して 9 割以上の勝率を確認しています。
* `candidate_radius` と `initial_radius` を指定すると、既存の石の近傍や盤面中央に候補手を絞り込めます。19×19 の connect6 でも無駄な探索を抑えつつ、`rollout_limit` でロールアウトの最大手数を制御できます。
* `RandomMCTSAgent.last_report` から直近探索の勝率（0〜100%）や訪問統計を取得でき、GUI 表示やログ出力に活用できます。
* 石が増えた局面では候補手抽出を省略して全合法手からロールアウトを行い、connect6 の終盤でも探索が 1 手あたり数秒以内で完了するよう最適化を加えました。
* 即勝ち検出は候補手に限定し、合法手が大量に残る序盤は多手順の総当たり探索を抑制することで、GUI でも 200 回程度の探索が数秒で完了します。
* 終盤では合法手を総当たりして必勝手や相手の即勝ちを検出し、乱数ロールアウト前に勝敗が確定する局面を素早く判断します。

## MCTS 対人戦 GUI モード

* `app.mcts_player_gui.launch_mcts_vs_player_gui()` を実行すると、Tkinter ベースの簡易 GUI が立ち上がり、人間プレイヤーと乱数 MCTS が対戦できます。
* GUI 上で先手・後手や MCTS のシミュレーション回数を設定して新しい対局を開始でき、盤面をクリックして着手します。思考中の勝率（訪問統計）もラベルで確認できます。
* connect6 ルール（19×19）を前提にしており、初手は 1 石、それ以降は 2 石を連続で配置する挙動に対応しました。既存の石から距離 2 以内や盤面中央付近に候補手を絞ることで、ニューラルネットワークなしでも 19×19 の探索を現実的な時間で行えます。
* コマンドラインからは `python -m app.mcts_player_gui` を実行してください。初期設定では設定ファイルの `game.first_player` が人間の担当色になります。

## ゲーム設定の拡張

* `config/config.yaml` の `game.rule` で「三目並べ (tic_tac_toe)」と「六目並べ (connect6)」を切り替えられるようになりました。
* connect6 ルールでは初手のみ 1 石、以降は 2 石ずつ配置する挙動を `game.State` が自動で扱います。勝利条件は縦横斜めいずれかの 6 連です。
* ルールを切り替えると推奨盤面サイズと座標軸が自動補完されます。connect6 ではデフォルトで 19×19、座標軸は `Aa` 形式になります。
* `tests/test_game_state.py` で 19x19 の盤面を用いた connect6 のターン処理と勝利判定を検証しています。

## 棋譜保存とビューア

* 自己対戦ループは `training.save_game_records` が `true` のとき、`training.game_record_interval` の間隔（デフォルトで 5 局ごと、1 局目を含む）で棋譜を JSON 形式で保存します。
* 棋譜は `training.artifacts_dir` 配下の `training.game_record_dirname` ディレクトリ（デフォルト `artifacts/game_records`）に `game_0001.json` のようなファイル名で出力されます。
* `app.record_viewer.launch_game_record_viewer()` を呼び出すか、`python -m app.record_viewer` を実行すると Tkinter ベースのビューアが起動し、保存済みの棋譜を一覧から選択して盤面を確認できます。スライダーで途中の手数に移動し、最新手の座標を確認できます。

## ネットワーク構成の切り替え
`config/config.yaml` の `network.architecture` で `"basic"`（従来）と `"katago"`（KataGo 風拡張）を切り替えられます。KataGo 風構成では以下の特徴を持ちます。

* 3×3 畳み込み 2 層 + Squeeze-and-Excitation を備えた深い残差スタック。
* ポリシーヘッドでグローバル平均プーリングから得た特徴をローカルロジットへ加算。
* バリューヘッドはグローバル平均プーリング後の MLP（デフォルト 256→64→1）で構成し、過学習を抑制。

`network.katago` セクションでフィルタ数やヘッドの隠れユニットを調整できます。VRAM 負荷が高まるため、自己対戦の並列度と合わせて調整してください。
