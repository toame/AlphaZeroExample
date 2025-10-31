"""自己対戦向け推論のバッチ化効果を計測する簡易スクリプト。"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List

import numpy as np

from config.loader import load_default_config
from game import State
from network import Net


def _generate_random_features(num_samples: int, seed: int) -> np.ndarray:
    """ランダムな手順で得た状態特徴量を生成する。"""

    cfg = load_default_config()
    game_cfg = cfg.game
    rng = np.random.default_rng(seed)
    features: List[np.ndarray] = []

    while len(features) < num_samples:
        state = State(game_cfg)
        while not state.terminal() and len(features) < num_samples:
            features.append(state.feature())
            legal = state.legal_actions()
            action = int(rng.choice(legal))
            state.play(action)
    return np.stack(features, axis=0)


def _benchmark_single(net: Net, features: np.ndarray, repeats: int) -> float:
    """既存のシングル推論の平均秒数を返す。"""

    elapsed_list: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for feature in features:
            net.predict_from_feature(feature)
        elapsed_list.append(time.perf_counter() - start)
    return float(np.mean(elapsed_list))


def _benchmark_batch(net: Net, features: np.ndarray, batch_size: int, repeats: int) -> float:
    """バッチ推論の平均秒数を返す。"""

    elapsed_list: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for start_idx in range(0, len(features), batch_size):
            batch = features[start_idx : start_idx + batch_size]
            net.predict_batch(batch)
        elapsed_list.append(time.perf_counter() - start)
    return float(np.mean(elapsed_list))


def run_benchmark(samples: int, repeats: int, batch_sizes: Iterable[int]) -> None:
    """指定パラメータでベンチマークを実行する。"""

    cfg = load_default_config()
    net = Net(cfg.game, cfg.network)
    features = _generate_random_features(samples, seed=0)

    # ウォームアップして初期化コストを除外する。
    net.predict_from_feature(features[0])
    print(f"サンプル数: {samples}, リピート: {repeats}")

    single_time = _benchmark_single(net, features, repeats)
    single_throughput = samples / single_time
    print(f"シングル推論: {single_time:.6f} 秒, {single_throughput:.1f} サンプル/秒")

    for batch_size in batch_sizes:
        if batch_size <= 1:
            continue
        batch_time = _benchmark_batch(net, features, batch_size, repeats)
        throughput = samples / batch_time
        speedup = single_time / batch_time
        print(
            f"バッチサイズ {batch_size:>3}: {batch_time:.6f} 秒, {throughput:.1f} サンプル/秒, "
            f"速度向上倍率 {speedup:.2f}"
        )


def main() -> None:
    """コマンドライン引数を解析してベンチマークを実行する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048, help="計測に使用するサンプル数")
    parser.add_argument("--repeats", type=int, default=3, help="平均化のための反復回数")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[2, 4, 8, 16, 32, 64],
        help="計測するバッチサイズ一覧 (1 は自動的にスキップ)",
    )
    args = parser.parse_args()

    run_benchmark(args.samples, args.repeats, args.batch_sizes)


if __name__ == "__main__":
    main()
