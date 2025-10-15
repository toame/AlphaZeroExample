# main.py
from __future__ import annotations
import numpy as np
from config import AppConfig
from config.loader import load_default_config
from game import State
from mcts import SearchReport, Tree
from network import Net
from training import (
    Trainer,
    create_default_optimizer,
    show_net,
    vs_random,
)

def demo_network_outputs(cfg: AppConfig) -> None:
    game_cfg = cfg.game
    net_cfg = cfg.network

    print("initial state")
    show_net(Net(game_cfg, net_cfg), State(game_cfg))

    print("WIN by put")
    show_net(Net(game_cfg, net_cfg), State(game_cfg).play("A1 C1 A2 C2"))

    print("LOSE by opponent's double reach")
    show_net(Net(game_cfg, net_cfg), State(game_cfg).play("B2 A2 A3 C1 B3"))

    print("WIN through double reach")
    show_net(Net(game_cfg, net_cfg), State(game_cfg).play("B2 A2 A3 C1"))

    print("strategic WIN by following double")
    show_net(Net(game_cfg, net_cfg), State(game_cfg).play("B1 A3"))

def demo_mcts(cfg: AppConfig) -> None:
    mcts_cfg = cfg.mcts
    game_cfg = cfg.game
    net_cfg = cfg.network
    demo = mcts_cfg.num_simulations_demo
    demo_mid = mcts_cfg.num_simulations_demo_mid

    def reporter(state: State, report: SearchReport) -> None:
        if report.best_action is None:
            return
        pv_text = " ".join([state.action2str(a) for a in report.pv])
        best = state.action2str(report.best_action)
        print(
            "%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s"
            % (
                report.elapsed,
                best,
                report.best_q or 0.0,
                report.best_n or 0,
                report.total_simulations,
                pv_text,
            )
        )

    tree = Tree(Net(game_cfg, net_cfg), mcts_cfg, progress_callback=reporter)
    initial_state = State(game_cfg)
    print(initial_state)
    tree.think(initial_state, demo)

    tree = Tree(Net(game_cfg, net_cfg), mcts_cfg, progress_callback=reporter)
    mid_state = State(game_cfg).play("A1 C1 A2 C2")
    print(mid_state)
    tree.think(mid_state, demo_mid)

    tree = Tree(Net(game_cfg, net_cfg), mcts_cfg, progress_callback=reporter)
    losing_state = State(game_cfg).play("B2 A2 A3 C1 B3")
    print(losing_state)
    tree.think(losing_state, demo_mid)

    tree = Tree(Net(game_cfg, net_cfg), mcts_cfg, progress_callback=reporter)
    winning_state = State(game_cfg).play("B2 A2 A3 C1")
    print(winning_state)
    tree.think(winning_state, demo_mid)

def self_play_and_train(cfg: AppConfig) -> Net:
    game_cfg = cfg.game
    net_cfg = cfg.network
    train_cfg = cfg.training
    mcts_cfg = cfg.mcts
    num_games = train_cfg.num_games
    num_train_steps = train_cfg.num_train_steps
    num_simulations = mcts_cfg.num_simulations_train

    net = Net(game_cfg, net_cfg)
    optimizer = create_default_optimizer(net, train_cfg)
    trainer = Trainer(game_cfg, train_cfg, net, optimizer)
    episodes = []
    result_distribution = {1: 0, 0: 0, -1: 0}

    print(
        "vs_random = ",
        sorted(vs_random(net, game_cfg, train_cfg.vs_random_matches).items()),
    )

    for g in range(num_games):
        record, p_targets = [], []
        state = State(game_cfg)
        tree = Tree(net, mcts_cfg)
        temperature = mcts_cfg.temperature_init
        while not state.terminal():
            p_target = tree.think(state, num_simulations, temperature)
            action = np.random.choice(np.arange(len(p_target)), p=p_target)
            state.play(action)
            record.append(action)
            p_targets.append(p_target)
            temperature *= 0.8
        reward = state.terminal_reward() * (1 if len(record) % 2 == 0 else -1)
        result_distribution[reward] += 1
        episodes.append((record, reward, p_targets))
        if g % num_train_steps == 0:
            print("game ", end="")
        print(g, " ", end="")
        if (g + 1) % num_train_steps == 0:
            print("generated = ", sorted(result_distribution.items()))
            result = trainer.fit(episodes)
            print(
                f"train loss policy={result.policy_loss:.6f} value={result.value_loss:.6f}"
            )
            print(
                "vs_random = ",
                sorted(vs_random(net, game_cfg, train_cfg.vs_random_matches).items()),
            )

    print("finished")
    return net

if __name__ == "__main__":
    # 必要に応じてコメントアウトを外して実行してください。
    app_config = load_default_config()
    demo_network_outputs(app_config)
    # demo_mcts(app_config)
    # trained = self_play_and_train(app_config)
