# main.py
from __future__ import annotations
import numpy as np
from game import State
from network import Net
from mcts import Tree
from training import show_net, train, vs_random
from config.loader import cfg

def demo_network_outputs() -> None:
    print("initial state")
    show_net(Net(), State())

    print("WIN by put")
    show_net(Net(), State().play("A1 C1 A2 C2"))

    print("LOSE by opponent's double reach")
    show_net(Net(), State().play("B2 A2 A3 C1 B3"))

    print("WIN through double reach")
    show_net(Net(), State().play("B2 A2 A3 C1"))

    print("strategic WIN by following double")
    show_net(Net(), State().play("B1 A3"))

def demo_mcts() -> None:
    M = cfg.mcts
    demo = M.num_simulations_demo
    demo_mid = M.num_simulations_demo_mid

    tree = Tree(Net())
    tree.think(State(), demo, show=True)

    tree = Tree(Net())
    tree.think(State().play("A1 C1 A2 C2"), demo_mid, show=True)

    tree = Tree(Net())
    tree.think(State().play("B2 A2 A3 C1 B3"), demo_mid, show=True)

    tree = Tree(Net())
    tree.think(State().play("B2 A2 A3 C1"), demo_mid, show=True)

def self_play_and_train():
    T = cfg.training
    M = cfg.mcts
    num_games = T.num_games
    num_train_steps = T.num_train_steps
    num_simulations = M.num_simulations_train

    net = Net()
    episodes = []
    result_distribution = {1: 0, 0: 0, -1: 0}

    print("vs_random = ", sorted(vs_random(net).items()))

    for g in range(num_games):
        record, p_targets = [], []
        state = State()
        tree = Tree(net)
        temperature = M.temperature_init
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
            net = train(episodes)
            print("vs_random = ", sorted(vs_random(net).items()))

    print("finished")
    return net

if __name__ == "__main__":
    # 必要に応じてコメントアウトを外して実行してください。
    demo_network_outputs()
    # demo_mcts()
    # trained = self_play_and_train()
