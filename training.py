# training.py
from __future__ import annotations
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple
from game import State
from network import Net
from config.loader import cfg

T = cfg.training
BATCH_SIZE = T.batch_size
NUM_EPOCHS = T.num_epochs
LR = T.lr
WEIGHT_DECAY = T.weight_decay
MOMENTUM = T.momentum
LR_DECAY = T.lr_decay

def gen_target(ep: Tuple[List[int], int, List[np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    turn_idx = np.random.randint(len(ep[0]))
    state = State()
    for a in ep[0][:turn_idx]:
        state.play(a)
    v = ep[1]
    x = state.feature()
    p_target = ep[2][turn_idx]
    v_target = np.array([v if turn_idx % 2 == 0 else -v], dtype=np.float32)
    return x, p_target.astype(np.float32), v_target

def train(episodes: List[Tuple[List[int], int, List[np.ndarray]]]) -> Net:
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

    for epoch in range(NUM_EPOCHS):
        p_loss_sum, v_loss_sum = 0.0, 0.0
        net.train()
        for _ in range(0, len(episodes), BATCH_SIZE):
            batch = [gen_target(episodes[np.random.randint(len(episodes))]) for _ in range(BATCH_SIZE)]
            x, p_target, v_target = zip(*batch)
            x = torch.FloatTensor(np.array(x))
            p_target = torch.FloatTensor(np.array(p_target))
            v_target = torch.FloatTensor(np.array(v_target))

            p, v = net(x)
            p_loss = torch.sum(-p_target * torch.log(p + 1e-12))
            v_loss = torch.sum((v_target - v) ** 2)

            p_loss_sum += float(p_loss.item())
            v_loss_sum += float(v_loss.item())

            optimizer.zero_grad()
            (p_loss + v_loss).backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] *= LR_DECAY

    print(f"p_loss {p_loss_sum / max(len(episodes),1):.6f} v_loss {v_loss_sum / max(len(episodes),1):.6f}")
    return net

def vs_random(net: Net, n: int | None = None) -> Dict[int, int]:
    if n is None:
        n = T.vs_random_matches
    results: Dict[int, int] = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = State()
        while not state.terminal():
            if turn:
                p, _ = net.predict(state)
                legal = state.legal_actions()
                action = sorted([(a, p[a]) for a in legal], key=lambda x: -x[1])[0][0]
            else:
                action = np.random.choice(state.legal_actions())
            state.play(action)
            turn = not turn
        r = state.terminal_reward() if turn else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
    return results

def show_net(net: Net, state: State) -> None:
    print(state)
    p, v = net.predict(state)
    n = state.size
    print("p = ")
    print((p * 1000).astype(int).reshape((n, n)))
    print("v = ", v)
    print()
