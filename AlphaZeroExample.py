# AlphaZeroExample: 5ファイル構成へのリファクタ



# ゲームの実装 ここを他のドメインの実装に置き換えれば色々なゲームで動かせる
# 高速な言語でこの部分のみ実装すると手軽に高速化出来る(C++ならpybind11やBoost.Pythonを利用可)

import numpy as np

BLACK, WHITE =1, -1 # 先手後手

class State:
    '''○×ゲームの盤面実装'''
    X, Y = 'ABC',  '123'
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self):
        self.board = np.zeros((3, 3)) # (x, y)
        self.color = 1
        self.win_color = 0
        self.record = []

    def action2str(self, a):
        return self.X[a // 3] + self.Y[a % 3]

    def str2action(self, s):
        return self.X.find(s[0]) * 3 + self.Y.find(s[1])

    def record_string(self):
        return ' '.join([self.action2str(a) for a in self.record])

    def __str__(self):
        # 表示
        s = '   ' + ' '.join(self.Y) + '\n'
        for i in range(3):
            s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(3)]) + '\n'
        s += 'record = ' + self.record_string()
        return s

    def play(self, action):
        # 行動で状態を進める関数
        # action は board 上の位置 (0 ~ 8) または行動系列の文字列
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return self

        x, y = action // 3, action % 3
        self.board[x, y] = self.color

        # 3つ揃ったか調べる
        if self.board[x, :].sum() == 3 * self.color \
          or self.board[:, y].sum() == 3 * self.color \
          or (x == y and np.diag(self.board, k=0).sum() == 3 * self.color) \
          or (x == 2 - y and np.diag(self.board[::-1,:], k=0).sum() == 3 * self.color):
            self.win_color = self.color

        self.color = -self.color
        self.record.append(action)
        return self

    def terminal(self):
        # 終端状態かどうか返す
        return self.win_color != 0 or len(self.record) == 3 * 3

    def terminal_reward(self):
        # 終端状態での勝敗による報酬を返す
        return self.win_color if self.color == BLACK else -self.win_color

    def legal_actions(self):
        # 可能な行動リストを返す
        return [a for a in range(3 * 3) if self.board[a // 3, a % 3] == 0]

    def feature(self):
        # ニューラルネットに入力する状態表現を返す
        return np.stack([self.board == self.color, self.board == -self.color]).astype(np.float32)


state = State().play('B1')
print(state)
print('input feature')
print(state.feature())
state = State().play('B2 A1 C2')
print('input feature')
print(state.feature())

# ニューラルネットの実装(PyTorch)
# AlphaZeroの論文のネットワーク構成を小さく再現

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))

num_filters = 16
num_blocks = 6

class Net(nn.Module):
    '''ニューラルネット計算を行うクラス'''
    def __init__(self):
        super().__init__()
        state = State()
        self.input_shape = state.feature().shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

        self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
        self.conv_p2 = Conv(4, 1, 1)

        self.conv_v = Conv(num_filters, 4, 1, bn=True)
        self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)

        h_p = F.relu(self.conv_p1(h))
        h_p = self.conv_p2(h_p).view(-1, self.board_size)

        h_v = F.relu(self.conv_v(h))
        h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

        # value(状態価値)にtanhを適用するので負け -1 ~ 勝ち 1
        return F.softmax(h_p, dim=-1), torch.tanh(h_v)

    def predict(self, state):
        # 探索中に呼ばれる推論関数
        self.eval()
        x = torch.from_numpy(state.feature()).unsqueeze(0)
        with torch.no_grad():
            p, v = self.forward(x)
        return p.cpu().numpy()[0], v.cpu().numpy()[0][0]

def show_net(net, state):
    '''方策 (p)　と　状態価値 (v) を表示'''
    print(state)
    p, v = net.predict(state)
    print('p = ')
    print((p *1000).astype(int).reshape((-1, *net.input_shape[1:3])))
    print('v = ', v)
    print()

# 学習前なのでランダムな出力を得る
show_net(Net(), State())



# モンテカルロ木探索の実装

class Node:
    '''ある1状態の探索結果を保存するクラス'''
    def __init__(self, p, v):
        self.p, self.v = p, v
        self.n, self.q_sum = np.zeros_like(p), np.zeros_like(p)
        self.n_all, self.q_sum_all = 1, v / 2 # 事前分布(見解が分かれる点)

    def update(self, action, q_new):
        # 行動のスタッツを更新
        self.n[action] += 1
        self.q_sum[action] += q_new

        # ノード全体のスタッツも更新
        self.n_all += 1
        self.q_sum_all += q_new

import time, copy

class Tree:
    '''探索木を保持してモンテカルロ木探索を行うクラス'''
    def __init__(self, net):
        self.net = net
        self.nodes = {}
    
    def search(self, state, depth):
        # 終端状態の場合は末端報酬を返す
        if state.terminal():
            return state.terminal_reward()

        # まだ未到達の状態はニューラルネットを計算して推定価値を返す
        key = state.record_string()
        if key not in self.nodes:
            p, v = self.net.predict(state)
            self.nodes[key] = Node(p, v)
            return v

        # 到達済みの状態はバンディットで行動を選んで状態を進める
        node = self.nodes[key]
        p = node.p
        if depth == 0:
            # ルートノード(現局面)では方策にノイズを加える
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.15] * len(p))

        best_action, best_ucb = None, -float('inf')
        for action in state.legal_actions():
            n, q_sum = 1 + node.n[action], node.q_sum_all / node.n_all + node.q_sum[action]
            ucb = q_sum / n + 2.0 * np.sqrt(node.n_all) * p[action] / n # PUCBの式

            if ucb > best_ucb:
                best_action, best_ucb = action, ucb

        # 一手進めて再帰で探索
        state.play(best_action)
        q_new = -self.search(state, depth + 1) # 1手ごとの手番交代を想定
        node.update(best_action, q_new)

        return q_new

    def think(self, state, num_simulations, temperature = 0, show=False):
        # 探索のエンドポイント
        if show:
            print(state)
        start, prev_time = time.time(), 0
        for _ in range(num_simulations):
            self.search(copy.deepcopy(state), depth=0)

            # 1秒ごとに探索結果を表示
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root, pv = self.nodes[state.record_string()], self.pv(state)
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s'
                          % (tmp_time, state.action2str(pv[0]), root.q_sum[pv[0]] / root.n[pv[0]],
                             root.n[pv[0]], root.n_all, ' '.join([state.action2str(a) for a in pv])))

        #  訪問回数で重みつけた確率分布を返す
        n = root = self.nodes[state.record_string()].n + 1
        n = (n / np.max(n)) ** (1 / (temperature + 1e-8))
        return n / n.sum()

    def pv(self, state):
        # 最善応手列（読み筋）を返す
        s, pv_seq = copy.deepcopy(state), []
        while True:
            key = s.record_string()
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            best_action = sorted([(a, self.nodes[key].n[a]) for a in s.legal_actions()], key=lambda x: -x[1])[0][0]
            pv_seq.append(best_action)
            s.play(best_action)
        return pv_seq

# 初期ネットワークで探索を行う
tree = Tree(Net())
tree.think(State(), 1000, show=True)

tree = Tree(Net())
tree.think(State().play('A1 C1 A2 C2'), 10000, show=True)

tree = Tree(Net())
tree.think(State().play('B2 A2 A3 C1 B3'), 10000, show=True)

tree = Tree(Net())
tree.think(State().play('B2 A2 A3 C1'), 10000, show=True)


# ニューラルネットの学習

import torch.optim as optim

batch_size = 32
num_epochs = 30

def gen_target(ep):
    '''ニューラルネットの学習用 input, targets を生成'''
    turn_idx = np.random.randint(len(ep[0]))
    state = State()
    for a in ep[0][:turn_idx]:
        state.play(a)
    v = ep[1]
    return state.feature(), ep[2][turn_idx], [v if turn_idx % 2 == 0 else -v]

def train(episodes):
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.75)
    for epoch in range(num_epochs):
        p_loss_sum, v_loss_sum = 0, 0
        net.train()
        for i in range(0, len(episodes), batch_size):
            x, p_target, v_target = zip(*[gen_target(episodes[np.random.randint(len(episodes))]) for j in range(batch_size)])
            x = torch.FloatTensor(np.array(x))
            p_target = torch.FloatTensor(np.array(p_target))
            v_target = torch.FloatTensor(np.array(v_target))

            p, v = net(x)
            p_loss = torch.sum(-p_target * torch.log(p))
            v_loss = torch.sum((v_target - v) ** 2)

            p_loss_sum += p_loss.item()
            v_loss_sum += v_loss.item()

            optimizer.zero_grad()
            (p_loss + v_loss).backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.85
    print('p_loss %f v_loss %f' % (p_loss_sum / len(episodes), v_loss_sum / len(episodes)))
    return net

#  ランダムプレーヤーとの対戦実験

def vs_random(net, n=100):
    results = {}
    for i in range(n):
        first_turn = i % 2 == 0
        turn = first_turn
        state = State()
        while not state.terminal():
            if turn:
                p, _ = net.predict(state)
                action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x:-x[1])[0][0]
            else:
                action = np.random.choice(state.legal_actions())
            state.play(action)
            turn = not turn
        r = state.terminal_reward() if turn else -state.terminal_reward()
        results[r] = results.get(r, 0) + 1
    return results

# AlphaZeroのアルゴリズムメイン

num_games = 500
num_train_steps = 50
num_simulations = 50

net = Net()
episodes = []
result_distribution = {1:0, 0:0, -1:0}
print('vs_random = ', sorted(vs_random(net).items()))
for g in range(num_games):
    # 1対戦のエピソード生成
    record, p_targets = [], []
    state = State()
    tree = Tree(net)
    temperature = 0.7 # 探索結果から方策のtargetを作るときの温度
    while not state.terminal():
        p_target = tree.think(state, num_simulations, temperature)
        # 行動をランダムに選んで進める
        action = np.random.choice(np.arange(len(p_target)), p=p_target)
        state.play(action)
        record.append(action)
        p_targets.append(p_target)
        temperature *= 0.8
    # 先手視点の報酬
    reward = state.terminal_reward() * (1 if len(record) % 2 == 0 else -1)
    result_distribution[reward] += 1
    episodes.append((record, reward, p_targets))
    if g % num_train_steps == 0:
        print('game ', end='')
    print(g, ' ', end='')

    # ニューラルネットの学習
    if (g + 1) % num_train_steps == 0:
        # 先手勝、引き分け、後手勝ちの割合を表示
        print('generated = ', sorted(result_distribution.items()))
        net = train(episodes)
        print('vs_random = ', sorted(vs_random(net).items()))
print('finished')