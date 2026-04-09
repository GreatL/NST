import argparse
import os
import random
import warnings

import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Embedding, Conv1d, MaxPool1d
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected, negative_sampling
from torch_geometric.nn import GCNConv, global_sort_pool

warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# 通用工具
# =========================

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_edge_list(path):
    raw_edges = []
    node_map = {}
    cur = 0
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u not in node_map:
                node_map[u] = cur
                cur += 1
            if v not in node_map:
                node_map[v] = cur
                cur += 1
            raw_edges.append((node_map[u], node_map[v]))
    raw_edges = np.array(raw_edges).T
    return raw_edges, len(node_map)


def do_edge_split(edge_index, num_nodes, val_ratio=0.05, test_ratio=0.1):
    """
    将无向边随机划分为 train/valid/test 正边。
    注意：这里只划分正边，不生成 neg。
    """
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]

    n = row.size(0)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    perm = torch.randperm(n)
    row, col = row[perm], col[perm]

    split = {"train": {}, "valid": {}, "test": {}}

    split["valid"]["pos"] = torch.stack([row[:n_val], col[:n_val]])
    split["test"]["pos"] = torch.stack([row[n_val:n_val + n_test],
                                       col[n_val:n_val + n_test]])
    split["train"]["pos"] = torch.stack([row[n_val + n_test:], col[n_val + n_test:]])

    return split


# =========================
# SEAL 子图构造
# =========================

def neighbors(fringe, A):
    return set(A[list(fringe)].indices)


def k_hop_subgraph(src, dst, k, A):
    nodes = [src, dst]
    visited = set([src, dst])
    fringe = set([src, dst])
    for _ in range(1, k + 1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited |= fringe
        if len(fringe) == 0:
            break
        nodes += list(fringe)
    sub = A[nodes][:, nodes]
    sub[0, 1] = 0
    sub[1, 0] = 0
    return nodes, sub


def drnl(adj, src, dst):
    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_s = adj[idx][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_d = adj[idx][:, idx]

    d2s = shortest_path(adj_d, directed=False, unweighted=True, indices=src)
    d2d = shortest_path(adj_s, directed=False, unweighted=True, indices=dst - 1)

    d2s = np.insert(d2s, dst, 0)
    d2d = np.insert(d2d, src, 0)

    dist = d2s + d2d
    dist2 = dist // 2
    distm = dist % 2

    z = 1 + np.minimum(d2s, d2d)
    z += dist2 * (dist2 + distm - 1)

    z[src] = 1
    z[dst] = 1
    z[np.isnan(z)] = 0

    return torch.LongTensor(z)


def construct_graph_seal(nodes, sub, y):
    u, v, _ = ssp.find(sub)
    edge_index = torch.LongTensor([u, v])
    z = drnl(sub, 0, 1)
    return Data(
        edge_index=edge_index,
        z=z,
        y=torch.tensor([y], dtype=torch.float),
        num_nodes=sub.shape[0],
    )


def extract_seal(edge_index, A, k, y):
    data_list = []
    for u, v in tqdm(edge_index.T.tolist(), disable=True):
        nodes, sub = k_hop_subgraph(u, v, k, A)
        data_list.append(construct_graph_seal(nodes, sub, y))
    return data_list


# =========================
# 固定 flatten 维度的 DGCNN
# =========================

class DGCNN(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, max_z, k_sort):
        super().__init__()
        self.emb = Embedding(max_z, hidden_dim)

        self.convs = ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, 1))

        self.k = k_sort
        total_dim = hidden_dim * num_layers + 1  # concat 特征维度

        self.conv1 = Conv1d(1, 16, kernel_size=5, stride=1)
        self.pool = MaxPool1d(2, 2)
        self.conv2 = Conv1d(16, 32, kernel_size=5, stride=1)

        # 通过虚拟张量推一遍 conv 流程，确定 flatten 后的 in_dim
        with torch.no_grad():
            dummy = torch.zeros(1, self.k * total_dim)
            dummy = dummy.unsqueeze(1)  # [1,1,L]
            x = F.relu(self.conv1(dummy))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            in_dim = x.view(1, -1).size(1)

        self.lin1 = Linear(in_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch):
        x = self.emb(z)

        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)  # [N_sub, total_dim]

        x = global_sort_pool(x, batch, self.k)  # [B, k * total_dim]
        x = x.unsqueeze(1)  # [B,1,L]

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x).view(-1)
        return x


# =========================
# two-hop-aware 训练负采样
# =========================

def build_twohop_negatives(train_pos, A_train, num_nodes,
                           n_rand_per_pos=1, n_twohop_per_pos=1):
    pos_edges = train_pos.T.tolist()
    neg_edges = []
    all_nodes = list(range(num_nodes))

    for (u, v) in pos_edges:
        # 全局 random neg
        for _ in range(n_rand_per_pos):
            while True:
                w = random.choice(all_nodes)
                if w == u:
                    continue
                if A_train[u, w] == 0 and A_train[w, u] == 0:
                    neg_edges.append((u, w))
                    break
        # two-hop neg
        neighbors_u_1 = set(A_train[u].indices)
        neighbors_u_2 = set()
        for x in neighbors_u_1:
            neighbors_u_2.update(A_train[x].indices)
        twohop = (neighbors_u_1 | neighbors_u_2) - {u}
        twohop = [w for w in twohop if A_train[u, w] == 0 and A_train[w, u] == 0]
        twohop = list(twohop)
        for _ in range(n_twohop_per_pos):
            if not twohop:
                break
            w = random.choice(twohop)
            neg_edges.append((u, w))

    if len(neg_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    neg_edges = np.array(neg_edges, dtype=np.int64).T
    return torch.from_numpy(neg_edges)


def build_random_negatives(pos_edges, num_nodes, num_neg):
    all_nodes = list(range(num_nodes))
    pos_set = set([(int(u), int(v)) for u, v in pos_edges.T.tolist()] +
                  [(int(v), int(u)) for u, v in pos_edges.T.tolist()])
    neg_edges = []
    while len(neg_edges) < num_neg:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u == v:
            continue
        if (u, v) in pos_set:
            continue
        neg_edges.append((u, v))
    neg_edges = np.array(neg_edges, dtype=np.int64).T
    return torch.from_numpy(neg_edges)


# =========================
# 训练 / 验证
# =========================

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for d in loader:
        d = d.to(device)
        opt.zero_grad()
        logits = model(d.z, d.edge_index, d.batch)
        y = d.y.view(-1).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * d.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_auc_ap(model, loader, device):
    model.eval()
    yp, yt = [], []
    for d in loader:
        d = d.to(device)
        logit = model(d.z, d.edge_index, d.batch)
        yp.append(logit.cpu())
        yt.append(d.y.view(-1).cpu())
    yp = torch.cat(yp).numpy()
    yt = torch.cat(yt).numpy()
    auc = roc_auc_score(yt, yp)
    ap = average_precision_score(yt, yp)
    return auc, ap


# =========================
# 主流程
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_path", type=str, required=True)

    parser.add_argument("--num_hops", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--max_z", type=int, default=1000)
    parser.add_argument("--k_sort", type=int, default=30)

    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--n_rand_per_pos", type=int, default=1)
    parser.add_argument("--n_twohop_per_pos", type=int, default=1)

    parser.add_argument("--model_dir", type=str, default="models_twohop")
    parser.add_argument("--model_name", type=str, default=None)

    args = parser.parse_args()
    setup_seed(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    e, n = load_edge_list(args.edge_path)
    edge_index = torch.LongTensor(e)
    edge_index = to_undirected(edge_index)

    split = do_edge_split(
        edge_index, n,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    train_pos = split["train"]["pos"]

    # A_train
    A_train = ssp.csr_matrix(
        (np.ones(train_pos.size(1)),
         (train_pos[0], train_pos[1])),
        shape=(n, n)
    )
    A_train = A_train + A_train.T

    # two-hop-aware train neg
    print("Building two-hop-aware training negatives...")
    train_neg = build_twohop_negatives(
        train_pos, A_train, n,
        n_rand_per_pos=args.n_rand_per_pos,
        n_twohop_per_pos=args.n_twohop_per_pos
    )
    print(f"Train pos edges: {train_pos.size(1)}, train neg edges: {train_neg.size(1)}")

    # valid/test random neg (1:1)
    valid_pos = split["valid"]["pos"]
    test_pos = split["test"]["pos"]

    print("Building random negatives for valid/test...")
    valid_neg = build_random_negatives(valid_pos, n, valid_pos.size(1))
    test_neg = build_random_negatives(test_pos, n, test_pos.size(1))

    # subgraphs
    print("Extracting train subgraphs...")
    train_pos_data = extract_seal(train_pos, A_train, args.num_hops, 1)
    train_neg_data = extract_seal(train_neg, A_train, args.num_hops, 0)
    train_data = train_pos_data + train_neg_data

    print("Extracting valid subgraphs...")
    valid_pos_data = extract_seal(valid_pos, A_train, args.num_hops, 1)
    valid_neg_data = extract_seal(valid_neg, A_train, args.num_hops, 0)
    valid_data = valid_pos_data + valid_neg_data

    print("Extracting test subgraphs...")
    test_pos_data = extract_seal(test_pos, A_train, args.num_hops, 1)
    test_neg_data = extract_seal(test_neg, A_train, args.num_hops, 0)
    test_data = test_pos_data + test_neg_data

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGCNN(
        hidden_dim=args.hidden,
        num_layers=args.layers,
        max_z=args.max_z,
        k_sort=args.k_sort
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Start training with two-hop-aware negatives...")
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, device)
        val_auc, val_ap = eval_auc_ap(model, val_loader, device)
        if ep % 5 == 0 or ep == args.epochs:
            print(f"Epoch {ep} | Loss={loss:.4f} | Val AUC={val_auc:.4f} | Val AP={val_ap:.4f}")

    test_auc, test_ap = eval_auc_ap(model, test_loader, device)
    print(f"Final TEST AUC (random neg, 1:1) = {test_auc:.4f}")
    print(f"Final TEST AP  (random neg, 1:1) = {test_ap:.4f}")

    dataset_name = os.path.splitext(os.path.basename(args.edge_path))[0]
    base_name = args.model_name or (dataset_name + f"_seal_twohop_nr{args.n_rand_per_pos}_nt{args.n_twohop_per_pos}")
    model_path = os.path.join(args.model_dir, f"{base_name}.pt")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "num_nodes": n,
        "args": vars(args),
    }
    torch.save(ckpt, model_path)
    print(f"Two-hop-aware model saved to: {model_path}")
