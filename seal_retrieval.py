import argparse
import os
import random
import warnings
import time

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
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv, global_sort_pool

warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# 工具
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

    # valid/test 用 random neg 做 global LP 评测
    def build_random_neg(pos_edges):
        all_nodes = list(range(num_nodes))
        pos_set = set([(int(u), int(v)) for u, v in pos_edges.T.tolist()] +
                      [(int(v), int(u)) for u, v in pos_edges.T.tolist()])
        neg_edges = []
        while len(neg_edges) < pos_edges.size(1):
            u = random.choice(all_nodes)
            v = random.choice(all_nodes)
            if u == v:
                continue
            if (u, v) in pos_set:
                continue
            neg_edges.append((u, v))
        neg_edges = np.array(neg_edges, dtype=np.int64).T
        return torch.from_numpy(neg_edges)

    split["valid"]["neg"] = build_random_neg(split["valid"]["pos"])
    split["test"]["neg"] = build_random_neg(split["test"]["pos"])

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


def construct_graph_seal(nodes, sub, y, src=None):
    u, v, _ = ssp.find(sub)
    edge_index = torch.LongTensor([u, v])
    z = drnl(sub, 0, 1)
    data = Data(
        edge_index=edge_index,
        z=z,
        y=torch.tensor([y], dtype=torch.float),
        num_nodes=sub.shape[0],
    )
    if src is not None:
        data.src = torch.tensor([src], dtype=torch.long)
    return data


def extract_seal(edge_index, A, k, y, with_src=False):
    data_list = []
    for u, v in tqdm(edge_index.T.tolist(), disable=True):
        nodes, sub = k_hop_subgraph(u, v, k, A)
        data_list.append(construct_graph_seal(nodes, sub, y, src=u if with_src else None))
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
        total_dim = hidden_dim * num_layers + 1

        self.conv1 = Conv1d(1, 16, kernel_size=5, stride=1)
        self.pool = MaxPool1d(2, 2)
        self.conv2 = Conv1d(16, 32, kernel_size=5, stride=1)

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
        x = torch.cat(xs[1:], dim=-1)

        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.lin1(x))
        x = self.lin2(x).view(-1)
        return x


# =========================
# 训练 / 全局评测
# =========================

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
def infer_scores(model, loader, device):
    model.eval()
    yp, yt, srcs = [], [], []
    for d in loader:
        d = d.to(device)
        logit = model(d.z, d.edge_index, d.batch)
        yp.append(logit.cpu())
        yt.append(d.y.view(-1).cpu())
        if hasattr(d, "src"):
            srcs.append(d.src.view(-1).cpu())
    yp = torch.cat(yp).numpy()
    yt = torch.cat(yt).numpy()
    if srcs:
        srcs = torch.cat(srcs).numpy()
    else:
        srcs = None
    return yp, yt, srcs


@torch.no_grad()
def eval_global_auc_ap(model, data_list, device, batch_size=256):
    loader = DataLoader(data_list, batch_size=batch_size)
    scores, labels, _ = infer_scores(model, loader, device)
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap


# =========================
# 候选生成与 retrieval 评测
# =========================

def generate_candidates_random_k(u, v, num_nodes, K):
    all_nodes = list(range(num_nodes))
    if u in all_nodes:
        all_nodes.remove(u)
    cand = random.sample(all_nodes, min(K - 1, len(all_nodes)))
    if v not in cand:
        if len(cand) < K:
            cand.append(v)
        else:
            cand[0] = v
    return cand


def generate_candidates_2hop(u, v, A_train, num_nodes, K):
    neighbors_u_1 = set(A_train[u].indices)
    neighbors_u_2 = set()
    for x in neighbors_u_1:
        neighbors_u_2.update(A_train[x].indices)
    cand = neighbors_u_1 | neighbors_u_2
    cand.discard(u)

    cand.add(v)
    cand_list = list(cand)

    if len(cand_list) > K:
        if v in cand_list:
            cand_list.remove(v)
            sampled = random.sample(cand_list, K - 1)
            sampled.append(v)
            cand_list = sampled
        else:
            cand_list = random.sample(cand_list, K)

    if len(cand_list) < K:
        needed = K - len(cand_list)
        existing = set(cand_list)
        existing.add(u)
        pool = [x for x in range(num_nodes) if x not in existing]
        if len(pool) > needed:
            extra = random.sample(pool, needed)
        else:
            extra = pool
        cand_list.extend(extra)

    if v not in cand_list:
        if len(cand_list) > 0:
            cand_list[0] = v
        else:
            cand_list = [v]

    return cand_list


def generate_candidates_cn_topk(u, v, A_train, num_nodes, K):
    neighbors_u = set(A_train[u].indices)
    cn_scores = np.zeros(num_nodes, dtype=np.int64)
    for w in range(num_nodes):
        if w == u:
            cn_scores[w] = -1
            continue
        neighbors_w = set(A_train[w].indices)
        cn_scores[w] = len(neighbors_u & neighbors_w)
    indices = np.argsort(-cn_scores)
    cand = []
    for idx in indices:
        if idx == u:
            continue
        cand.append(idx)
        if len(cand) >= K - 1:
            break
    if v not in cand:
        cand.append(v)
    if len(cand) > K:
        cand.remove(v)
        cand = random.sample(cand, K - 1)
        cand.append(v)
    return cand


def build_retrieval_data(test_pos, A_train, num_nodes, K, num_hops, candidate_strategy):
    edges = []
    for u, v in test_pos.T.tolist():
        if candidate_strategy == "random_k":
            cand_nodes = generate_candidates_random_k(u, v, num_nodes, K)
        elif candidate_strategy == "two_hop":
            cand_nodes = generate_candidates_2hop(u, v, A_train, num_nodes, K)
        elif candidate_strategy == "cn_topk":
            cand_nodes = generate_candidates_cn_topk(u, v, A_train, num_nodes, K)
        else:
            raise ValueError(f"Unknown candidate_strategy: {candidate_strategy}")
        edges.append((u, v, 1))
        for w in cand_nodes:
            if w == v:
                continue
            edges.append((u, w, 0))
    edges = np.array(edges, dtype=np.int64)
    data_list = []
    for (u, v, y) in tqdm(edges.tolist(), disable=True):
        nodes, sub = k_hop_subgraph(u, v, num_hops, A_train)
        data_list.append(construct_graph_seal(nodes, sub, y, src=u))
    return data_list


@torch.no_grad()
def eval_retrieval(model, data_list, device, batch_size=256, Ks=(10, 20)):
    if len(data_list) == 0:
        return {f"Prec@{K}": None for K in Ks} | {"MRR": None, "mAP_local": None}
    loader = DataLoader(data_list, batch_size=batch_size)
    scores, labels, srcs = infer_scores(model, loader, device)
    per_u = {}
    for i in range(len(scores)):
        u = int(srcs[i])
        per_u.setdefault(u, []).append((scores[i], labels[i]))
    K_list = list(Ks)
    prec_at_k = {K: [] for K in K_list}
    mrr_list = []
    ap_local_list = []
    for u, lst in per_u.items():
        lst_sorted = sorted(lst, key=lambda x: -x[0])
        scores_u = np.array([s for s, _ in lst_sorted], dtype=np.float64)
        labels_u = np.array([y for _, y in lst_sorted], dtype=np.int64)
        num_pos = labels_u.sum()
        if num_pos == 0:
            continue
        for K in K_list:
            k = min(K, len(labels_u))
            prec = labels_u[:k].sum() / float(k)
            prec_at_k[K].append(prec)
        pos_indices = np.where(labels_u == 1)[0]
        if len(pos_indices) > 0:
            rank1 = pos_indices[0] + 1
            mrr_list.append(1.0 / rank1)
        try:
            ap_local = average_precision_score(labels_u, scores_u)
            ap_local_list.append(ap_local)
        except Exception:
            pass
    metrics = {}
    for K in K_list:
        metrics[f"Prec@{K}"] = float(f"{np.mean(prec_at_k[K]):.4f}") if prec_at_k[K] else None
    metrics["MRR"] = float(f"{np.mean(mrr_list):.4f}") if mrr_list else None
    metrics["mAP_local"] = float(f"{np.mean(ap_local_list):.4f}") if ap_local_list else None
    return metrics


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
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--retrieval_K", type=int, default=50)
    parser.add_argument("--candidate_strategy", type=str, default="two_hop",
                        choices=["random_k", "two_hop", "cn_topk"])
    parser.add_argument("--eval_global", action="store_true")
    parser.add_argument("--load_model", type=str, default=None)
    args = parser.parse_args()

    setup_seed(args.seed)
    e, n = load_edge_list(args.edge_path)
    edge_index = torch.LongTensor(e)
    edge_index = to_undirected(edge_index)

    split = do_edge_split(edge_index, n, args.val_ratio, args.test_ratio)
    train_pos = split["train"]["pos"]
    A_train = ssp.csr_matrix(
        (np.ones(train_pos.size(1)),
         (train_pos[0], train_pos[1])),
        shape=(n, n)
    )
    A_train = A_train + A_train.T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGCNN(args.hidden, args.layers, args.max_z, args.k_sort).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.load_model is not None:
        ckpt = torch.load(args.load_model, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded model from {args.load_model}")
    else:
        # 构造 random 训练负例
        train_neg = build_random_negatives(train_pos, n, train_pos.size(1))
        train_pos_data = extract_seal(train_pos, A_train, args.num_hops, 1, with_src=False)
        train_neg_data = extract_seal(train_neg, A_train, args.num_hops, 0, with_src=False)
        train_data = train_pos_data + train_neg_data

        valid_pos = split["valid"]["pos"]
        valid_neg = split["valid"]["neg"]
        valid_pos_data = extract_seal(valid_pos, A_train, args.num_hops, 1, with_src=False)
        valid_neg_data = extract_seal(valid_neg, A_train, args.num_hops, 0, with_src=False)
        valid_data = valid_pos_data + valid_neg_data

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(valid_data, batch_size=args.batch_size)

        print("Start training (random negatives)...")
        for ep in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, opt, device)
            if ep % 5 == 0 or ep == args.epochs:
                val_auc, val_ap = eval_global_auc_ap(model, valid_data, device, args.batch_size)
                print(f"Epoch {ep} | Loss={loss:.4f} | Val AUC={val_auc:.4f} | Val AP={val_ap:.4f}")

    # global eval
    if args.eval_global:
        test_pos = split["test"]["pos"]
        test_neg = split["test"]["neg"]
        test_pos_data = extract_seal(test_pos, A_train, args.num_hops, 1, with_src=False)
        test_neg_data = extract_seal(test_neg, A_train, args.num_hops, 0, with_src=False)
        test_data = test_pos_data + test_neg_data
        global_auc, global_ap = eval_global_auc_ap(model, test_data, device, args.batch_size)
        print(f"Global LP AUC={global_auc:.4f}, AP={global_ap:.4f}")

    # retrieval eval
    test_pos = split["test"]["pos"]
    print(f"Building retrieval candidates ({args.candidate_strategy}, K={args.retrieval_K})...")
    retrieval_data = build_retrieval_data(
        test_pos, A_train, n, args.retrieval_K, args.num_hops,
        args.candidate_strategy
    )
    print(f"Total retrieval candidate edges: {len(retrieval_data)}")
    metrics = eval_retrieval(model, retrieval_data, device,
                             batch_size=args.batch_size, Ks=(10, 20))
    print("Retrieval metrics:", metrics)
