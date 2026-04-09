import argparse
import random
import warnings
import time

import numpy as np
import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# 工具函数
# =========================

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_edge_list(path):
    """
    从边列表文件读取边，返回:
    - raw_edges: np.array shape (2, E)
    - num_nodes: 节点数
    文件格式: 每行 "u v"
    """
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
    与 SEAL 的划分方式类似：只划分正边，不生成 neg。
    """
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]

    n_edges = row.size  # 修正：size 是属性，不是方法
    n_val = int(n_edges * val_ratio)
    n_test = int(n_edges * test_ratio)

    perm = np.random.permutation(n_edges)
    row, col = row[perm], col[perm]

    split = {"train": {}, "valid": {}, "test": {}}
    split["valid"]["pos"] = np.stack([row[:n_val], col[:n_val]], axis=0)
    split["test"]["pos"] = np.stack([row[n_val:n_val + n_test],
                                     col[n_val:n_val + n_test]], axis=0)
    split["train"]["pos"] = np.stack([row[n_val + n_test:], col[n_val + n_test:]], axis=0)
    return split


# =========================
# 邻接 / 两跳邻域 / 启发式
# =========================

def build_adj(num_nodes, edges):
    """
    edges: np.array shape [2, E]
    返回 scipy csr 邻接矩阵 (无向、不含自环)
    """
    row, col = edges
    data = np.ones(row.shape[0], dtype=np.float32)
    A = ssp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def neighbors_1hop(A, u):
    """返回 u 的 1-hop 邻居集合"""
    return set(A[u].indices)


def neighbors_2hop(A, u):
    """返回 u 的 2-hop 邻居集合（包含 1-hop 和 2-hop，去掉 u 自身）"""
    n1 = neighbors_1hop(A, u)
    n2 = set()
    for x in n1:
        n2.update(A[x].indices)
    pool = (n1 | n2)
    if u in pool:
        pool.remove(u)
    return pool


def common_neighbors(A, u, v):
    """CN(u,v) = |Gamma(u) ∩ Gamma(v)|"""
    nu = neighbors_1hop(A, u)
    nv = neighbors_1hop(A, v)
    return len(nu & nv)


def resource_allocation(A, u, v):
    """RA(u,v) = sum_{x in Gamma(u)∩Gamma(v)} 1/deg(x)"""
    nu = neighbors_1hop(A, u)
    nv = neighbors_1hop(A, v)
    intersect = nu & nv
    if not intersect:
        return 0.0
    deg = A.sum(axis=1).A1  # 每个节点度
    score = sum(1.0 / deg[x] for x in intersect if deg[x] > 0)
    return score


# =========================
# 全局启发式评测 (global LP)
# =========================

def build_global_negatives(num_nodes, pos_edges, num_neg):
    """
    在全图上 random 采样 num_neg 个负例，确保不与 pos_edges 重合。
    pos_edges: np.array [2, E_pos]
    """
    pos_set = set([(int(u), int(v)) for u, v in pos_edges.T] +
                  [(int(v), int(u)) for u, v in pos_edges.T])
    neg_edges = []
    all_nodes = list(range(num_nodes))
    while len(neg_edges) < num_neg:
        u = random.choice(all_nodes)
        v = random.choice(all_nodes)
        if u == v:
            continue
        if (u, v) in pos_set:
            continue
        neg_edges.append((u, v))
    neg_edges = np.array(neg_edges, dtype=np.int64).T
    return neg_edges


def eval_global_heuristic(A, num_nodes, pos_edges, heuristic="CN"):
    """
    在 global 协议下评估 CN/RA 的 AUC 和 AP。
    pos_edges: np.array [2, E_test_pos]
    """
    num_pos = pos_edges.shape[1]
    neg_edges = build_global_negatives(num_nodes, pos_edges, num_pos)

    scores = []
    labels = []

    # 正例
    for u, v in pos_edges.T:
        u, v = int(u), int(v)
        if heuristic == "CN":
            s = common_neighbors(A, u, v)
        elif heuristic == "RA":
            s = resource_allocation(A, u, v)
        else:
            raise ValueError(f"Unknown heuristic: {heuristic}")
        scores.append(s)
        labels.append(1)

    # 负例
    for u, v in neg_edges.T:
        u, v = int(u), int(v)
        if heuristic == "CN":
            s = common_neighbors(A, u, v)
        elif heuristic == "RA":
            s = resource_allocation(A, u, v)
        scores.append(s)
        labels.append(0)

    scores = np.array(scores, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    return auc, ap


# =========================
# Two-hop retrieval 评测 (CN/RA ranking)
# =========================

def eval_twohop_retrieval_heuristic(A, num_nodes, train_edges, test_edges,
                                    heuristic="CN", Ks=(10, 20), K_candidate=50):
    """
    在 two-hop retrieval 协议下评估启发式的 Prec@K, MRR, mAP_local。
    - A 使用 G_train 的邻接；
    - train_edges: E_train 正边，用于定义 G_train（这里只用来建 A_train，函数内部不再用）；
    - test_edges: E_test 正边，用于定义 per-node ground truth。
    """
    # 构造 test positives per source u
    test_pos_per_u = {}
    for u, v in test_edges.T:
        u = int(u)
        v = int(v)
        test_pos_per_u.setdefault(u, set()).add(v)

    Ks = list(Ks)
    prec_at_k = {K: [] for K in Ks}
    mrr_list = []
    ap_local_list = []

    for u, pos_vs in tqdm(test_pos_per_u.items(),
                          desc=f"two-hop retrieval ({heuristic})"):
        # two-hop candidate pool
        cand_pool = neighbors_2hop(A, u)
        cand_pool = set(cand_pool)

        # 保证包含正例 v（如果 v 不在 two-hop 内，则强行加入）
        for v in pos_vs:
            if v != u:
                cand_pool.add(v)

        if not cand_pool:
            continue

        # 若候选超过 K_candidate，随机下采样 (保留所有 positives)
        cand_pos = [v for v in pos_vs if v in cand_pool]
        others = list(cand_pool - set(cand_pos))

        if len(cand_pool) > K_candidate:
            need = K_candidate - len(cand_pos)
            if need > 0 and len(others) > 0:
                sampled = random.sample(others, min(need, len(others)))
                cand = cand_pos + sampled
            else:
                # 只有正例或者正例数已超过 K_candidate
                cand = cand_pos[:K_candidate]
        else:
            cand = list(cand_pool)

        if not cand:
            continue

        labels_u = np.array([1 if w in pos_vs else 0 for w in cand],
                            dtype=np.int64)
        if labels_u.sum() == 0:
            continue

        scores_u = []
        for w in cand:
            if heuristic == "CN":
                s = common_neighbors(A, u, w)
            elif heuristic == "RA":
                s = resource_allocation(A, u, w)
            else:
                raise ValueError(f"Unknown heuristic: {heuristic}")
            scores_u.append(s)
        scores_u = np.array(scores_u, dtype=np.float64)

        # 排序
        order = np.argsort(-scores_u)
        labels_sorted = labels_u[order]

        # Precision@K
        for K in Ks:
            k = min(K, len(labels_sorted))
            prec = labels_sorted[:k].sum() / float(k)
            prec_at_k[K].append(prec)

        # MRR
        pos_idx = np.where(labels_sorted == 1)[0]
        if len(pos_idx) > 0:
            rank1 = pos_idx[0] + 1
            mrr_list.append(1.0 / rank1)

        # local AP
        try:
            ap_local = average_precision_score(labels_u, scores_u)
            ap_local_list.append(ap_local)
        except Exception:
            pass

    def mean_safe(lst):
        return float(np.mean(lst)) if lst else None

    metrics = {f"Prec@{K}": mean_safe(prec_at_k[K]) for K in Ks}
    metrics["MRR"] = mean_safe(mrr_list)
    metrics["mAP_local"] = mean_safe(ap_local_list)
    return metrics


# =========================
# 主函数
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_path", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--candidate_K", type=int, default=50)
    parser.add_argument("--Ks", type=str, default="10,20",
                        help="comma-separated K values for Precision@K, e.g., 10,20")
    args = parser.parse_args()

    setup_seed(args.seed)
    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]

    # 读图
    edges, n = load_edge_list(args.edge_path)
    edge_index = np.array(edges, dtype=np.int64)

    # 划分 train/valid/test 正边
    split = do_edge_split(edge_index, n,
                          val_ratio=args.val_ratio,
                          test_ratio=args.test_ratio)

    train_pos = split["train"]["pos"]
    valid_pos = split["valid"]["pos"]
    test_pos = split["test"]["pos"]

    # 用 train_pos 构造 G_train 邻接
    A_train = build_adj(n, train_pos)

    print(f"Num nodes: {n}, train edges: {train_pos.shape[1]}, "
          f"val edges: {valid_pos.shape[1]}, test edges: {test_pos.shape[1]}")

    # 1) Global 协议评测启发式 (test_pos vs random neg)
    for heuristic in ["CN", "RA"]:
        auc, ap = eval_global_heuristic(A_train, n, test_pos, heuristic=heuristic)
        print(f"[Global] Heuristic={heuristic} | AUC={auc:.4f}, AP={ap:.4f}")

    # 2) Two-hop retrieval 协议评测启发式 (只用 train/test)
    for heuristic in ["CN", "RA"]:
        t0 = time.time()
        metrics = eval_twohop_retrieval_heuristic(
            A_train, n, train_pos, test_pos,
            heuristic=heuristic,
            Ks=Ks,
            K_candidate=args.candidate_K
        )
        t1 = time.time()
        print(f"[Two-hop] Heuristic={heuristic}, K_candidate={args.candidate_K}, Ks={Ks}")
        for k in Ks:
            print(f"  Prec@{k}: {metrics[f'Prec@{k}']:.4f}")
        print(f"  MRR: {metrics['MRR']:.4f} | mAP_local: {metrics['mAP_local']:.4f} | time={t1-t0:.1f}s")
