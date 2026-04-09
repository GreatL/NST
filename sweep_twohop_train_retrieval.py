import subprocess
import csv
import os
import re
import time

# 配置区：根据你的实际路径和命令行需要调整

TRAIN_SCRIPT = "seal_train_twohop_neg.py"
RETRIEVAL_SCRIPT = "seal_retrieval.py"

DATASETS = [
    # (name, edge_path)
    #("USAir", "data/USAir.txt"),
    #("Celegans", "data/Celegans.txt"),
    #("Power", "data/Power.txt"),
    #("Router", "data/Router.txt"),
    #("NS", "data/NS.txt"),
    #("PB", "data/PB.txt"),
    #("Yeast", "data/Yeast.txt"),
    #("ecoli", "data/ecoli.txt"),
    #("ADV", "data/ADV.txt"),
    #("BUP", "data/BUP.txt"),
    #("CDM", "data/CDM.txt"),
    #("CGS", "data/CGS.txt"),        
    #("EML", "data/EML.txt"),
    #("ERD", "data/ERD.txt"),
    ###("FBK", "data/FBK.txt"),
    ("GRQ", "data/GRQ.txt"),
    ("HMT", "data/HMT.txt"),
    ("HPD", "data/HPD.txt"),
    ("HTC", "data/HTC.txt"),
    ("INF", "data/INF.txt"),
    ("KHN", "data/KHN.txt"),
    ("LDG", "data/LDG.txt"),
    ("NSC", "data/NSC.txt"),
    ("PGP", "data/PGP.txt"),
    ("SMG", "data/SMG.txt"),
    ("YST", "data/YST.txt"),
    ("ZWL", "data/ZWL.txt"),
]

# 训练负采样配置：
# 对于 baseline random 训练，你可以在这里用一个特殊标记，比如 ("random", 1, 0)
# 并在 TRAIN 逻辑中对它做特殊处理（或直接用 seal_retrieval 的内部 random 训练作为 baseline）。
NEG_SAMPLING_CONFIGS = [
    # tag, n_rand_per_pos, n_twohop_per_pos
    
    ("nr1_nt0", 1, 0),  # N_neg=1, pure random, baseline: 只 random 负采样 (通过 seal_retrieval.py 训练)
    ("nr0_nt1", 0, 1),  # N_neg=1, pure twohop

    ("nr2_nt0", 2, 0),  # N_neg=2, pure random
    ("nr1_nt1", 1, 1),  # N_neg=2, 50/50 two-hop-aware
    ("nr0_nt2", 0, 2),  # N_neg=2, pure twohop

    ("nr3_nt0", 3, 0),  # N_neg=3, pure random
    ("nr2_nt1", 2, 1),  # N_neg=3, twohop-light
    ("nr1_nt2", 1, 2),  # N_neg=3, twohop-heavy
    ("nr0_nt3", 0, 3),  # N_neg=3, pure twohop
    
]

RETRIEVAL_K_LIST = [20,50,100]

# DGCNN 超参（必须与训练/评测脚本保持一致）
HIDDEN = 32
LAYERS = 3
MAX_Z = 1000
K_SORT = 30
NUM_HOPS = 1
EPOCHS = 30
BATCH_SIZE = 32
VAL_RATIO = 0.05
TEST_RATIO = 0.10
SEED = 12345


def run_cmd(cmd):
    """
    简单封装 subprocess.run，打印命令和输出。
    """
    print("\n[CMD]", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t1 = time.time()
    print(proc.stdout)
    print(f"[CMD] Done in {t1 - t0:.1f} s")
    return proc.stdout


def ensure_twohop_model(dataset_name, edge_path, tag, nr, nt, model_dir):
    """
    如果 tag != "random"，确保 two-hop-aware 模型存在；
    返回 checkpoint 路径。
    """
    if tag == "random":
        # random baseline 模型不一定需要用 twohop 训练，可以直接用 retrieval 内部训练，
        # 此时我们让 sweep 逻辑在调用 retrieval 时不加 --load_model，让其自己训练。
        return None

    model_name = f"{dataset_name}_seal_twohop_{tag}"
    ckpt_path = os.path.join(model_dir, f"{model_name}.pt")
    if os.path.exists(ckpt_path):
        print(f"[INFO] Found existing model: {ckpt_path}")
        return ckpt_path

    # 调用 twohop 训练脚本
    cmd = [
        "python", TRAIN_SCRIPT,
        "--edge_path", edge_path,
        "--num_hops", str(NUM_HOPS),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--hidden", str(HIDDEN),
        "--layers", str(LAYERS),
        "--max_z", str(MAX_Z),
        "--k_sort", str(K_SORT),
        "--val_ratio", str(VAL_RATIO),
        "--test_ratio", str(TEST_RATIO),
        "--seed", str(SEED),
        "--n_rand_per_pos", str(nr),
        "--n_twohop_per_pos", str(nt),
        "--model_dir", model_dir,
        "--model_name", model_name,
    ]
    run_cmd(cmd)
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Model not found after training: {ckpt_path}")
        return None
    return ckpt_path


def parse_retrieval_metrics(output_text):
    """
    从 seal_retrieval.py 的输出文本中解析 Retrieval metrics 行。
    预期格式类似：
      Retrieval metrics: {'Prec@10': 0.0681, 'Prec@20': 0.0578, 'MRR': 0.2573, 'mAP_local': 0.1935}
    """
    # 找到包含 "Retrieval metrics:" 的行
    lines = output_text.splitlines()
    metrics_line = None
    for line in lines:
        if "Retrieval metrics:" in line:
            metrics_line = line
            break
    if metrics_line is None:
        print("[WARN] No 'Retrieval metrics' line found!")
        return None

    # 提取大括号中的内容
    m = re.search(r"\{.*\}", metrics_line)
    if not m:
        print("[WARN] Unable to parse metrics dict from line:", metrics_line)
        return None

    metrics_str = m.group(0)
    # 使用 eval / literal_eval 解析为 dict
    try:
        metrics = eval(metrics_str, {"__builtins__": {}})
    except Exception as e:
        print("[WARN] eval metrics failed:", e)
        return None

    # 期待 keys: 'Prec@10', 'Prec@20', 'MRR', 'mAP_local'
    return metrics


def run_retrieval_eval(edge_path, candidate_strategy, K, ckpt_path=None, runs=1):
    """
    调用 seal_retrieval.py 做 retrieval 评测：
    - 若 ckpt_path 为 None，则用内部 random 训练；
    - 若 ckpt_path 非 None，则跳过训练，加载 checkpoint。
    返回 metrics 平均值（跨 runs）。
    """
    prec10_list, prec20_list, mrr_list, map_list = [], [], [], []

    for r in range(runs):
        cmd = [
            "python", RETRIEVAL_SCRIPT,
            "--edge_path", edge_path,
            "--num_hops", str(NUM_HOPS),
            "--epochs", "30" if ckpt_path is None else "0",
            "--batch_size", str(BATCH_SIZE),
            "--hidden", str(HIDDEN),
            "--layers", str(LAYERS),
            "--max_z", str(MAX_Z),
            "--k_sort", str(K_SORT),
            "--val_ratio", str(VAL_RATIO),
            "--test_ratio", str(TEST_RATIO),
            "--seed", str(SEED + r),
            "--retrieval_K", str(K),
            "--candidate_strategy", candidate_strategy,
        ]
        if ckpt_path is not None:
            cmd.extend(["--load_model", ckpt_path])

        out = run_cmd(cmd)
        metrics = parse_retrieval_metrics(out)
        if metrics is None:
            continue

        prec10_list.append(metrics.get("Prec@10", None))
        prec20_list.append(metrics.get("Prec@20", None))
        mrr_list.append(metrics.get("MRR", None))
        map_list.append(metrics.get("mAP_local", None))

    def _mean(lst):
        lst = [x for x in lst if x is not None]
        return sum(lst) / len(lst) if lst else None

    return {
        "Prec@10": _mean(prec10_list),
        "Prec@20": _mean(prec20_list),
        "MRR": _mean(mrr_list),
        "mAP_local": _mean(map_list),
    }


if __name__ == "__main__":
    # 输出 CSV 文件
    out_csv = "sweep_twohop_train_retrieval_results.csv"
    os.makedirs("sweep_logs", exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "neg_sampling_tag",
            "n_rand_per_pos",
            "n_twohop_per_pos",
            "retrieval_K",
            "candidate_strategy",
            "Prec@10",
            "Prec@20",
            "MRR",
            "mAP_local",
        ])

        for dataset_name, edge_path in DATASETS:
            model_dir = os.path.join("models_twohop", dataset_name)
            os.makedirs(model_dir, exist_ok=True)

            for tag, nr, nt in NEG_SAMPLING_CONFIGS:
                # 准备模型
                if tag == "random":
                    ckpt_path = None  # random baseline 使用 retrieval 内部训练
                else:
                    ckpt_path = ensure_twohop_model(dataset_name, edge_path, tag, nr, nt, model_dir)

                for K in RETRIEVAL_K_LIST:
                    print(f"\n=== Dataset={dataset_name}, neg_tag={tag}, (nr,nt)=({nr},{nt}), K={K} ===")
                    metrics = run_retrieval_eval(
                        edge_path=edge_path,
                        candidate_strategy="two_hop",
                        K=K,
                        ckpt_path=ckpt_path,
                        runs=1,  # 可以设为>1取平均
                    )
                    print("[RESULT]", metrics)

                    writer.writerow([
                        dataset_name,
                        tag,
                        nr,
                        nt,
                        K,
                        "two_hop",
                        metrics.get("Prec@10"),
                        metrics.get("Prec@20"),
                        metrics.get("MRR"),
                        metrics.get("mAP_local"),
                    ])
                    f.flush()

    print(f"\nAll done. Results saved to {out_csv}")
