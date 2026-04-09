# NST: Negative Sampling Tuning for Two-hop Retrieval in SEAL

This repository contains the code used in the paper:

> Dawei Liu, **Beyond Random Negatives: Structural Negative Sampling and Two-hop Retrieval for Robust Link Prediction**.

The code implements:
- SEAL-style subgraph-based link prediction;
- Two-hop retrieval evaluation (per-node candidate sets and retrieval metrics);
- Structural negative sampling schemes mixing global random and two-hop negatives;
- Negative Sampling Tuning (NST), a simple per-graph procedure to select the negative sampling configuration;
- Baseline heuristic evaluators (CN/RA) under global and two-hop retrieval protocols.

---

## 1. Environment

The code is based on Python and PyTorch Geometric. A typical environment includes:

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- PyTorch Geometric (matching your PyTorch/CUDA version)  
- NumPy, SciPy, scikit-learn, tqdm

You can install the main dependencies with:

```bash
pip install torch torchvision torchaudio  # choose versions per your CUDA
pip install torch-geometric               # see https://pytorch-geometric.readthedocs.io
pip install numpy scipy scikit-learn tqdm
```

## 2. Data format
Each graph is stored as an undirected edge list in a text file, one edge per line:
```text
u v
u' v'
```

The paper uses eight standard benchmark graphs:

USAir, Celegans, Power, Router, NS, PB, Yeast, Ecoli

and the corresponding edge-list files are expected in data/ as:

data/USAir.txt
data/Celegans.txt
data/Power.txt
data/Router.txt
data/NS.txt
data/PB.txt
data/Yeast.txt
data/ecoli.txt

(These files can be constructed from publicly available sources cited in the paper.)

## 3. Main scripts
### 3.1 Training SEAL with structural negative sampling
File: seal_train_twohop_neg.py

This script trains a SEAL model under a given structural negative sampling configuration (n_rand_per_pos, n_twohop_per_pos):

- For each positive training edge (u, v), it samples:
  - n_rand_per_pos global random negatives (u, w),
  - n_twohop_per_pos two-hop structural negatives (u, w), where w lies in the 1/2-hop neighborhood of u in the training graph but is not an edge.
- It then extracts k-hop enclosing subgraphs, applies DRNL labeling, and trains a DGCNN encoder as in SEAL.
- A checkpoint is saved for later retrieval evaluation.
Example: train SEAL on USAir with (n_rand_per_pos, n_twohop_per_pos) = (1, 1), k = 1:

python seal_train_twohop_neg.py
--edge_path data/USAir.txt
--num_hops 1
--epochs 30
--batch_size 32
--hidden 32
--layers 3
--max_z 1000
--k_sort 30
--val_ratio 0.05
--test_ratio 0.10
--seed 12345
--n_rand_per_pos 1
--n_twohop_per_pos 1
--model_dir models_twohop/USAir
--model_name USAir_seal_twohop_nr1_nt1

A checkpoint will be saved to:
models_twohop/USAir/USAir_seal_twohop_nr1_nt1.pt

### 3.2 Two-hop retrieval evaluation
File: seal_retrieval.py

This script implements the two-hop retrieval protocol and evaluation metrics:

- Split edges into train/valid/test positives;
- Build the training graph G_train from train positives;
- Construct candidate sets per source node using a specified candidate strategy:
  - two_hop: nodes in the 1/2-hop neighborhood of the source in G_train,
  - random_k: K random candidates,
  - cn_topk: top-K by common neighbors score;
- Extract enclosing subgraphs for all candidate pairs (u, w) and score them with a SEAL model;
- Compute Precision@K, MRR, and mAP_local under the retrieval protocol;
- Optionally compute global AUC/AP with random negatives.
You can either:
- Let seal_retrieval.py train a random-negative model internally (epochs > 0, no --load_model), or
- Load a pre-trained checkpoint (trained with structural negatives) via --load_model.
Example: evaluate a pre-trained SEAL model under two-hop retrieval on USAir, K = 50:

python seal_retrieval.py
--edge_path data/USAir.txt
--num_hops 1
--epochs 0
--batch_size 32
--hidden 32
--layers 3
--max_z 1000
--k_sort 30
--val_ratio 0.05
--test_ratio 0.10
--seed 12345
--retrieval_K 50
--candidate_strategy two_hop
--load_model models_twohop/USAir/USAir_seal_twohop_nr1_nt1.pt

The script prints a line like:

Retrieval metrics: {'Prec@10': ..., 'Prec@20': ..., 'MRR': ..., 'mAP_local': ...}

### 3.3 Sweeping negative sampling configurations (NST grid)
File: sweep_twohop_train_retrieval2.py

This script automates the grid search over negative sampling configurations (the NST search space) and datasets:

- DATASETS: list of (dataset_name, edge_path) pairs, e.g., the 8 benchmark graphs.
- NEG_SAMPLING_CONFIGS: the grid of (tag, n_rand_per_pos, n_twohop_per_pos), e.g., (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)
- RETRIEVAL_K_LIST: list of candidate sizes K (e.g., 20, 50, 100).
For each dataset, each configuration, and each K, it:

- Ensures the corresponding two-hop-aware SEAL model exists (calls seal_train_twohop_neg.py if needed);
- Evaluates it under the two-hop retrieval protocol (calls seal_retrieval.py);
- Writes a row to the CSV file sweep_twohop_train_retrieval_results2.csv.
Example:

python sweep_twohop_train_retrieval2.py

The resulting CSV is used to construct:

Per-graph negative sampling grids (e.g., USAir, NS);
NST-selected configurations per graph (the best validation mAP_local or MRR).

### 3.4 Heuristic baselines (CN/RA)
File: heuristic_retrieval.py

This script evaluates classical heuristics (common neighbors and resource allocation) under:
- Global protocol: AUC and AP against random negatives;
- Two-hop retrieval protocol: Precision@K, MRR, and mAP_local with two-hop candidate sets.
Example:

python heuristic_retrieval.py
--edge_path data/USAir.txt
--val_ratio 0.05
--test_ratio 0.10
--seed 12345
--candidate_K 50
--Ks 10,20

This prints:
- Global AUC/AP for CN and RA; and
- Two-hop retrieval metrics (Prec@K, MRR, mAP_local) at the specified Ks.

# 4. Reproducing the main experiments
High-level workflow to reproduce the main results in the paper:

1. Prepare data
- Place the edge list files for USAir, Celegans, Power, Router, NS, PB, Yeast, Ecoli in the data/ directory.
- Make sure file names match those referenced in DATASETS.
2. Train and evaluate SEAL with structural negatives
- For a single configuration on a single dataset, call:
  - seal_train_twohop_neg.py (train)
  - seal_retrieval.py (two-hop retrieval evaluation)
- For the full NST grid across configurations and datasets, run:
  - sweep_twohop_train_retrieval2.py
3. Run heuristic baselines
- Use heuristic_retrieval.py to obtain CN/RA global and two-hop retrieval metrics.
4. Analyze CSV outputs
- Use the sweep CSV files to:
  - Build negative sampling performance grids,
  - Determine NST-selected configurations per graph,
  - Compare global vs two-hop performance, and structured vs random negatives.
