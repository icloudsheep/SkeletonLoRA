# SkeletonLoRA

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Overview

SkeletonLoRA is a **CKKS-based low-rank skeleton decryption** framework for federated LoRA aggregation. Instead of decrypting the entire aggregated update matrix $\Delta W \in \mathbb{R}^{d \times d}$, the server decrypts only a small subset of rows and columns (the "skeleton"), then reconstructs the full matrix via **CUR decomposition**:

$$\Delta W_{\text{rec}} = C_r \cdot M_r^{-1} \cdot R_r$$

where $C_r \in \mathbb{R}^{d \times r}$ are $r$ decrypted columns, $R_r \in \mathbb{R}^{r \times d}$ are $r$ decrypted rows, and $M_r \in \mathbb{R}^{r \times r}$ is their intersection block. This yields significant speedup and communication savings proportional to $d/r$ — for a 3200×3200 matrix, skeleton decryption with $r=16$ achieves **>100× speedup** while maintaining relative error below $10^{-4}$.

The framework serves as the **CKKS homomorphic encryption baseline** for the SecLoRA paper, benchmarking homomorphic aggregation of LoRA updates against the PC-MCFE cryptographic scheme.

## Key Concepts

### Low-Rank Skeleton Decryption

In federated LoRA fine-tuning, each client $i$ holds factor matrices $B_i \in \mathbb{R}^{d \times R}$ and $A_i \in \mathbb{R}^{R \times d}$ (where $R \ll d$ is the LoRA rank). The local update is $\Delta W_i = B_i A_i$, and the server aggregates:

$$\Delta W = \sum_{i=1}^{N} \Delta W_i$$

Since each $\Delta W_i$ has rank at most $R$, the aggregate has rank at most $N \cdot R$. For $N\!=\!4$ clients with $R\!=\!4$, $\text{rank}(\Delta W) \le 16 \ll 3200$. This low-rank structure means only $r$ linearly independent rows and columns are needed for exact reconstruction.

### Three Index Selection Strategies

| Strategy | Plaintext Access | Description |
|----------|:---:|------|
| **mincond** (default) | ✓ | Randomly samples 20,000 candidate $(I_r, J_r)$ pairs, picks the one minimizing $\text{cond}(M_r)$ |
| **leverage** | ✓ | Computes row/column leverage scores from the top-$r$ singular vectors of $\Delta W$ |
| **uniform** | ✗ | Evenly-spaced indices — privacy-preserving, no plaintext access needed |

### Two Pipelines

- **Pipeline A — Homomorphic Aggregation**: Each client CKKS-encrypts rows and columns of $\Delta W_i$ locally; the server adds ciphertexts homomorphically. Simulates a real federated deployment.
- **Pipeline B — Plaintext Shortcut**: Computes $\Delta W$ in plaintext first, then encrypts. Serves as a control to isolate CKKS encryption noise from homomorphic aggregation noise.

## Project Structure

```
SkeletonLoRA/
├── ckks_skeleton_test.py                          # Main experiment: 3200×3200, 4 clients
├── Low-Rank Skeleton Decryption Correctness Test.md # Formal test documentation
├── environment.yaml                                 # Conda environment (lora_fe, py3.10)
├── CKKS/
│   ├── ckks.py                                      # CKKS encrypt/decrypt helpers (TenSEAL)
│   └── gen_ckks_key.py                              # CKKS key generation
├── _scripts/
│   ├── paper_table_2_CKKS.py                        # CKKS baseline for Table 2 (hybrid A-col encryption)
│   ├── paper_table_2_CKKS_Packed.py                  # Packed CKKS variant (merged into above)
│   └── paper_table_3_CKKS.py                        # Full CKKS B×A pipeline for Table 3
├── _res/                                            # Experiment results (CSV + PNG charts)
│   ├── ckks_skeleton_results.{csv,png}              # Pipeline A vs B comparison
│   ├── ckks_strategy_comparison.{csv,png}           # mincond vs leverage vs uniform
│   ├── demo_10x4_results.{csv,png}                  # 2-client 10×4 demo
│   └── demo_float_vs_int.{csv,png}                  # CKKS encoding precision test
└── temp_output_dir/                                 # LoRA adapter weights (gitignored)
    └── client_{0..49}_output/final_lora/
```

## Installation

### Prerequisites

- Conda (Miniconda or Anaconda)
- CUDA Toolkit 12.8+ (for GPU acceleration)

### Setup

```shell
git clone https://github.com/icloudsheep/SkeletonLoRA.git
cd SkeletonLoRA

# Create and activate the conda environment
conda env create -f environment.yaml
conda activate lora_fe
```

The environment includes:
- **TenSEAL 0.3.16** — CKKS homomorphic encryption
- **PyTorch 2.9.1** — GPU-accelerated tensor operations
- **HuggingFace ecosystem** — safetensors, peft, transformers
- **charm-crypto** — pairing-based cryptography (for PC-MCFE)
- **matplotlib, seaborn** — visualization

### Generate CKKS Keys

```shell
python CKKS/gen_ckks_key.py
```

## Usage

### Main Experiment (3200×3200, 4 clients)

```shell
python ckks_skeleton_test.py
```

This runs the full homomorphic aggregation pipeline with 4 clients, loads pre-trained LoRA adapters from `temp_output_dir/`, tests skeleton ranks $r \in \{2,3,\dots,16\}$, and produces comparison charts under `_res/`.

### Demo Mode (10×10, 2 clients)

```shell
python ckks_skeleton_test.py --demo
```

A quick sanity check with randomly generated 10×4 LoRA matrices — runs in seconds without pre-trained weights.

### Float vs Integer Precision Comparison

```shell
python ckks_skeleton_test.py --demo-compare
```

Compares reconstruction error between floating-point and integer-valued matrices with identical structure, isolating CKKS encoding precision (~$2^{-40}$) from encryption noise.

### Paper Scripts

Individual benchmark scripts under `_scripts/` can be run independently:

```shell
# Table 2: CKKS hybrid encryption benchmarks (traffic + timing)
python _scripts/paper_table_2_CKKS.py

# Table 3: Full CKKS B×A pipeline simulation
python _scripts/paper_table_3_CKKS.py
```

## Results

Representative results from the main experiment (3200×3200, 4 clients):

| Metric | Full Decryption | Skeleton ($r\!=\!16$) | Improvement |
|--------|:---:|:---:|:---:|
| Decryption time | ~5 s | ~0.04 s | **>100×** |
| Download data | ~82 MB | ~800 KB | **~99%** |
| Relative error ($\varepsilon$) | $10^{-10}$ | $10^{-5}$ | within $\tau = 10^{-4}$ |

The skeleton reconstruction error drops sharply as $r$ approaches the true rank of $\Delta W$ (typically 12–16 for 4 clients with $R=4$). All three index selection strategies achieve $\varepsilon < 10^{-4}$ once $r \ge \text{rank}(\Delta W)$.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
