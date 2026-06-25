
# SecLoRA

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Overview

This repository provides the implementation of the paper SecLoRA: Secure Aggregation of Low-Rank Matrix Products via Functional Encryption, the first decentralized framework achieving exact aggregation of LoRA updates with linear communication complexity. The core of SecLoRA is a novel cryptographic primitive: Pairwise Composable Multi-Client Functional Encryption (PC-MCFE). The core of SecLoRA is a novel cryptographic primitive: Pairwise Composable Multi-Client Functional Encryption (PC-MCFE). Unlike traditional functional encryption, which treats ciphertext recombination as an attack, the dual-encryption architecture of PC-MCFE ($\mathsf{Enc}_A, \mathsf{Enc}_B$) is intentionally designed to harness this property. It allows any ciphertexts $\mathsf{ct}_A$ and $\mathsf{ct}_B$ to be arbitrarily paired and evaluated via a functional key to reveal their inner product. This unique property enables secure and decentralized aggregation of matrix products without losing LoRA's linear communication advantages. Furthermore, SecLoRA ensures round-isolated decryption to prevent temporal leakage without extra interaction. Evaluation shows that SecLoRA is practical for cross-silo deployments.


<table>
  <tr>
    <td width="100%"><img src="_res/Comparison_of_Privacy_of_Preserving_Aggregation_Paradigms.png" alt="Comparison Overview"></td>
  </tr>
<tr>
    <td width="100%">Comparison of different paradigm</td>
  </tr>
  <tr>
    <td width="100%"><img src="_res/Client_Network_Traffic_Comparison.png" alt="Network Comparison"></td>
  </tr>
  <tr>
    <td width="100%">Network Traffic of SecLoRA & Other Methods (Lower is Better)</td>
  </tr>
  <tr>
    <td width="100%"><img src="_res/overlap.png" alt="Encrytion Overlap Comparison">
      <embed>
    </td>
  </tr>
  <tr>
    <td width="100%">Encryption Overlap of SecLoRA & Other Methods (Higher is Better)</td>
  </tr>
</table>

## Introduction

SecLoRA consists of the following three key components:

### 1. Parameter Sensitivity Calculation

To avoid the substantial computational cost of evaluating loss variations with forward passes, we use the channel sensitivity at the structural level using the weight magnitudes. Specifically, for the LoRA matrix $\mathbf{A}$, we evaluate the importance of each column (corresponding to input features); for the LoRA matrix $\mathbf{B}$, we evaluate the importance of each row (corresponding to output features). Let $v$ represent a specific column vector in matrix $\mathbf{A}$ or a row vector in matrix $\mathbf{B}$. Its sensitivity $\Omega(v)$ is defined by its L2-norm: 

 
$\Omega(v) = \lVert v \rVert_2 = \sqrt{\sum_{i=1}^{R} v_i^2}  $

where $R$ is the intrinsic rank of the LoRA layers. The rows or columns with the lowest sensitivity scores $\Omega(v)$ are considered insignificant and are deterministically pruned (zeroed out).

Relevant code:

- [`client/sensitivity.py`](./client/sensitivity.py): Client-side tool for row and column sensitivity calculation, pruning, and statistical tracking. It computes the L2-norm for vectors in LoRA A/B matrices, zeros out the lowest sensitivity features based on a specified threshold, and logs the resulting sparsity without requiring client-server negotiation.



### 2. Zero Encryption Optimization

After sensitivity pruning, a large fraction of row/column vectors become all-zero. PC-MCFE exploits this sparsity via a three-layer zero-skip architecture:

**Layer 1 — Structural Sparsity.** The pruning threshold grows progressively across rounds (configured in [`ENCRYPTION_THRESHOLD(round_idx)`](utils/constants.py:45)), zeroing up to ~85% of vectors pre-encryption.

**Layer 2 — Client-Side Zero Skip.** In [`PC_MCFE_Client.encrypt()`](MCFE/mcfe.py:235), zero vectors skip the expensive iFE operations (each $O(R)$ exponentiations in $G_1$ or $G_2$) and store only a lightweight `is_zero` flag plus a DSum mask. For [`keygen()`](MCFE/mcfe.py:271), zero vectors only generate a lightweight non-hiding key: 

> if $\mathbf{b}_i = 0$:
>
> $$
> ct_{\ell_b,i,b}^{(q)} := \langle (0^{R_i}, 0^{R_i}, t_{\ell_b,q}, 0), \mathbf{s}_1 \rangle \in \mathbb{Z}_p
> $$
>
> with an `is_zero` flag, where $\mathbf{s}_1$ is the first $R_i$ element of $ek_i^{(q)}$.

**Layer 3 — Server-Side Zero Skip.** In [`PC_MCFE_Server.decrypt_and_aggregate()`](MCFE/mcfe.py:337), cells where either $\mathsf{ct}_v$ or $\mathsf{sk}_u$ is zero bypass the full pairing (each costs $2R+4 \approx 12$ pairings at $R=4$), yielding quadratic speedup as sparsity increases.

> **A special case:**
> If $ct_{\ell_b,i,b}^{(q)}$ is a zero encryption, and $ct_{\ell_a,i,a}^{(q)} = (c_1, c_2[0:2R_i+2])$ is not a zero encryption:
>
> $$
> \mathsf{Dec}(ct_{\ell_a,i,a}^{(q)}, ct_{\ell_b,i,b}^{(q)}) = \langle (0^{R_i}, 0^{R_i}, t_{\ell_b,q}, 0), c_2[1:2R_i+2] \rangle + c_2[0] \cdot ct_{\ell_b,i,b}^{(q)}
> $$

**Communication Savings.** A zero ciphertext stores only a 1-byte flag + DSum mask (~30 bytes), versus ~400 bytes for a non-zero ciphertext with MNT224 group elements — a $>10\times$ reduction per entry. The cumulative speedup for aggregation (pairing) is $\approx 1/(1-\tau)^2$, reaching $\sim 44\times$ at $\tau=0.85$.

Relevant code:
- [`client/sensitivity.py`](client/sensitivity.py): L2-norm sensitivity pruning
- [`MCFE/mcfe.py`](MCFE/mcfe.py): `_encrypt_zero()` / `_keygen_zero()` (lines 228–233), server-side `is_zero` gating (line 363)
- [`utils/constants.py`](utils/constants.py): `ENCRYPTION_THRESHOLD(round_idx)` sparsity schedule

### 3. PC-MCFE: Client Encryption & Server Aggregation

SecLoRA's core is **PC-MCFE**, a dual-encryption scheme securely computing inner-product aggregation of LoRA matrices without exposing individual updates.

#### Client-Side Encryption

Each client $i$ holds LoRA matrices $\mathbf{A}_i \in \mathbb{Z}^{R \times d_\text{in}}$ and $\mathbf{B}_i \in \mathbb{Z}^{d_\text{out} \times R}$, quantized by factor $F=1000$.

**Encrypt $\mathbf{A}_i$ (column-wise).** For column $v$, an extended vector $\hat{\mathbf{a}}_v \in \mathbb{Z}_p^{2R+2}$ is constructed and encrypted via the iFE core:

$$\hat{\mathbf{a}}_v = [\mathbf{a}_v \;\|\; \mathbf{0}_R \;\|\; x_i \;\|\; 0], \quad x_i \xleftarrow{\$} \mathbb{Z}_p$$

The ciphertext is $\mathsf{ct}_v$ = $(c_1, \mathbf{c}_2)$, with a DSum mask computed in $\mathbb{Z}_p$: $\mathsf{dsum}_v = x_i + \sum_{j \neq i} \mathsf{PRF}_{s_{ij}}(\mathsf{label})$.

**Generate key for $\mathbf{B}_i$ (row-wise).** For row $u$, the client constructs $\hat{\mathbf{b}}_u = [\mathbf{b}_u \;\|\; \mathbf{0}_R \;\|\; t_{\mathsf{label}} \;\|\; 0] \in \mathbb{Z}_p^{2R+2}$ and generates $\mathsf{sk}_u = (k_1, \mathbf{k}_2)$ via the iFE core. Zero vectors skip the full iFE operation entirely.

#### Server-Side Aggregation

The server homomorphically pairs all clients' ciphertexts and keys without decryption. For each cell $(u, v)$:

$$\mathsf{gt}_{u,v} = \prod_{i=1}^{N} \mathsf{Pair}\bigl(\mathsf{sk}_u^{(i)}, \mathsf{ct}_v^{(i)}\bigr) \cdot g_T^{-t_{\mathsf{label}} \cdot \sum_i \mathsf{dsum}_v^{(i)}} = g_T^{\sum_{i=1}^{N} \langle \mathbf{b}_u^{(i)}, \mathbf{a}_v^{(i)} \rangle}$$

The inner product is recovered via BSGS (baby-step giant-step) in $G_T$: $\Delta\mathbf{W}[u,v] = \mathrm{BSGS}(g_T, \mathsf{gt}_{u,v}, D_{\max})$, then dequantized by $\frac{1}{F^2 \cdot N}$. The server learns only the aggregated sum — individual matrices remain hidden under the DDH assumption.

Relevant code:
- [`MCFE/mcfe.py`](MCFE/mcfe.py): Core iFE, DSum, PC-MCFE client/server primitives
- [`client/mcfe_client.py`](client/mcfe_client.py): Client-side encryption orchestration
- [`server/mcfe_server.py`](server/mcfe_server.py): Server-side aggregation & binary protocol


## Data Flow

The following diagram illustrates the complete PC-MCFE workflow orchestrated by `main.py`:

```
main.py
├─ MCFEServerContext(n_clients, STANDARD_RANK)
│   └─ SystemCoordinator.global_setup() → PKI, DSum keys
│
├─ ClientPart(mcfe_context, ...)
│   ├─ context_blob = mcfe_context.serialize_client_context(cid)  ← bytes
│   └─ client.encrypt_adapters(final_path, context_blob=blob)
│       └─ encrypt_with_pc_mcfe(...)  → client_{id}_pc_mcfe.bin
│
└─ ServerPart(mcfe_context, ...)
    └─ server.aggregate_mi_dmcfe()
        └─ mcfe_context.aggregate(round_idx, round_prefix)
            ├─ parse client_{id}_pc_mcfe.bin
            ├─ PC_MCFE_Server.decrypt_and_aggregate(...)
            └─ save server_aggregated_mi_dmcfe_delta_w.pth
```

**Key modules:**

| Module | File | Role |
|--------|------|------|
| `MCFEServerContext` | `server/mcfe_server.py` | PKI setup, context blob serialization, binary parsing, PC-MCFE aggregation |
| `encrypt_with_pc_mcfe` | `client/mcfe_client.py` | Deserializes context blob, reconstructs `PC_MCFE_Client`, encrypts LoRA A + keygen B |
| `Server` | `server/server.py` | Thin orchestration layer, delegates `aggregate_mi_dmcfe()` to `MCFEServerContext` |
| `Client` | `client/client.py` | LoRA training, pruning, SVD residuals, calls `encrypt_with_pc_mcfe()` with context blob |

## How to use

### 1. Deployment

#### Clone the github repository

```shell
git clone https://github.com/....XXXX
cd SecLoRA
```

#### Create running environment

Then, you can run `chmod +x deployment.sh && deployment.sh` to deploy the relevant runtime environment with one click. If you encounter any issues, you can also try manually entering the commands below.

```shell
conda env create -f environment.yml
conda activate sec-lora

git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### 2. Run SecLoRA

#### Federated Learning

##### By Shell (Recommanded)

This approach will save log to `./run_log` so that we can debug easily.

```shell
chmod +x ./run.sh && ./run.sh
```

##### By Python

```shell
python main.py
```

#### Evaluation

```shell
chmod +x ./eval.sh && ./eval.sh ./models/open_llama_3b_v2 ./temp_output_dir/client_output/mi_dmcfe_model
```

## Benchmark

Some of the data and images in the paper can be run with a single click using the script below:

```shell
chmod +x ./run_all_scripts.sh && ./run_all_scripts.sh
```

All paper experiment scripts under [`_scripts/`](_scripts/) can be run at once via [`run_all_scripts.sh`](run_all_scripts.sh).

- [`paper_figure_3.py`](_scripts/paper_figure_3.py) — Reads per-round averaged final loss from `experiment_logs/loss_final_*.csv` and plots the loss curve (Figure 3).
- [`paper_figure_4_5.py`](_scripts/paper_figure_4_5.py) — Extracts LoRA A/B from client safetensors, selects top rows/cols by L2 norm, and renders a heatmap of cross-client selection overlap (Figures 4 & 5).
- [`paper_figure_6.py`](_scripts/paper_figure_6.py) — Compares SecLoRA vs. SHE-LoRA in terms of the fraction of encrypted elements as the number of clients grows (Figure 6).
- [`paper_table_2.py`](_scripts/paper_table_2.py) — Orchestrates Table 2: iterates over sparsity ratios, invokes the MCFE evaluator, and writes per-stage timing & traffic to `NetworkTrafficTest_Results.csv`.
- [`paper_table_2_MCFE.py`](_scripts/paper_table_2_MCFE.py) — SecLoRA (MCFE) evaluator for Table 2. Extracts LoRA matrices, applies L2-norm sparsification, computes residuals, runs PC-MCFE encryption, and performs SVD compression on residuals.
- [`paper_table_2_CKKS.py`](_scripts/paper_table_2_CKKS.py) — CKKS baseline evaluator for Table 2. Encrypts the most important A columns via CKKS, sends the rest in plaintext, and measures upload/download traffic and encryption time.
- [`paper_table_2_CKKS_Packed.py`](_scripts/paper_table_2_CKKS_Packed.py) — CKKS packed variant for Table 2. Packs multiple columns into a single ciphertext via SIMD; also accounts for one-time Galois/Relin key upload.
- [`paper_table_3_CKKS.py`](_scripts/paper_table_3_CKKS.py) — Full CKKS pipeline simulation for Table 3: client-side CKKS encryption, network transfer, server-side parallel homomorphic B×A multiplication + aggregation, and client-side decryption.
- [`paper_table_3_MCFE.py`](_scripts/paper_table_3_MCFE.py) — Full SecLoRA (MCFE) pipeline simulation for Table 3. Client-side extraction, sparsification, PC-MCFE encryption, and SVD residual compression; server-side binary parsing, pairing-based aggregation, BSGS discrete-log decryption, residual merging, and final SVD reconstruction.

