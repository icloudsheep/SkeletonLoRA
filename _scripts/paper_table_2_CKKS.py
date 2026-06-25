#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paper_table_2_CKKS.py
=====================
Self-contained evaluation of CKKS-based hybrid encryption for LoRA.
Supports two encryption modes:
  - "naive":  per-column CKKS encryption (one ciphertext per column vector)
  - "packed": packed CKKS encryption (flatten selected columns, chunk into
              MAX_SLOTS-sized vectors)

Measures uplink (Client → Server) and downlink (Server → Client) network
traffic at various sparsification ratios.

No external imports from CKKS/ or _she/ — fully self-contained.
"""

import os
import time
import csv
import torch
import numpy as np
from safetensors.torch import load_file
import tenseal as ts

# ---------------------------------------------------------------------------
# CKKS context helpers (self-contained, no CKKS.ckks import)
# ---------------------------------------------------------------------------

_CACHED_CTX = None
_CACHED_CTX_PATH = None


def _get_or_create_context(context_path=None):
    """Load CKKS context from file, or create a new one if not found."""
    global _CACHED_CTX, _CACHED_CTX_PATH

    if context_path is not None and _CACHED_CTX is not None and _CACHED_CTX_PATH == context_path:
        return _CACHED_CTX

    if context_path is not None and os.path.exists(context_path):
        with open(context_path, "rb") as f:
            ctx = ts.context_from(f.read())
            _CACHED_CTX = ctx
            _CACHED_CTX_PATH = context_path
            print(f"[*] Loaded CKKS context from {context_path}")
            return ctx

    print("[*] Creating new CKKS context ...")
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    _CACHED_CTX = ctx
    _CACHED_CTX_PATH = context_path
    return ctx


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_RANK = 4
MAX_SLOTS = 4096  # half of poly_modulus_degree=8192

# ---------------------------------------------------------------------------
# Core evaluation class
# ---------------------------------------------------------------------------


class CKKSTable2Evaluator:
    """Evaluate uplink/downlink overhead of CKKS hybrid encryption for LoRA.

    Parameters
    ----------
    ratio : float
        Fraction of A-matrix columns to encrypt (0 < ratio <= 1).
    mode : str
        "naive"  – encrypt each selected column as a separate CKKS vector.
        "packed" – flatten all selected columns and pack into MAX_SLOTS chunks.
    context_path : str or None
        Path to a pre-saved TenSEAL context file (if any).
    """

    def __init__(self, ratio=0.1, mode="naive", context_path=None):
        self.ratio = ratio
        self.mode = mode
        self.ctx = _get_or_create_context(context_path)

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt_naive(self, A_tensor, encrypt_indices):
        """Per-column encryption: one CKKS vector per column."""
        encrypted_cols = []
        layer_enc_size = 0
        t0 = time.time()
        for col_idx in encrypt_indices:
            col_data = A_tensor[:, col_idx].tolist()
            enc_col = ts.ckks_vector(self.ctx, col_data)
            layer_enc_size += len(enc_col.serialize())
            encrypted_cols.append((col_idx.item(), enc_col))
        t_enc = time.time() - t0
        return encrypted_cols, layer_enc_size, t_enc

    def _encrypt_packed(self, A_tensor, encrypt_indices):
        """Packed encryption: flatten selected columns, chunk to MAX_SLOTS."""
        A_enc_tensor = A_tensor[:, encrypt_indices]
        flat_data = A_enc_tensor.flatten().tolist()
        chunks = [flat_data[i:i + MAX_SLOTS] for i in range(0, len(flat_data), MAX_SLOTS)]
        layer_enc_size = 0
        t0 = time.time()
        for chunk in chunks:
            enc_vec = ts.ckks_vector(self.ctx, chunk)
            layer_enc_size += len(enc_vec.serialize())
        t_enc = time.time() - t0
        return chunks, layer_enc_size, t_enc

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(self, safetensors_path):
        print("=" * 80, flush=True)
        print(f"[*] CKKS hybrid encryption overhead evaluation", flush=True)
        print(f"[*] Mode: {self.mode.upper()}  |  Ratio: {self.ratio * 100:.2f}%  "
              f"|  SVD rank: {STANDARD_RANK}", flush=True)
        print(f"[*] Weights: {safetensors_path}", flush=True)
        print("=" * 80, flush=True)

        state_dict = load_file(safetensors_path)

        A_keys = [k for k in state_dict.keys() if 'lora_A' in k]
        pairs = [(ak, ak.replace('lora_A', 'lora_B'))
                 for ak in A_keys if ak.replace('lora_A', 'lora_B') in state_dict]

        if not pairs:
            print("[!] No LoRA A/B pairs found.")
            return None

        num_pairs = len(pairs)
        print(f"[+] {num_pairs} LoRA pairs\n", flush=True)

        # ---------- Accumulators ----------
        total_A_cols = 0
        total_A_enc_cols = 0

        total_B_plain_size = 0
        total_A_plain_size = 0
        total_A_enc_size = 0

        total_BA_plain_original_size = 0
        total_BA_plain_svd_size = 0
        total_BA_enc_size = 0

        total_enc_time = 0.0
        total_plain_mul_svd_time = 0.0
        total_enc_mul_time = 0.0

        print(f"--- Layer-by-layer processing ---", flush=True)

        for idx, (a_key, b_key) in enumerate(pairs, 1):
            A = state_dict[a_key].float()       # (rank, in_features)
            B = state_dict[b_key].float()       # (out_features, rank)

            num_cols = A.shape[1]
            k = int(num_cols * self.ratio)
            total_A_cols += num_cols
            total_A_enc_cols += k

            # ---- Column selection by L2 norm (descending) ----
            col_norms = torch.norm(A, p=2, dim=0)
            sorted_indices = torch.argsort(col_norms, descending=True)
            encrypt_indices = sorted_indices[:k]
            plain_indices = sorted_indices[k:]

            A_plain = A[:, plain_indices]

            # ---- Uplink: plaintext sizes ----
            total_B_plain_size += B.numel() * B.element_size()
            total_A_plain_size += A_plain.numel() * A_plain.element_size()

            # ---- Uplink: encrypt selected columns ----
            if self.mode == "naive":
                encrypted_cols, layer_enc_size, t_enc = self._encrypt_naive(A, encrypt_indices)
                num_ct = len(encrypt_indices)
            else:  # packed
                A_enc_tensor = A[:, encrypt_indices]
                flat_data = A_enc_tensor.flatten().tolist()
                num_ct = (len(flat_data) + MAX_SLOTS - 1) // MAX_SLOTS
                _, layer_enc_size, t_enc = self._encrypt_packed(A, encrypt_indices)
                encrypted_cols = []  # not used in packed downlink path

            total_enc_time += t_enc
            total_A_enc_size += layer_enc_size

            # ---- Server-side: plaintext B×A + SVD compression (downlink) ----
            t0 = time.time()
            BA_plain = B @ A_plain

            original_plain_vol = BA_plain.numel() * BA_plain.element_size()
            total_BA_plain_original_size += original_plain_vol

            try:
                U_k, S_k, V_k = torch.svd_lowrank(
                    BA_plain.to(torch.float32), q=STANDARD_RANK, niter=2
                )
                Vh_k = V_k.t()
                sqrt_S = torch.diag(torch.sqrt(S_k))
                # (out_features, STANDARD_RANK)  +  (STANDARD_RANK, plain_cols)
                svd_plain_vol = (
                    U_k.numel() * U_k.element_size()
                    + sqrt_S.numel() * sqrt_S.element_size()
                    + Vh_k.numel() * Vh_k.element_size()
                )
                total_BA_plain_svd_size += svd_plain_vol
            except Exception as e:
                print(f"[!] SVD on {a_key} failed: {e}")
                total_BA_plain_svd_size += original_plain_vol

            t_plain_mul_svd = time.time() - t0
            total_plain_mul_svd_time += t_plain_mul_svd

            # ---- Server-side: homomorphic enc(A) × B^T (downlink ciphertext) ----
            B_T_list = B.t().tolist()
            t0 = time.time()

            if self.mode == "naive":
                for _, enc_col in encrypted_cols:
                    enc_res = enc_col.matmul(B_T_list)
                    total_BA_enc_size += len(enc_res.serialize())
            else:
                # Packed mode uplink, but downlink matmul is still per-column
                # (the server reconstructs columns before multiplication).
                for col_idx in encrypt_indices:
                    col_data = A[:, col_idx].tolist()
                    enc_col = ts.ckks_vector(self.ctx, col_data)
                    enc_res = enc_col.matmul(B_T_list)
                    total_BA_enc_size += len(enc_res.serialize())

            t_enc_mul = time.time() - t0
            total_enc_mul_time += t_enc_mul

            # ---- Logging ----
            layer_name = a_key.replace('.lora_A.weight', '').split('.')[-1]
            print(
                f"[{idx}/{num_pairs}] {layer_name:<10} | "
                f"enc {k}/{num_cols} cols | CTs: {num_ct} | "
                f"Time(E={t_enc:.2f}s M={t_plain_mul_svd:.4f}s H={t_enc_mul:.2f}s)",
                flush=True,
            )

        # ---------- Summary ----------
        total_upload = total_B_plain_size + total_A_plain_size + total_A_enc_size
        total_download = total_BA_plain_svd_size + total_BA_enc_size
        saving = total_BA_plain_original_size - total_BA_plain_svd_size

        print(f"\n{'=' * 80}")
        print(f"--- Summary  (mode={self.mode}, ratio={self.ratio * 100:.2f}%) ---")
        print(f"  Pairs: {num_pairs}  |  "
              f"Columns: {total_A_cols} ({total_A_enc_cols} encrypted)")
        print(f"  [UPLINK   Client → Server]")
        print(f"    B plain        : {total_B_plain_size / 1024 / 1024:.2f} MB")
        print(f"    A plain        : {total_A_plain_size / 1024 / 1024:.2f} MB")
        print(f"    A cipher       : {total_A_enc_size / 1024 / 1024:.2f} MB")
        print(f"    => Total Uplink : {total_upload / 1024 / 1024:.2f} MB")
        print(f"  [DOWNLINK  Server → Client]")
        print(f"    BA plain (raw) : {total_BA_plain_original_size / 1024 / 1024:.2f} MB")
        print(f"    BA plain (SVD) : {total_BA_plain_svd_size / 1024 / 1024:.2f} MB")
        print(f"    BA cipher      : {total_BA_enc_size / 1024 / 1024:.2f} MB")
        print(f"    => Total Downlnk: {total_download / 1024 / 1024:.2f} MB  "
              f"(saved {saving / 1024 / 1024:.2f} MB)")
        print(f"  [TIME] enc={total_enc_time:.2f}s  "
              f"plain={total_plain_mul_svd_time:.4f}s  "
              f"hom={total_enc_mul_time:.2f}s")
        print(f"{'=' * 80}\n", flush=True)

        return {
            "mode": self.mode,
            "ratio": self.ratio,
            "num_pairs": num_pairs,
            "total_A_cols": total_A_cols,
            "total_A_enc_cols": total_A_enc_cols,
            "B_plain_bytes": total_B_plain_size,
            "A_plain_bytes": total_A_plain_size,
            "A_enc_bytes": total_A_enc_size,
            "total_upload_bytes": total_upload,
            "BA_plain_orig_bytes": total_BA_plain_original_size,
            "BA_plain_svd_bytes": total_BA_plain_svd_size,
            "BA_enc_bytes": total_BA_enc_size,
            "total_download_bytes": total_download,
            "download_saved_bytes": saving,
            "enc_time_s": total_enc_time,
            "plain_mul_svd_time_s": total_plain_mul_svd_time,
            "enc_mul_time_s": total_enc_mul_time,
        }


# ---------------------------------------------------------------------------
# Main – batch evaluation across ratios + modes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    safetensors_file = './temp_output_dir/client_0_output/final_lora/adapter_model.safetensors'
    context_file = './CKKS/ckks_full_context.bytes'

    test_ratios = [
        0.00125, 0.0025, 0.005, 0.0075, 0.01,
        0.02, 0.03, 0.04, 0.05, 0.08,
        0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
    ]

    modes = ["naive", "packed"]
    csv_file = "CKKS_Table2_Results.csv"

    all_results = []

    for mode in modes:
        for ratio in test_ratios:
            print(f"\n{'#' * 80}")
            print(f"#  MODE={mode.upper()}  RATIO={ratio * 100:.2f}%")
            print(f"{'#' * 80}")
            evaluator = CKKSTable2Evaluator(
                ratio=ratio, mode=mode, context_path=context_file
            )
            res = evaluator.process(safetensors_file)
            if res is not None:
                all_results.append(res)

    # Write CSV
    if all_results:
        fieldnames = [
            "mode", "ratio", "num_pairs",
            "total_A_cols", "total_A_enc_cols",
            "B_plain_bytes", "A_plain_bytes", "A_enc_bytes", "total_upload_bytes",
            "BA_plain_orig_bytes", "BA_plain_svd_bytes", "BA_enc_bytes",
            "total_download_bytes", "download_saved_bytes",
            "enc_time_s", "plain_mul_svd_time_s", "enc_mul_time_s",
        ]
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"\n[*] All results saved to {csv_file}")
    else:
        print("[!] No results generated.")
