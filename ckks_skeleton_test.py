#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CKKS Skeleton Decryption — Complete Homomorphic Aggregation Pipeline
=====================================================================
[审查 #5] 本文件是单进程密码学原型。所有组件（客户端加密、服务端同态聚合、
解密验证）在同一 Python 进程中顺序执行，不存在网络隔离或进程隔离。

两条流水线：
  A. 同态聚合 — 每个客户端本地加密 ΔW_i → 服务端 CKKS 加法聚合 → 解密对比
  B. 明文捷径 — 明文聚合后加密 ΔW → 解密对比（纯验证骨架公式，隔离同态噪声）

三种索引选择策略（均在本文件中实现，可直接对比）：
  - mincond: 随机采样 + 条件数最小化（默认，需明文 ΔW）
  - leverage: 基于 SVD leverage scores 的确定性选择（基线，需明文 ΔW）
  - uniform:  均匀间隔采样（安全，无需明文 ΔW）

用法:
  python ckks_skeleton_test.py          # 3200×3200 主实验
  python ckks_skeleton_test.py --demo   # 2 客户端 10×4 演示

环境: conda skeleton_lora_fe (py3.10 + tenseal + safetensors + numpy + matplotlib)
"""

import argparse
import numpy as np
import tenseal as ts
from safetensors import safe_open
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time, os, sys, csv

# ═══════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════
TEMP_OUTPUT_DIR = "./temp_output_dir"
N_CLIENTS = 4
RANK = 4
N_DIM = 3200
R_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE = 2 ** 40
TAU = 1e-4
RES_DIR = "./_res"
os.makedirs(RES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# ── 工具函数 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def load_weights(n_clients, layer="q_proj", dim=None):
    """返回 (B_list, A_list)。每项 shape 为 (dim,4) 和 (4,dim)。"""
    Bs, As_ = [], []
    for cid in range(n_clients):
        p = os.path.join(TEMP_OUTPUT_DIR,
                         f"client_{cid}_output/final_lora/adapter_model.safetensors")
        with safe_open(p, framework="np") as f:
            bk, ak = None, None
            for k in f.keys():
                if layer in k:
                    if "lora_B" in k: bk = k
                    elif "lora_A" in k: ak = k
            Bs.append(f.get_tensor(bk))
            As_.append(f.get_tensor(ak))
    if dim is not None:
        Bs = [b[:dim, :] for b in Bs]
        As_ = [a[:, :dim] for a in As_]
    return Bs, As_


def relative_error(rec, ref):
    d = np.linalg.norm(rec - ref, "fro")
    n = np.linalg.norm(ref, "fro")
    return float(d / n) if n > 0 else float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# ── 三种索引选择策略 ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def select_indices_mincond(delta_w, r, n_trials=20000):
    """策略 1：随机采样 + 条件数最小化（默认）—— 需明文 ΔW。"""
    m, n = delta_w.shape
    best_cond = float("inf")
    best = (None, None)
    rng = np.random.RandomState(42)
    for _ in range(n_trials):
        Ic = np.sort(rng.choice(m, r, replace=False))
        Jc = np.sort(rng.choice(n, r, replace=False))
        s = np.linalg.svd(delta_w[np.ix_(Ic, Jc)], compute_uv=False)
        c = s[0] / s[-1] if s[-1] > 1e-15 else float("inf")
        if c < best_cond:
            best_cond = c
            best = (Ic, Jc)
    return best[0], best[1], best_cond


def select_indices_leverage(delta_w, r):
    """策略 2：SVD leverage scores 基线 —— 需明文 ΔW。

    对 ΔW 做完整 SVD，取 top-r 奇异向量计算 leverage scores，
    选得分最高的 r 行和 r 列。比随机采样快但条件数通常更差。
    [审查 #2] 此函数为新增，用于与 mincond 策略直接对比。
    """
    U, S, Vt = np.linalg.svd(delta_w, full_matrices=False)
    r_eff = min(r, len(S))
    col_scores = np.sum(Vt[:r_eff, :] ** 2, axis=0)
    row_scores = np.sum(U[:, :r_eff] ** 2, axis=1)
    I_r = np.sort(np.argsort(row_scores)[-r:])
    J_r = np.sort(np.argsort(col_scores)[-r:])
    M_check = delta_w[np.ix_(I_r, J_r)]
    s = np.linalg.svd(M_check, compute_uv=False)
    cond = s[0] / s[-1] if s[-1] > 1e-15 else float("inf")
    return I_r, J_r, cond


def select_indices_uniform(m, n, r):
    """策略 3：均匀间隔采样 —— 无需明文 ΔW。

    在 [0, m) 和 [0, n) 上均匀取 r 个索引。
    不保证条件数最优，但不需要任何明文信息，安全性最强。
    [审查 #4] 此函数为新增，作为无需明文的对比基线。
    """
    I_r = np.linspace(0, m - 1, r, dtype=int)
    J_r = np.linspace(0, n - 1, r, dtype=int)
    return I_r, J_r, float("inf")  # cond 未知（不访问明文）


# ═══════════════════════════════════════════════════════════════════════════
# ── CKKS 加解密 ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def encrypt_vectors(vecs, context, tag=""):
    cts, nbytes, t0 = [], 0, time.time()
    for i, v in enumerate(vecs):
        ser = ts.ckks_vector(context, v.tolist()).serialize()
        cts.append(ser); nbytes += len(ser)
        if (i + 1) % 500 == 0:
            print(f"    {tag} {i + 1}/{len(vecs)}")
    dt = time.time() - t0
    print(f"    {tag}: {len(cts)} ct, {nbytes/1024/1024:.1f} MB, {dt:.2f}s")
    return cts, dt, nbytes


def decrypt_selected(ct_list, indices, context):
    rows, nbytes = [], 0
    t0 = time.time()
    for idx in indices:
        rows.append(np.array(ts.ckks_vector_from(context, ct_list[idx]).decrypt()))
        nbytes += len(ct_list[idx])
    return np.array(rows), time.time() - t0, nbytes


# ═══════════════════════════════════════════════════════════════════════════
# ── 流水线 A：同态聚合 ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def pipeline_homomorphic_aggregation(B_list, A_list, ctx, dim):
    """
    [审查 #5] 单进程原型 — 所有客户端在同一个 for 循环中顺序加密，
    同态聚合是本地 agg_rows[k] += client_rows[k] 操作。

    逻辑角色：
      客户端：本地计算 ΔW_i，双编码 CKKS 加密后"发送"（存入本地变量）
      服务端：CKKS 密文加法聚合，全程无法解密

    返回 (enc_cols, enc_rows)，各为 list[bytes] × dim。
    """
    n = len(B_list)
    print(f"\n  Homomorphic aggregation ({n} clients, dual encoding, dim={dim})")
    agg_cols, agg_rows = None, None
    t_enc, t_add = 0.0, 0.0

    for ci, (Bi, Ai) in enumerate(zip(B_list, A_list)):
        print(f"\n  Client {ci}:")
        t0 = time.time()
        dWi = Bi.astype(np.float64) @ Ai.astype(np.float64)
        print(f"    ΔW_i computed: {time.time()-t0:.3f}s")

        t0 = time.time()
        cr = [ts.ckks_vector(ctx, dWi[k, :].tolist()) for k in range(dim)]
        dt = time.time() - t0; t_enc += dt
        print(f"    Row encrypt: {dim} ct, {dt:.2f}s")

        t0 = time.time()
        cc = [ts.ckks_vector(ctx, dWi[:, j].tolist()) for j in range(dim)]
        dt = time.time() - t0; t_enc += dt
        print(f"    Col encrypt: {dim} ct, {dt:.2f}s")

        t0 = time.time()
        if agg_rows is None:
            agg_rows, agg_cols = cr, cc
        else:
            for k in range(dim): agg_rows[k] += cr[k]
            for j in range(dim): agg_cols[j] += cc[j]
        dt = time.time() - t0; t_add += dt
        print(f"    Homomorphic add: {dt:.2f}s")

    print(f"\n  Total encrypt: {t_enc:.1f}s, total add: {t_add:.1f}s")
    print("  Serializing ...")
    t0 = time.time()
    ec = [v.serialize() for v in agg_cols]
    er = [v.serialize() for v in agg_rows]
    bc = sum(len(s) for s in ec); br = sum(len(s) for s in er)
    print(f"  {len(ec)}+{len(er)} ct, {(bc+br)/1024/1024:.1f} MB, {time.time()-t0:.2f}s")
    return ec, er


# ═══════════════════════════════════════════════════════════════════════════
# ── 流水线 B：明文捷径 ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def pipeline_plaintext_shortcut(B_list, A_list, ctx, dim):
    delta_w = np.zeros((dim, dim), dtype=np.float64)
    for Bi, Ai in zip(B_list, A_list):
        delta_w += Bi.astype(np.float64) @ Ai.astype(np.float64)
    print(f"\n  Plaintext ΔW: rank={np.linalg.matrix_rank(delta_w)}, "
          f"‖ΔW‖={np.linalg.norm(delta_w,'fro'):.4f}")
    print("  Encrypting rows ...")
    row_ct, _, _ = encrypt_vectors([delta_w[i, :] for i in range(dim)], ctx, "Row")
    print("  Encrypting cols ...")
    col_ct, _, _ = encrypt_vectors([delta_w[:, j] for j in range(dim)], ctx, "Col")
    return delta_w, col_ct, row_ct


# ═══════════════════════════════════════════════════════════════════════════
# ── 解密对比（含多策略）────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def run_decryption_comparison(delta_w_plain, enc_cols, enc_rows, ctx,
                               label, true_rank, dim, strategies=None,
                               r_values=None):
    """
    全量 + 骨架解密对比，可选多种索引选择策略。

    strategies: dict of {name: callable(delta_w, r) -> (I, J, cond)}
    r_values:   骨架秩列表（默认用全局 R_VALUES）
    """
    if strategies is None:
        strategies = {"mincond": lambda dw, r: select_indices_mincond(dw, r)}
    if r_values is None:
        r_values = R_VALUES

    print(f"\n{'─'*60}")
    print(f"  [{label}] Decryption comparison  (dim={dim})")
    print(f"{'─'*60}")

    # 全量解密基线
    print("  Full decryption (all columns) ...")
    t0 = time.time()
    _colmat, t_full, b_full = decrypt_selected(enc_cols, list(range(dim)), ctx)
    _full_mat = _colmat.T  # 列堆叠 → 转置
    eps_full = relative_error(_full_mat, delta_w_plain)
    print(f"    t={t_full:.2f}s, bytes={b_full/1024/1024:.1f}MB, ε_full={eps_full:.2e}")

    # 骨架解密 — 每种策略分别跑
    max_r = min(dim, true_rank)
    rs = [r for r in r_values if r <= max_r]
    all_results = {}

    for sname, sfunc in strategies.items():
        print(f"\n  Index strategy: {sname}")
        results = []
        for r in rs:
            I_r, J_r, cond = sfunc(delta_w_plain, r)
            C_raw, t_cols, b_cols = decrypt_selected(enc_cols, J_r, ctx)
            C_r = C_raw.T
            R_r, t_rows, b_rows = decrypt_selected(enc_rows, I_r, ctx)
            M_r = R_r[:, J_r]

            # 验证交叉块满秩
            if np.linalg.matrix_rank(M_r) < r:
                print(f"    r={r:2d}: rank(M_r) < r, skip")
                continue

            dW_rec = C_r @ np.linalg.inv(M_r) @ R_r
            eps = relative_error(dW_rec, delta_w_plain)
            t_skel = t_cols + t_rows; b_skel = b_cols + b_rows

            results.append(dict(
                r=r, cond_Mr=cond, error=eps,
                time_skeleton=t_skel, time_full=t_full,
                bytes_skeleton=b_skel, bytes_full=b_full,
                speedup=t_full / t_skel if t_skel > 0 else float("inf"),
                comm_saving=(1 - b_skel / b_full) * 100 if b_full > 0 else 0,
            ))
            status = "✓" if eps < TAU else "✗"
            print(f"    r={r:2d}: ε={eps:.2e} {status}  cond={cond:.0f}  "
                  f"t_skel={t_skel:.3f}s  speedup={t_full/t_skel:.0f}×  "
                  f"save={results[-1]['comm_saving']:.1f}%")
        all_results[sname] = results

    return all_results, eps_full


# ═══════════════════════════════════════════════════════════════════════════
# ── 可视化 ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results_a, results_b, eps_full_a, eps_full_b,
                 true_rank, out_name):
    """并排对比流水线 A（同态聚合）和流水线 B（明文捷径）。"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("CKKS Skeleton Decryption — Homomorphic vs Plaintext",
                 fontsize=14, fontweight="bold")
    ca, cb = "#2ecc71", "#3498db"
    x = np.arange(len(results_a)); rs = [r["r"] for r in results_a]
    w = 0.3

    def _db(ax, va, vb, yl, t):
        ax.bar(x - w / 2, va, w, label="Homomorphic agg", color=ca)
        ax.bar(x + w / 2, vb, w, label="Plaintext shortcut", color=cb)
        ax.set_xticks(x); ax.set_xticklabels(rs)
        ax.set_ylabel(yl); ax.set_title(t); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 0]
    t_skel_a = [r["time_skeleton"] for r in results_a]
    t_skel_b = [r["time_skeleton"] for r in results_b]
    t_full = results_a[0]["time_full"]
    # 用对数刻度 — 骨架 ~0.01s，全量 ~5s，线性会被压扁
    ax.bar(x - w / 2, t_skel_a, w, label="Hom agg", color=ca)
    ax.bar(x + w / 2, t_skel_b, w, label="Shortcut", color=cb)
    ax.bar(x[-1] + w * 2, t_full, w * 2, label=f"Full ({t_full:.1f}s)",
           color="red", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xticks(list(x) + [x[-1] + w * 2]); ax.set_xticklabels(rs + ["full"])
    ax.set_ylabel("Time (s, log scale)"); ax.set_title("(a) Decryption Time")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    _db(ax, [r["bytes_skeleton"] / 1024 for r in results_a],
        [r["bytes_skeleton"] / 1024 for r in results_b],
        "KB", "(b) Downlink Data (Server→Decryptor)")
    ax.axhline(y=results_a[0]["bytes_full"] / 1024, color="red", ls="--", lw=1,
               label=f"Full ({results_a[0]['bytes_full']/1024/1024:.0f} MB)")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    ax.semilogy(rs, [r["error"] for r in results_a], "o-", color=ca, lw=2, ms=8, label="Hom agg")
    ax.semilogy(rs, [r["error"] for r in results_b], "s--", color=cb, lw=2, ms=8, label="Shortcut")
    ax.axhline(y=TAU, color="gray", ls="--", label=f"τ={TAU}")
    ax.axhline(y=eps_full_a, color=ca, ls=":", label=f"ε_full(hom)={eps_full_a:.1e}")
    ax.axhline(y=eps_full_b, color=cb, ls=":", label=f"ε_full(short)={eps_full_b:.1e}")
    ax.set_xticks(rs); ax.set_xticklabels(rs)
    ax.set_xlabel("r"); ax.set_ylabel("Relative Frobenius Error")
    ax.set_title(f"(c) Error (rank={true_rank})"); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    _db(ax, [r["speedup"] for r in results_a],
        [r["speedup"] for r in results_b], "×", "(d) Speedup T_full/T_skel")

    ax = axes[1, 1]
    _db(ax, [r["comm_saving"] for r in results_a],
        [r["comm_saving"] for r in results_b],
        "%", "(e) Comm Saving = (1 - skel/full)×100%")

    ax = axes[1, 2]
    ax.semilogy(rs, [r["cond_Mr"] for r in results_a], "o-", color=ca, lw=2, ms=8, label="Hom agg")
    ax.semilogy(rs, [r["cond_Mr"] for r in results_b], "s--", color=cb, lw=2, ms=8, label="Shortcut")
    ax.set_xticks(rs); ax.set_xticklabels(rs)
    ax.set_xlabel("r"); ax.set_ylabel("cond(M_r)")
    ax.set_title("(f) Condition Number of M_r"); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(RES_DIR, out_name)
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Chart → {p}")


def plot_strategy_comparison(strat_results, eps_full, true_rank, out_name):
    """对比三种索引选择策略的误差和条件数。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Index Selection Strategy Comparison", fontsize=14, fontweight="bold")
    colors = {"mincond": "#2ecc71", "leverage": "#e74c3c", "uniform": "#f39c12"}
    markers = {"mincond": "o-", "leverage": "s--", "uniform": "^:"}

    for sname, results in strat_results.items():
        rs = [r["r"] for r in results]
        axes[0].semilogy(rs, [r["error"] for r in results], markers[sname],
                         color=colors[sname], lw=2, ms=8, label=f"{sname}")
        # 只对非 inf 条件数画线
        conds = [r["cond_Mr"] for r in results if r["cond_Mr"] < float("inf")]
        c_rs = [r["r"] for r in results if r["cond_Mr"] < float("inf")]
        if conds:
            axes[1].semilogy(c_rs, conds, markers[sname],
                             color=colors[sname], lw=2, ms=8, label=f"{sname}")

    axes[0].axhline(y=TAU, color="gray", ls="--", label=f"τ={TAU}")
    axes[0].set_xlabel("r"); axes[0].set_ylabel("ε")
    axes[0].set_title(f"(a) Reconstruction Error (rank={true_rank})")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("r"); axes[1].set_ylabel("cond(M_r)")
    axes[1].set_title("(b) Condition Number of M_r")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(RES_DIR, out_name)
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Chart → {p}")


# ═══════════════════════════════════════════════════════════════════════════
# ── CSV ──────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def save_csv(results_a, results_b, fname):
    rows = []
    for ra, rb in zip(results_a, results_b):
        rows.append(dict(
            r=ra["r"],
            cond_Mr_hom=ra["cond_Mr"], cond_Mr_short=rb["cond_Mr"],
            error_hom=ra["error"], error_short=rb["error"],
            t_skel_hom=ra["time_skeleton"], t_skel_short=rb["time_skeleton"],
            t_full_hom=ra["time_full"], t_full_short=rb["time_full"],
            speedup_hom=ra["speedup"], speedup_short=rb["speedup"],
            comm_save_hom=ra["comm_saving"], comm_save_short=rb["comm_saving"],
        ))
    with open(os.path.join(RES_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  CSV → {os.path.join(RES_DIR, fname)}")


def save_strategy_csv(all_results, eps_full, fname):
    rows = []
    strat_names = list(all_results.keys())
    n_r = len(all_results[strat_names[0]])
    for i in range(n_r):
        row = {"r": all_results[strat_names[0]][i]["r"]}
        for sn in strat_names:
            r = all_results[sn][i]
            row[f"{sn}_error"] = r["error"]
            row[f"{sn}_cond"] = r["cond_Mr"]
            row[f"{sn}_speedup"] = r["speedup"]
        row["eps_full"] = eps_full
        rows.append(row)
    with open(os.path.join(RES_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  CSV → {os.path.join(RES_DIR, fname)}")


# ═══════════════════════════════════════════════════════════════════════════
# ── Demo：2 客户端 10×4 ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _run_demo_pipeline(B0, A0, B1, A1, label, d, rk, r_values=None):
    """内部：运行单组 demo 流水线，返回 (all_res, eps_full, true_r, dW)。"""
    dW0 = B0 @ A0; dW1 = B1 @ A1; dW = dW0 + dW1
    true_r = np.linalg.matrix_rank(dW)

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2 ** 40; ctx.generate_galois_keys()

    ec, er = pipeline_homomorphic_aggregation([B0, B1], [A0, A1], ctx, dim=d)

    strategies = {
        "mincond":  lambda dw, r: select_indices_mincond(dw, r, n_trials=5000),
        "leverage": lambda dw, r: select_indices_leverage(dw, r),
        "uniform":  lambda dw, r: select_indices_uniform(d, d, r),
    }
    all_res, eps_full = run_decryption_comparison(
        dW, ec, er, ctx, label, true_r, dim=d, strategies=strategies,
        r_values=r_values)
    return all_res, eps_full, true_r, dW


def demo_mode():
    """2 客户端 10×4 基础演示。"""
    print("=" * 70)
    print("  DEMO: 2 clients, 10×4 LoRA → ΔW (10×10)")
    print("=" * 70)

    np.random.seed(42)
    d = 10; rk = 4
    r_vals = list(range(2, d + 1))  # [2,3,4,5,6,7,8,9,10]
    B0 = np.random.randn(d, rk) * 0.1; A0 = np.random.randn(rk, d) * 0.1
    B1 = np.random.randn(d, rk) * 0.1; A1 = np.random.randn(rk, d) * 0.1

    all_res, eps_full, true_r, dW = _run_demo_pipeline(
        B0, A0, B1, A1, "Demo (float)", d, rk, r_values=r_vals)

    print(f"\n  ΔW =\n{np.array2string(dW, precision=3, suppress_small=True)}")
    print(f"\n  ε_full = {eps_full:.2e}")
    for sname, results in all_res.items():
        n_pass = sum(1 for r in results if r["error"] < TAU)
        print(f"  {sname}: {n_pass}/{len(results)} pass")
        for r in results:
            print(f"    r={r['r']:2d}: ε={r['error']:.2e}  cond={r['cond_Mr']:.0f}  "
                  f"speedup={r['speedup']:.0f}×")

    save_strategy_csv(all_res, eps_full, "demo_10x4_results.csv")
    plot_strategy_comparison(all_res, eps_full, true_r, "demo_10x4_results.png")
    print("\nDemo done.")


def demo_float_vs_int():
    """
    浮点 vs 整数矩阵对比实验。

    同一随机结构（同 seed），两组版本：
      - 浮点组：raw * 0.1（连续值，CKKS 编码有舍入误差 ≈ 2⁻⁴⁰ ≈ 9e-13）
      - 整数组：round(raw * 200)（精确整数，CKKS 编码零误差）

    两组矩阵 ΔW 的结构完全相同（同 rank、同奇异值分布），
    唯一区别是值是否为整数 — 隔离 CKKS 编码精度的影响。
    """
    print("=" * 70)
    print("  DEMO: Float vs Integer — CKKS Encoding Precision")
    print("=" * 70)

    d = 10; rk = 4
    # Demo 小矩阵使用更密的 r 采样 — 看清 rank=8 附近的相变
    r_values_demo = list(range(2, d + 1))  # [2,3,4,5,6,7,8,9,10]

    # ── 生成相同随机结构的原始数据 ──
    np.random.seed(42)
    B0_raw = np.random.randn(d, rk)
    A0_raw = np.random.randn(rk, d)
    B1_raw = np.random.randn(d, rk)
    A1_raw = np.random.randn(rk, d)

    # ── 浮点组：连续值（≈ [-0.05, 0.05]）─────────────────────────
    B0_f = B0_raw * 0.1;  A0_f = A0_raw * 0.1
    B1_f = B1_raw * 0.1;  A1_f = A1_raw * 0.1
    print("\n  [Float group] values * 0.1  (CKKS encoding: has rounding)")
    print(f"    B range: [{B0_f.min():.3f}, {B0_f.max():.3f}]")

    # ── 整数组：精确整数（≈ [-50, 50]）────────────────────────────
    B0_i = np.round(B0_raw * 200).astype(np.float64)
    A0_i = np.round(A0_raw * 200).astype(np.float64)
    B1_i = np.round(B1_raw * 200).astype(np.float64)
    A1_i = np.round(A1_raw * 200).astype(np.float64)
    print("\n  [Integer group] round(raw * 200)  (CKKS encoding: exact)")
    print(f"    B all integer? {np.all(B0_i == B0_i.astype(np.int64))}")
    print(f"    B range: [{B0_i.min():.0f}, {B0_i.max():.0f}]")

    # ── 运行两组 ─────────────────────────────────────────────────
    all_res_f, eps_f, true_r_f, dW_f = _run_demo_pipeline(
        B0_f, A0_f, B1_f, A1_f, "Float", d, rk, r_values=r_values_demo)
    all_res_i, eps_i, true_r_i, dW_i = _run_demo_pipeline(
        B0_i, A0_i, B1_i, A1_i, "Integer", d, rk, r_values=r_values_demo)

    # ── 对比输出 ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Float vs Integer")
    print(f"{'='*70}")
    print(f"  ΔW (float)   rank={true_r_f}, ‖ΔW‖={np.linalg.norm(dW_f,'fro'):.2f}")
    print(f"  ΔW (integer) rank={true_r_i}, ‖ΔW‖={np.linalg.norm(dW_i,'fro'):.1f}")
    print(f"\n  {'Metric':<25} {'Float':>16} {'Integer':>16} {'Δ':>16}")
    print(f"  {'-'*70}")
    print(f"  {'ε_full':<25} {eps_f:>16.2e} {eps_i:>16.2e} "
          f"{abs(eps_f - eps_i) / max(eps_f, eps_i) * 100:>15.1f}% diff")

    for r_val in [4, 6, 8]:
        rf = [r for r in all_res_f["mincond"] if r["r"] == r_val]
        ri = [r for r in all_res_i["mincond"] if r["r"] == r_val]
        if rf and ri:
            ef = rf[0]["error"]; ei = ri[0]["error"]
            diff_pct = abs(ef - ei) / max(ef, ei) * 100 if max(ef, ei) > 1e-15 else 0
            print(f"  {'ε_skel(r='+str(r_val)+') mincond':<25} {ef:>16.2e} {ei:>16.2e} "
                  f"{diff_pct:>15.1f}% diff")

    # 中间过程对比（r=8, mincond）
    rf8 = [r for r in all_res_f["mincond"] if r["r"] == 8][0]
    ri8 = [r for r in all_res_i["mincond"] if r["r"] == 8][0]
    print(f"\n  {'Intermediate (r=8, mincond)':<25} {'Float':>16} {'Integer':>16}")
    print(f"  {'-'*50}")
    print(f"  {'cond(M_8)':<25} {rf8['cond_Mr']:>16.0f} {ri8['cond_Mr']:>16.0f}")
    print(f"  {'t_skel (s)':<25} {rf8['time_skeleton']:>16.4f} {ri8['time_skeleton']:>16.4f}")
    print(f"  {'t_full (s)':<25} {rf8['time_full']:>16.4f} {ri8['time_full']:>16.4f}")
    print(f"  {'speedup':<25} {rf8['speedup']:>16.0f}× {ri8['speedup']:>15.0f}×")
    print(f"  {'bytes_skeleton':<25} {rf8['bytes_skeleton']/1024:>16.1f}KB "
          f"{ri8['bytes_skeleton']/1024:>15.1f}KB")

    # ── 分析：CKKS 编码误差的理论界限 ──
    print(f"\n  CKKS encoding precision (scale=2^40):")
    print(f"    Max encoding error: 0.5 / 2^40 ≈ 4.5e-13  (per element)")
    print(f"    vs encryption noise: ε_full ≈ {min(eps_f, eps_i):.1e}")
    print(f"    Encoding error is ~{min(eps_f, eps_i)/4.5e-13:.0f}× smaller than encryption noise")
    print(f"    → 预期：两组 ε 无明显差异（编码误差被加密噪声淹没）")

    # ── 图表 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Float vs Integer — CKKS Encoding Precision Impact",
                 fontsize=14, fontweight="bold")

    rs = [r["r"] for r in all_res_f["mincond"]]
    for ax, sname in zip(axes, ["mincond", "uniform"]):
        ef = [r["error"] for r in all_res_f[sname]]
        ei = [r["error"] for r in all_res_i[sname]]
        ax.semilogy(rs, ef, "o-", color="#e74c3c", lw=2, ms=8, label="Float")
        ax.semilogy(rs, ei, "s--", color="#2980b9", lw=2, ms=8, label="Integer")
        ax.axhline(y=TAU, color="gray", ls="--", label=f"τ={TAU}")
        ax.set_xticks(rs); ax.set_xticklabels(rs)
        ax.set_xlabel("r"); ax.set_ylabel("ε")
        ax.set_title(f"Reconstruction Error ({sname})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        # 标注差异
        for ri_val, efv, eiv in zip(rs, ef, ei):
            diff = abs(efv - eiv) / max(efv, eiv) * 100 if max(efv, eiv) > 1e-15 else 0
            if diff > 1:
                ax.annotate(f"Δ={diff:.0f}%", (ri_val, max(efv, eiv)),
                           textcoords="offset points", xytext=(0, 8),
                           ha="center", fontsize=7, color="purple")

    plt.tight_layout()
    png = os.path.join(RES_DIR, "demo_float_vs_int.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Chart → {png}")

    # CSV
    rows = []
    for i, rf_val in enumerate(all_res_f["mincond"]):
        ri_val = all_res_i["mincond"][i]
        rows.append(dict(
            r=rf_val["r"],
            float_eps_full=eps_f, int_eps_full=eps_i,
            float_error_mincond=all_res_f["mincond"][i]["error"],
            int_error_mincond=all_res_i["mincond"][i]["error"],
            float_error_leverage=all_res_f["leverage"][i]["error"],
            int_error_leverage=all_res_i["leverage"][i]["error"],
            float_error_uniform=all_res_f["uniform"][i]["error"],
            int_error_uniform=all_res_i["uniform"][i]["error"],
            float_cond=all_res_f["mincond"][i]["cond_Mr"],
            int_cond=all_res_i["mincond"][i]["cond_Mr"],
        ))
    csv_p = os.path.join(RES_DIR, "demo_float_vs_int.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  CSV → {csv_p}")

    print("\nFloat vs Integer comparison done.")


# ═══════════════════════════════════════════════════════════════════════════
# ── 主实验 ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CKKS Skeleton Decryption — Full Homomorphic Pipeline Test")
    print("=" * 70)
    print(f"  [审查 #5] 单进程密码学原型 — 所有操作在同一进程中顺序执行")
    print(f"  {N_CLIENTS} clients, rank={RANK}, ΔW={N_DIM}×{N_DIM}")
    print(f"  CKKS: poly_modulus_degree={POLY_MODULUS_DEGREE}, "
          f"scale=2^{int(np.log2(GLOBAL_SCALE))}")

    # ── 加载 ──
    print("\n[1] Loading weights ...")
    Bs, As_ = load_weights(N_CLIENTS)
    print(f"  Loaded {len(Bs)} clients, B={Bs[0].shape}, A={As_[0].shape}")

    delta_w_plain = np.zeros((N_DIM, N_DIM), dtype=np.float64)
    for Bi, Ai in zip(Bs, As_):
        delta_w_plain += Bi.astype(np.float64) @ Ai.astype(np.float64)
    true_rank = np.linalg.matrix_rank(delta_w_plain)
    print(f"  ΔW_plain rank={true_rank}, "
          f"‖ΔW‖={np.linalg.norm(delta_w_plain, 'fro'):.4f}")

    # ── CKKS 上下文 ──
    print("\n[2] Creating CKKS context ...")
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=POLY_MODULUS_DEGREE,
                     coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
    ctx.global_scale = GLOBAL_SCALE; ctx.generate_galois_keys()
    print(f"  Done (max_slots={POLY_MODULUS_DEGREE // 2})")

    # ═══════════════════════════════════════════════════════════════════
    # 流水线 A
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PIPELINE A: Homomorphic Aggregation (secure)")
    print("=" * 70)
    t0 = time.time()
    enc_cols_a, enc_rows_a = pipeline_homomorphic_aggregation(Bs, As_, ctx, N_DIM)
    t_pipe_a = time.time() - t0
    print(f"\n  Total pipeline A time: {t_pipe_a:.1f}s")

    # 流水线 A 上对比三种索引策略
    strategies = {
        "mincond":  lambda dw, r: select_indices_mincond(dw, r),
        "leverage": lambda dw, r: select_indices_leverage(dw, r),
        "uniform":  lambda dw, r: select_indices_uniform(N_DIM, N_DIM, r),
    }
    all_a, eps_full_a = run_decryption_comparison(
        delta_w_plain, enc_cols_a, enc_rows_a, ctx,
        "Pipeline A (homomorphic agg)", true_rank, N_DIM,
        strategies=strategies)

    # ═══════════════════════════════════════════════════════════════════
    # 流水线 B
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PIPELINE B: Plaintext Shortcut (control)")
    print("=" * 70)
    t0 = time.time()
    dw_short, enc_cols_b, enc_rows_b = pipeline_plaintext_shortcut(Bs, As_, ctx, N_DIM)
    t_pipe_b = time.time() - t0
    print(f"\n  Total pipeline B time: {t_pipe_b:.1f}s")

    all_b, eps_full_b = run_decryption_comparison(
        dw_short, enc_cols_b, enc_rows_b, ctx,
        "Pipeline B (plaintext shortcut)", true_rank, N_DIM,
        strategies=strategies)

    # ═══════════════════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Pipeline A (homomorphic agg): {t_pipe_a:.1f}s, ε_full={eps_full_a:.2e}")
    print(f"  Pipeline B (plaintext short):  {t_pipe_b:.1f}s, ε_full={eps_full_b:.2e}")

    # 策略对比
    print(f"\n  Index strategy comparison (pipeline A, r=16):")
    for sname in ["leverage", "mincond", "uniform"]:
        r16 = [r for r in all_a[sname] if r["r"] == 16]
        if r16:
            rr = r16[0]
            cond_str = f"cond={rr['cond_Mr']:.0f}" if rr['cond_Mr'] < float("inf") else "cond=N/A"
            print(f"    {sname:10s}: ε={rr['error']:.2e}, {cond_str}, "
                  f"speedup={rr['speedup']:.0f}×")

    # 图表 + CSV
    res_a_mincond = all_a.get("mincond", [])
    res_b_mincond = all_b.get("mincond", [])
    if res_a_mincond and res_b_mincond:
        save_csv(res_a_mincond, res_b_mincond, "ckks_skeleton_results.csv")
        plot_results(res_a_mincond, res_b_mincond, eps_full_a, eps_full_b,
                     true_rank, "ckks_skeleton_results.png")

    save_strategy_csv(all_a, eps_full_a, "ckks_strategy_comparison.csv")
    plot_strategy_comparison(all_a, eps_full_a, true_rank,
                             "ckks_strategy_comparison.png")

    print("\nDone.")


# ═══════════════════════════════════════════════════════════════════════════
# ── 入口 ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CKKS Skeleton Decryption Test")
    p.add_argument("--demo", action="store_true",
                   help="Run 2-client 10×4 illustrative demo")
    p.add_argument("--demo-compare", action="store_true",
                   help="Run float vs integer comparison on 10×4 demo")
    args = p.parse_args()
    if args.demo_compare:
        demo_float_vs_int()
    elif args.demo:
        demo_mode()
    else:
        main()
