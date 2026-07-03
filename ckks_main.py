"""主实验入口：main() / main_partial()。"""

import time
import numpy as np
import tenseal as ts
import os, csv

from ckks_config import (N_CLIENTS, N_DIM, RANK, R_VALUES,
                          POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES, GLOBAL_SCALE,
                          TAU, RES_DIR, PARTIAL_RATIOS)
from ckks_utils import load_weights, create_ckks_context, make_public_context
from ckks_indices import (select_indices_mincond, select_indices_leverage,
                           select_indices_uniform, select_encrypt_indices,
                           uniform_index_union)
from ckks_pipelines import (pipeline_homomorphic_aggregation,
                            pipeline_plaintext_shortcut,
                            pipeline_partial_encryption,
                            pipeline_homomorphic_factor_mult)
from ckks_compare import run_decryption_comparison
from ckks_plot import (plot_results, plot_strategy_comparison,
                       save_csv, save_strategy_csv)


def main():
    """全加密主实验 — 流水线 A vs B + 三策略对比。"""
    print("=" * 70)
    print("  CKKS Skeleton Decryption — Full Homomorphic Pipeline Test")
    print("=" * 70)
    print(f"  [单进程密码学原型] {N_CLIENTS} clients, rank={RANK}, ΔW={N_DIM}×{N_DIM}")
    print(f"  CKKS: poly_modulus_degree={POLY_MODULUS_DEGREE}, scale=2^{int(np.log2(GLOBAL_SCALE))}")

    # 加载
    print("\n[1] Loading weights ...")
    Bs, As_ = load_weights(N_CLIENTS)
    print(f"  Loaded {len(Bs)} clients, B={Bs[0].shape}, A={As_[0].shape}")
    delta_w_plain = np.zeros((N_DIM, N_DIM), dtype=np.float64)
    for Bi, Ai in zip(Bs, As_):
        delta_w_plain += Bi.astype(np.float64) @ Ai.astype(np.float64)
    true_rank = np.linalg.matrix_rank(delta_w_plain)
    print(f"  ΔW_plain rank={true_rank}")

    # CKKS
    print("\n[2] Creating CKKS context ...")
    ctx = create_ckks_context()
    print(f"  Done (max_slots={POLY_MODULUS_DEGREE//2})")

    strategies = {
        "mincond":  lambda dw, r: select_indices_mincond(dw, r),
        "leverage": lambda dw, r: select_indices_leverage(dw, r),
        "uniform":  lambda dw, r: select_indices_uniform(N_DIM, N_DIM, r),
    }

    # Pipeline A
    print("\n" + "=" * 70)
    print("  PIPELINE A: Homomorphic Aggregation (secure)")
    print("=" * 70)
    t0 = time.time()
    enc_cols_a, enc_rows_a = pipeline_homomorphic_aggregation(Bs, As_, ctx, N_DIM)
    t_a = time.time() - t0
    all_a, eps_a = run_decryption_comparison(
        delta_w_plain, enc_cols_a, enc_rows_a, ctx,
        "Pipeline A", true_rank, N_DIM, strategies=strategies)

    # Pipeline B
    print("\n" + "=" * 70)
    print("  PIPELINE B: Plaintext Shortcut (control)")
    print("=" * 70)
    t0 = time.time()
    dw_short, enc_cols_b, enc_rows_b = pipeline_plaintext_shortcut(Bs, As_, ctx, N_DIM)
    t_b = time.time() - t0
    all_b, eps_b = run_decryption_comparison(
        dw_short, enc_cols_b, enc_rows_b, ctx,
        "Pipeline B", true_rank, N_DIM, strategies=strategies)

    # Summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  Pipeline A: {t_a:.1f}s, ε_full={eps_a:.2e}")
    print(f"  Pipeline B: {t_b:.1f}s, ε_full={eps_b:.2e}")
    print(f"\n  Strategy comparison (pipeline A, r=16):")
    for sn in ["leverage", "mincond", "uniform"]:
        r16 = [r for r in all_a[sn] if r["r"] == 16]
        if r16:
            rr = r16[0]
            cs = f"cond={rr['cond_Mr']:.0f}" if rr['cond_Mr'] < float("inf") else "cond=N/A"
            print(f"    {sn:10s}: ε={rr['error']:.2e}, {cs}, speedup={rr['speedup']:.0f}×")

    save_csv(all_a["mincond"], all_b["mincond"], "ckks_skeleton_results.csv")
    plot_results(all_a["mincond"], all_b["mincond"], eps_a, eps_b,
                 true_rank, "ckks_skeleton_results.png")
    save_strategy_csv(all_a, eps_a, "ckks_strategy_comparison.csv")
    plot_strategy_comparison(all_a, eps_a, true_rank, "ckks_strategy_comparison.png")
    print("\nDone.")


def main_partial():
    """部分加密扫描 — 5 个 encrypt_ratio × 1 策略 × 8 r 值。"""
    print("=" * 70)
    print(f"  Partial Encryption Scan — {[f'{r:.0%}' for r in PARTIAL_RATIOS]}")
    print("=" * 70)

    Bs, As_ = load_weights(N_CLIENTS)
    delta_w_plain = np.zeros((N_DIM, N_DIM), dtype=np.float64)
    for Bi, Ai in zip(Bs, As_):
        delta_w_plain += Bi.astype(np.float64) @ Ai.astype(np.float64)
    true_rank = np.linalg.matrix_rank(delta_w_plain)

    ctx = create_ckks_context()
    strategies = {"mincond": lambda dw, r: select_indices_mincond(dw, r)}
    p_rvals = [2, 4, 6, 8, 10, 12, 14, 16]

    summaries, detail_rows = [], []

    for ratio in PARTIAL_RATIOS:
        print(f"\n{'='*70}")
        k = max(1, int(N_DIM * ratio))
        print(f"  ratio={ratio:.0%}  (k={k}/{N_DIM})")
        print(f"{'='*70}")
        t0 = time.time()

        if ratio >= 1.0:
            enc_cols, enc_rows = pipeline_homomorphic_aggregation(Bs, As_, ctx, N_DIM)
            plain_cols = plain_rows = None
        else:
            er_idx, ec_idx = select_encrypt_indices(delta_w_plain, ratio)
            enc_cols, enc_rows, plain_cols, plain_rows = \
                pipeline_partial_encryption(Bs, As_, ctx, N_DIM, ratio, er_idx, ec_idx)

        t_pipe = time.time() - t0
        n_enc = sum(1 for c in enc_cols if c is not None)

        all_res, eps_full = run_decryption_comparison(
            delta_w_plain, enc_cols, enc_rows, ctx,
            f"ratio={ratio:.0%}", true_rank, N_DIM,
            strategies=strategies, r_values=p_rvals,
            plain_cols=plain_cols, plain_rows=plain_rows)

        summaries.append(dict(ratio=ratio, eps_full=eps_full, t_pipe=t_pipe, n_enc=n_enc))
        for sn, results in all_res.items():
            for r in results:
                detail_rows.append(dict(ratio=ratio, strategy=sn, **r))

    # Output
    print(f"\n{'='*70}\n  PARTIAL ENCRYPTION SUMMARY\n{'='*70}")
    print(f"  {'Ratio':>8s}  {'k':>5s}  {'ε_full':>10s}  {'ε_skel(r=16)':>14s}  "
          f"{'t_pipe':>8s}  {'t_skel':>8s}  {'enc_hit%':>8s}")
    print(f"  {'-'*70}")
    for s in summaries:
        r16 = [r for r in detail_rows
               if r["ratio"] == s["ratio"] and r["strategy"] == "mincond" and r["r"] == 16]
        if r16:
            rr = r16[0]
            print(f"  {s['ratio']:>7.0%}  {s['n_enc']:>5d}  {s['eps_full']:>10.2e}  "
                  f"{rr['error']:>14.2e}  {s['t_pipe']:>7.1f}s  "
                  f"{rr['time_skeleton']:>7.3f}s  {rr['enc_hit_ratio']:>7.0%}")

    # Chart
    import matplotlib.pyplot as plt
    ratios_pct = [s["ratio"] * 100 for s in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Partial Encryption Impact", fontsize=14, fontweight="bold")

    axes[0].semilogx(ratios_pct, [s["eps_full"] for s in summaries],
                     "o-", color="#e74c3c", lw=2, ms=8)
    axes[0].set_xlabel("Encrypt ratio (%)"); axes[0].set_ylabel("ε_full")
    axes[0].set_title("(a) Full Decryption Error"); axes[0].grid(True, alpha=0.3)

    pts = sorted([(r["ratio"] * 100, r["error"])
                  for r in detail_rows if r["strategy"] == "mincond" and r["r"] == 16])
    xs, ys = zip(*pts) if pts else ([], [])
    axes[1].loglog(xs, ys, "o-", color="#2ecc71", lw=2, ms=8)
    axes[1].set_xlabel("Encrypt ratio (%)"); axes[1].set_ylabel("ε_skel(r=16)")
    axes[1].set_title("(b) Skeleton Error at r=16"); axes[1].grid(True, alpha=0.3)

    axes[2].bar(np.arange(len(summaries)) - 0.2, [s["t_pipe"] for s in summaries],
                0.35, label="Pipeline", color="#3498db")
    r16t = []
    for s in summaries:
        rr = [r for r in detail_rows
              if r["ratio"] == s["ratio"] and r["strategy"] == "mincond" and r["r"] == 16]
        r16t.append(rr[0]["time_skeleton"] if rr else 0)
    axes[2].bar(np.arange(len(summaries)) + 0.2, r16t, 0.35,
                label="Skeleton", color="#2ecc71")
    axes[2].set_xticks(range(len(summaries)))
    axes[2].set_xticklabels([f"{r:.0%}" for r in PARTIAL_RATIOS], rotation=45)
    axes[2].set_ylabel("Time (s)"); axes[2].set_title("(c) Time Cost")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    png = os.path.join(RES_DIR, "partial_encryption_results.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Chart → {png}")

    csv_p = os.path.join(RES_DIR, "partial_encryption_results.csv")
    fields = ["ratio", "strategy", "r", "error", "cond_Mr", "time_skeleton",
              "time_full", "speedup", "enc_hit_ratio"]
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(detail_rows)
    print(f"  CSV → {csv_p}")
    print("\nPartial encryption scan done.")


def main_factor_mult():
    """流水线 C：客户端只发全加密 A/B，服务端密文乘法 + 聚合 + 平均 + 骨架加速。"""
    print("=" * 70)
    print("  CKKS Skeleton — Server-side Cipher Factor Multiplication (Pipeline C)")
    print("=" * 70)
    print(f"  [客户端仅上传加密 A/B] {N_CLIENTS} clients, rank={RANK}, ΔW={N_DIM}×{N_DIM}")
    print(f"  CKKS: poly_modulus_degree={POLY_MODULUS_DEGREE}, scale=2^{int(np.log2(GLOBAL_SCALE))}")

    # 加载（注意：参考矩阵用密文域同款的均值 ΔW_mean = (Σ B_i A_i)/N）
    print("\n[1] Loading weights ...")
    Bs, As_ = load_weights(N_CLIENTS)
    print(f"  Loaded {len(Bs)} clients, B={Bs[0].shape}, A={As_[0].shape}")
    delta_w_mean = np.zeros((N_DIM, N_DIM), dtype=np.float64)
    for Bi, Ai in zip(Bs, As_):
        delta_w_mean += Bi.astype(np.float64) @ Ai.astype(np.float64)
    delta_w_mean /= N_CLIENTS
    true_rank = np.linalg.matrix_rank(delta_w_mean)
    print(f"  ΔW_mean rank={true_rank}")

    # CKKS：分离公私钥边界
    #   secret_ctx —— 持私钥，只用于最后的解密方（可信第三方/聚合器之外）
    #   public_ctx —— 去私钥，分发给客户端(加密)与服务端(同态乘加)，无法解密
    # galois=False：Pipeline C 全程逐元素 ct×ct + 广播标量，不做旋转，省去 galois key
    print("\n[2] Creating CKKS context (secret + derived public) ...")
    secret_ctx = create_ckks_context(galois=False)
    public_ctx = make_public_context(secret_ctx)
    print(f"  secret.is_private()={secret_ctx.is_private()}, "
          f"public.is_private()={public_ctx.is_private()}  (max_slots={POLY_MODULUS_DEGREE//2})")

    # 骨架索引：uniform 的 r-sweep 并集（客户端事先可知，无需明文）
    I_union, J_union = uniform_index_union(N_DIM, R_VALUES)
    print(f"\n[3] Skeleton index union (uniform): "
          f"{len(I_union)} rows + {len(J_union)} cols over r={R_VALUES}")

    # Pipeline C —— 客户端加密 + 服务端同态运算，全程用公开 context（无私钥）
    print("\n" + "=" * 70)
    print("  PIPELINE C: Server-side Cipher Multiplication (secure, factor-only)")
    print("=" * 70)
    t0 = time.time()
    enc_cols, enc_rows = pipeline_homomorphic_factor_mult(
        Bs, As_, public_ctx, N_DIM, I_union, J_union, N_CLIENTS, RANK)
    t_c = time.time() - t0

    # 解密阶段 —— 唯一使用私钥 context 的地方
    strategies = {"uniform": lambda dw, r: select_indices_uniform(N_DIM, N_DIM, r)}
    all_c, eps_c = run_decryption_comparison(
        delta_w_mean, enc_cols, enc_rows, secret_ctx,
        "Pipeline C", true_rank, N_DIM, strategies=strategies, skip_full=True)

    # Summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  Pipeline C: {t_c:.1f}s (cipher mult + aggregate + average)")
    print(f"  Skeleton (uniform) errors:")
    for rr in all_c["uniform"]:
        status = "✓" if rr["error"] < TAU else "✗"
        print(f"    r={rr['r']:2d}: ε={rr['error']:.2e} {status}  "
              f"t_skel={rr['time_skeleton']:.3f}s")

    save_strategy_csv(all_c, eps_c, "ckks_pipelineC_results.csv")
    plot_strategy_comparison(all_c, eps_c, true_rank, "ckks_pipelineC_results.png")
    print("\nPipeline C done.")
