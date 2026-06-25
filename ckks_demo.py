"""Demo 实验：基础演示 / 浮点 vs 整数对比。"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ckks_config import RES_DIR, TAU
from ckks_utils import create_ckks_context
from ckks_indices import select_indices_mincond, select_indices_leverage, select_indices_uniform
from ckks_pipelines import pipeline_homomorphic_aggregation
from ckks_compare import run_decryption_comparison
from ckks_plot import save_strategy_csv, plot_strategy_comparison

os.makedirs(RES_DIR, exist_ok=True)


def _run_demo_pipeline(B0, A0, B1, A1, label, d, rk, r_values=None):
    dW0 = B0 @ A0; dW1 = B1 @ A1; dW = dW0 + dW1
    true_r = np.linalg.matrix_rank(dW)

    ctx = create_ckks_context()
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
    r_vals = list(range(2, d + 1))
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
    """浮点 vs 整数：隔离 CKKS 编码精度影响。"""
    print("=" * 70)
    print("  DEMO: Float vs Integer — CKKS Encoding Precision")
    print("=" * 70)

    d = 10; rk = 4
    r_vals = list(range(2, d + 1))

    np.random.seed(42)
    B0_raw = np.random.randn(d, rk); A0_raw = np.random.randn(rk, d)
    B1_raw = np.random.randn(d, rk); A1_raw = np.random.randn(rk, d)

    B0_f = B0_raw * 0.1; A0_f = A0_raw * 0.1
    B1_f = B1_raw * 0.1; A1_f = A1_raw * 0.1
    print("\n  [Float] values * 0.1 (CKKS encoding: rounding)")
    print(f"    B range: [{B0_f.min():.3f}, {B0_f.max():.3f}]")

    B0_i = np.round(B0_raw * 200).astype(np.float64)
    A0_i = np.round(A0_raw * 200).astype(np.float64)
    B1_i = np.round(B1_raw * 200).astype(np.float64)
    A1_i = np.round(A1_raw * 200).astype(np.float64)
    print("\n  [Integer] round(raw*200), CKKS encoding: exact")
    print(f"    B all integer? {np.all(B0_i == B0_i.astype(np.int64))}")
    print(f"    B range: [{B0_i.min():.0f}, {B0_i.max():.0f}]")

    all_res_f, eps_f, true_r_f, dW_f = _run_demo_pipeline(
        B0_f, A0_f, B1_f, A1_f, "Float", d, rk, r_values=r_vals)
    all_res_i, eps_i, true_r_i, dW_i = _run_demo_pipeline(
        B0_i, A0_i, B1_i, A1_i, "Integer", d, rk, r_values=r_vals)

    # ── 对比 ──
    print(f"\n{'='*70}")
    print("  COMPARISON: Float vs Integer")
    print(f"{'='*70}")
    print(f"  ΔW (float)   rank={true_r_f}, ‖ΔW‖={np.linalg.norm(dW_f,'fro'):.2f}")
    print(f"  ΔW (integer) rank={true_r_i}, ‖ΔW‖={np.linalg.norm(dW_i,'fro'):.1f}")
    print(f"\n  {'Metric':<25} {'Float':>16} {'Integer':>16} {'Δ':>16}")
    print(f"  {'-'*70}")
    print(f"  {'ε_full':<25} {eps_f:>16.2e} {eps_i:>16.2e} "
          f"{abs(eps_f-eps_i)/max(eps_f,eps_i)*100:>15.1f}% diff")

    for r_val in [4, 6, 8]:
        rf = [r for r in all_res_f["mincond"] if r["r"] == r_val]
        ri = [r for r in all_res_i["mincond"] if r["r"] == r_val]
        if rf and ri:
            ef = rf[0]["error"]; ei = ri[0]["error"]
            diff_pct = abs(ef-ei)/max(ef,ei)*100 if max(ef,ei)>1e-15 else 0
            print(f"  {'ε_skel(r='+str(r_val)+') mincond':<25} {ef:>16.2e} {ei:>16.2e} "
                  f"{diff_pct:>15.1f}% diff")

    rf8 = [r for r in all_res_f["mincond"] if r["r"] == 8][0]
    ri8 = [r for r in all_res_i["mincond"] if r["r"] == 8][0]
    print(f"\n  r=8, mincond: cond_f={rf8['cond_Mr']:.0f}, cond_i={ri8['cond_Mr']:.0f}")
    print(f"  CKKS encoding error ≤ 4.5e-13 vs ε_full ≈ {min(eps_f,eps_i):.1e} "
          f"(~{min(eps_f,eps_i)/4.5e-13:.0f}× larger)")

    # ── 图表 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Float vs Integer — CKKS Encoding Precision", fontsize=14, fontweight="bold")
    for ax, sname in zip(axes, ["mincond", "uniform"]):
        rs = [r["r"] for r in all_res_f[sname]]
        ef = [r["error"] for r in all_res_f[sname]]
        ei = [r["error"] for r in all_res_i[sname]]
        ax.semilogy(rs, ef, "o-", color="#e74c3c", lw=2, ms=8, label="Float")
        ax.semilogy(rs, ei, "s--", color="#2980b9", lw=2, ms=8, label="Integer")
        ax.axhline(y=TAU, color="gray", ls="--", label=f"τ={TAU}")
        ax.set_xticks(rs); ax.set_xticklabels(rs)
        ax.set_xlabel("r"); ax.set_ylabel("ε")
        ax.set_title(f"Error ({sname})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        for ri_val, efv, eiv in zip(rs, ef, ei):
            diff = abs(efv-eiv)/max(efv,eiv)*100 if max(efv,eiv)>1e-15 else 0
            if diff > 1:
                ax.annotate(f"Δ={diff:.0f}%", (ri_val, max(efv,eiv)),
                           textcoords="offset points", xytext=(0,8),
                           ha="center", fontsize=7, color="purple")
    plt.tight_layout()
    png = os.path.join(RES_DIR, "demo_float_vs_int.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Chart → {png}")

    # CSV
    rows = []
    for i in range(len(all_res_f["mincond"])):
        rows.append(dict(
            r=all_res_f["mincond"][i]["r"],
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
