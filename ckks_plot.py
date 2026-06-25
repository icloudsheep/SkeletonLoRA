"""可视化 + CSV 输出。"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ckks_config import RES_DIR, TAU

os.makedirs(RES_DIR, exist_ok=True)


def _dual_bars(ax, x, va, vb, w, rs, yl, title, ca="#2ecc71", cb="#3498db"):
    ax.bar(x - w/2, va, w, label="Hom agg", color=ca)
    ax.bar(x + w/2, vb, w, label="Shortcut", color=cb)
    ax.set_xticks(x); ax.set_xticklabels(rs)
    ax.set_ylabel(yl); ax.set_title(title); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)


def plot_results(results_a, results_b, eps_full_a, eps_full_b, true_rank, out_name):
    """2×3 对比图：流水线 A vs B。"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("CKKS Skeleton Decryption — Homomorphic vs Plaintext",
                 fontsize=14, fontweight="bold")
    ca, cb = "#2ecc71", "#3498db"
    x = np.arange(len(results_a)); rs = [r["r"] for r in results_a]; w = 0.3

    # (a) 时间 log
    ax = axes[0, 0]
    t_full = results_a[0]["time_full"]
    ax.bar(x - w/2, [r["time_skeleton"] for r in results_a], w, label="Hom agg", color=ca)
    ax.bar(x + w/2, [r["time_skeleton"] for r in results_b], w, label="Shortcut", color=cb)
    ax.bar(x[-1] + w*2, t_full, w*2, label=f"Full ({t_full:.1f}s)", color="red", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xticks(list(x) + [x[-1]+w*2]); ax.set_xticklabels(rs + ["full"])
    ax.set_ylabel("Time (s, log)"); ax.set_title("(a) Decryption Time")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # (b) 下行流量
    ax = axes[0, 1]
    _dual_bars(ax, x,
               [r["bytes_skeleton"]/1024 for r in results_a],
               [r["bytes_skeleton"]/1024 for r in results_b],
               w, rs, "KB", "(b) Downlink (Server→Decryptor)")
    ax.axhline(y=results_a[0]["bytes_full"]/1024, color="red", ls="--", lw=1,
               label=f"Full ({results_a[0]['bytes_full']/1024/1024:.0f}MB)")
    ax.legend(fontsize=7)

    # (c) 误差
    ax = axes[0, 2]
    ax.semilogy(rs, [r["error"] for r in results_a], "o-", color=ca, lw=2, ms=8, label="Hom agg")
    ax.semilogy(rs, [r["error"] for r in results_b], "s--", color=cb, lw=2, ms=8, label="Shortcut")
    ax.axhline(y=TAU, color="gray", ls="--", label=f"τ={TAU}")
    ax.set_xticks(rs); ax.set_xticklabels(rs)
    ax.set_xlabel("r"); ax.set_ylabel("ε")
    ax.set_title(f"(c) Error (rank={true_rank})"); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # (d) 加速比
    ax = axes[1, 0]
    _dual_bars(ax, x,
               [r["speedup"] for r in results_a],
               [r["speedup"] for r in results_b],
               w, rs, "×", "(d) Speedup T_full/T_skel")

    # (e) 通信节省
    ax = axes[1, 1]
    _dual_bars(ax, x,
               [r["comm_saving"] for r in results_a],
               [r["comm_saving"] for r in results_b],
               w, rs, "%", "(e) Comm Saving=(1-skel/full)×100%")

    # (f) 条件数
    ax = axes[1, 2]
    ax.semilogy(rs, [r["cond_Mr"] for r in results_a], "o-", color=ca, lw=2, ms=8, label="Hom agg")
    ax.semilogy(rs, [r["cond_Mr"] for r in results_b], "s--", color=cb, lw=2, ms=8, label="Shortcut")
    ax.set_xticks(rs); ax.set_xticklabels(rs)
    ax.set_xlabel("r"); ax.set_ylabel("cond(M_r)"); ax.set_title("(f) Condition Number")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.95])
    p = os.path.join(RES_DIR, out_name)
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Chart → {p}")


def plot_strategy_comparison(strat_results, eps_full, true_rank, out_name):
    """1×2：三种索引策略对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Index Strategy Comparison", fontsize=14, fontweight="bold")
    colors = {"mincond": "#2ecc71", "leverage": "#e74c3c", "uniform": "#f39c12"}
    markers = {"mincond": "o-", "leverage": "s--", "uniform": "^:"}

    for sname, results in strat_results.items():
        rs = [r["r"] for r in results]
        axes[0].semilogy(rs, [r["error"] for r in results], markers[sname],
                         color=colors[sname], lw=2, ms=8, label=sname)
        conds = [(r["r"], r["cond_Mr"]) for r in results if r["cond_Mr"] < float("inf")]
        if conds:
            crs, cvs = zip(*conds)
            axes[1].semilogy(crs, cvs, markers[sname], color=colors[sname], lw=2, ms=8, label=sname)

    axes[0].axhline(y=TAU, color="gray", ls="--", label=f"τ={TAU}")
    axes[0].set_xlabel("r"); axes[0].set_ylabel("ε")
    axes[0].set_title(f"(a) Error (rank={true_rank})"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("r"); axes[1].set_ylabel("cond(M_r)")
    axes[1].set_title("(b) Condition Number"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(RES_DIR, out_name)
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Chart → {p}")


def save_csv(results_a, results_b, fname):
    rows = []
    for ra, rb in zip(results_a, results_b):
        rows.append(dict(
            r=ra["r"], cond_Mr_hom=ra["cond_Mr"], cond_Mr_short=rb["cond_Mr"],
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
    sname_list = list(all_results.keys())
    for i in range(len(all_results[sname_list[0]])):
        row = {"r": all_results[sname_list[0]][i]["r"], "eps_full": eps_full}
        for sn in sname_list:
            r = all_results[sn][i]
            row[f"{sn}_error"] = r["error"]
            row[f"{sn}_cond"] = r["cond_Mr"]
            row[f"{sn}_speedup"] = r["speedup"]
        rows.append(row)
    with open(os.path.join(RES_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  CSV → {os.path.join(RES_DIR, fname)}")
