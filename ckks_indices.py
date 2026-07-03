"""索引选择策略：3 种骨架索引 + 加密列选择。"""

import numpy as np


# ── 骨架解密索引 ────────────────────────────────────────────────────────

def select_indices_mincond(delta_w, r, n_trials=20000):
    """随机采样 + 条件数最小化（默认，需明文 ΔW）。"""
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
            best_cond = c; best = (Ic, Jc)
    return best[0], best[1], best_cond


def select_indices_leverage(delta_w, r):
    """SVD leverage scores 基线（需明文 ΔW）。"""
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
    """均匀间隔采样（无需明文 ΔW，安全性最强）。"""
    I_r = np.linspace(0, m - 1, r, dtype=int)
    J_r = np.linspace(0, n - 1, r, dtype=int)
    return I_r, J_r, float("inf")


def uniform_index_union(dim, r_values):
    """r 扫描下所有 uniform 骨架索引的并集（行/列各一份）。

    供 Pipeline C 预先确定服务端需密文计算哪些行列：因 uniform 索引客户端事先
    可知，取并集即覆盖整个 r-sweep 所需的全部骨架位置。返回 (I_union, J_union)。
    """
    rows, cols = set(), set()
    for r in r_values:
        I_r, J_r, _ = select_indices_uniform(dim, dim, r)
        rows.update(int(i) for i in I_r)
        cols.update(int(j) for j in J_r)
    return np.array(sorted(rows)), np.array(sorted(cols))


# ── 部分加密索引 ──────────────────────────────────────────────────────────

def select_encrypt_indices(delta_w, ratio):
    """基于全局 L2 范数选择加密行列。所有客户端加密相同的 top-ratio 子集。"""
    dim = delta_w.shape[0]
    k = max(1, int(dim * ratio))
    col_norms = np.linalg.norm(delta_w, axis=0)
    row_norms = np.linalg.norm(delta_w, axis=1)
    enc_cols = np.sort(np.argsort(col_norms)[-k:])
    enc_rows = np.sort(np.argsort(row_norms)[-k:])
    return enc_rows, enc_cols
