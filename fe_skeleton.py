"""骨架索引选择与 CUR 重建工具。

骨架解密的思想：聚合矩阵 ΔW 秩很低（≤ N·RANK），只需解密 r 行 + r 列即可用
CUR 分解重建整个 d×d 矩阵，从而把解密量从 d 条密文降到 2r 条。

    ΔW_rec = C_r · M_r^{-1} · R_r

其中 C_r 是 r 个解密列（d×r），R_r 是 r 个解密行（r×d），M_r 是二者交叉块（r×r）。

索引策略只提供 uniform（均匀间隔）：mincond/leverage 需要明文 ΔW 才能挑索引，
而本场景中聚合结果始终为密文、无人持有明文 ΔW，故只有客户端可自行推算的 uniform
可用。这不是精度最优策略，而是密钥边界下的唯一可行选择。
"""

import numpy as np


def select_uniform_indices(dim, r):
    """在 [0, dim) 上均匀间隔取 r 个行索引与 r 个列索引。

    无需访问矩阵内容，仅凭维度与 r 即可确定，因此客户端与服务端可各自算出同一组
    索引、无需协商，天然契合「服务端只见密文」的约束。

    :param dim: 矩阵维度 d
    :param r: 骨架规模（行数 = 列数 = r）
    :return: (I_r, J_r)，各为长度 r 的升序索引数组
    """
    rows, cols = select_uniform_rect_indices(dim, dim, r)
    return rows, cols


def select_uniform_rect_indices(n_rows, n_cols, r):
    """分别在矩形矩阵的行维度和列维度上选择 uniform 索引。"""
    if n_rows < r or n_cols < r:
        raise ValueError(
            f"矩阵维度 ({n_rows}, {n_cols}) 小于 skeleton rank={r}"
        )
    rows = np.linspace(0, n_rows - 1, r, dtype=int)
    cols = np.linspace(0, n_cols - 1, r, dtype=int)
    if len(np.unique(rows)) != r or len(np.unique(cols)) != r:
        raise ValueError(f"uniform 索引重复：shape=({n_rows}, {n_cols}), rank={r}")
    return rows, cols


def union_indices(dim, r_values):
    """求一组 r 值下所有 uniform 骨架索引的并集（行、列各一份）。

    服务端只需对并集覆盖的行列做密文重建，即可支撑整个 r 扫描的解密需求，
    避免每个 r 重复计算。

    :param dim: 矩阵维度 d
    :param r_values: 待扫描的骨架规模列表
    :return: (I_union, J_union)，升序索引数组
    """
    rows, cols = set(), set()
    for r in r_values:
        I_r, J_r = select_uniform_indices(dim, r)
        rows.update(int(i) for i in I_r)
        cols.update(int(j) for j in J_r)
    return np.array(sorted(rows)), np.array(sorted(cols))


def union_rect_indices(n_rows, n_cols, r_values):
    """求多个 skeleton rank 在矩形行列维度上的 uniform 索引并集。"""
    rows, cols = set(), set()
    for r in r_values:
        row_idx, col_idx = select_uniform_rect_indices(n_rows, n_cols, r)
        rows.update(int(i) for i in row_idx)
        cols.update(int(j) for j in col_idx)
    return np.array(sorted(rows)), np.array(sorted(cols))


def cur_reconstruct(C_r, R_r, I_r, J_r):
    """由骨架行列用 CUR 公式重建完整矩阵。

    交叉块 M_r 取自已解密的行集合 R_r 在列索引 J_r 上的切片，因此重建只依赖
    已解密数据，无需再触碰其余密文。

    :param C_r: 解密得到的 r 个列，形状 (dim, r)，列顺序对应 J_r
    :param R_r: 解密得到的 r 个行，形状 (r, dim)，行顺序对应 I_r
    :param I_r: 行索引数组，长度 r，用于在 R_r 中定位交叉块的行
    :param J_r: 列索引数组，长度 r，用于在 R_r 中定位交叉块的列
    :return: (dW_rec, ok)。ok 为 False 表示交叉块降秩、无法求逆，dW_rec 为 None
    """
    dW_rec, ok, _ = cur_reconstruct_with_stats(C_r, R_r, I_r, J_r)
    return dW_rec, ok


def cur_reconstruct_with_stats(
    C_r,
    R_r,
    I_r,
    J_r,
    condition_threshold=1e12,
):
    """执行矩形 CUR，并返回可解释的数值稳定性统计。"""
    C_r = np.asarray(C_r, dtype=np.float64)
    R_r = np.asarray(R_r, dtype=np.float64)
    I_r = np.asarray(I_r, dtype=int)
    J_r = np.asarray(J_r, dtype=int)
    r = len(J_r)
    stats = {
        "rank": r,
        "numerical_rank": None,
        "max_singular_value": None,
        "min_singular_value": None,
        "condition_number": None,
        "inverse_method": None,
        "ok": False,
        "failure_reason": "",
    }
    if C_r.ndim != 2 or R_r.ndim != 2 or C_r.shape[1] != r or R_r.shape[0] != r:
        stats["failure_reason"] = (
            f"骨架形状不匹配：C={C_r.shape}，R={R_r.shape}，r={r}"
        )
        return None, False, stats
    if len(I_r) != r or C_r.shape[0] <= max(I_r, default=-1):
        stats["failure_reason"] = "行索引与 C 形状不匹配"
        return None, False, stats
    M_r = R_r[:, J_r]
    try:
        singular_values = np.linalg.svd(M_r, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        stats["failure_reason"] = f"交叉块 SVD 失败：{exc}"
        return None, False, stats
    if singular_values.size:
        stats["max_singular_value"] = float(singular_values[0])
        stats["min_singular_value"] = float(singular_values[-1])
        stats["condition_number"] = float(np.linalg.cond(M_r))
        stats["numerical_rank"] = int(np.linalg.matrix_rank(M_r))
    try:
        if not np.isfinite(stats["condition_number"]):
            raise np.linalg.LinAlgError("交叉块条件数非有限")
        if stats["condition_number"] > condition_threshold:
            inverse = np.linalg.pinv(M_r)
            stats["inverse_method"] = "pinv"
        else:
            inverse = np.linalg.inv(M_r)
            stats["inverse_method"] = "inv"
    except np.linalg.LinAlgError:
        try:
            inverse = np.linalg.pinv(M_r)
            stats["inverse_method"] = "pinv"
        except np.linalg.LinAlgError as exc:
            stats["failure_reason"] = f"交叉块求逆失败：{exc}"
            return None, False, stats
    dW_rec = C_r @ inverse @ R_r
    stats["ok"] = bool(np.all(np.isfinite(dW_rec)))
    if not stats["ok"]:
        stats["failure_reason"] = "CUR 输出包含非有限值"
        return None, False, stats
    return dW_rec, True, stats
