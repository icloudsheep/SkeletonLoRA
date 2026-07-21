"""full、partial 和明文 baseline 的矩形分块规则。"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ModePartition:
    """一个 AB 对的行列加密分区。"""

    mode: str
    ratio: float | None
    encrypted_rows: np.ndarray
    plain_rows: np.ndarray
    encrypted_cols: np.ndarray
    plain_cols: np.ndarray

    @property
    def output_encrypted_rows(self):
        return self.encrypted_rows

    @property
    def output_encrypted_cols(self):
        return self.encrypted_cols


def _prefix_count(length, ratio):
    if ratio < 0 or ratio > 100:
        raise ValueError(f"比例必须在 [0, 100]，实际为 {ratio}")
    return min(length, math.ceil(length * ratio / 100))


def build_partition(out_features, in_features, mode, ratio=None):
    """按模式构造 B 行和 A 列的加密索引。"""
    if mode == "plain_baseline":
        row_count = col_count = 0
    elif mode == "full":
        row_count = out_features
        col_count = in_features
    elif mode in {"partial_A", "partial_AB"}:
        if ratio is None:
            raise ValueError(f"模式 {mode} 需要 ratio")
        row_count = out_features if mode == "partial_AB" else 0
        col_count = _prefix_count(in_features, ratio)
        if mode == "partial_AB":
            row_count = _prefix_count(out_features, ratio)
    else:
        raise ValueError(f"未知实验模式：{mode}")
    rows = np.arange(out_features, dtype=int)
    cols = np.arange(in_features, dtype=int)
    return ModePartition(
        mode=mode,
        ratio=ratio,
        encrypted_rows=rows[:row_count],
        plain_rows=rows[row_count:],
        encrypted_cols=cols[:col_count],
        plain_cols=cols[col_count:],
    )


def encrypted_output_mask(partition, shape):
    """返回最终 BA 中需要密文表示的坐标 mask。"""
    n_rows, n_cols = shape
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    if partition.encrypted_rows.size:
        mask[partition.encrypted_rows, :] = True
    if partition.encrypted_cols.size:
        mask[:, partition.encrypted_cols] = True
    return mask


def plain_mixed_product(B, A, partition):
    """计算 partial 的明文参考，并返回明文/密文坐标掩码。"""
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    if B.ndim != 2 or A.ndim != 2 or B.shape[1] != A.shape[0]:
        raise ValueError(f"A/B 形状不匹配：A={A.shape}，B={B.shape}")
    result = B @ A
    return result, encrypted_output_mask(partition, result.shape)


def mean_reference(pairs, n_clients, scaling=1.0):
    """按固定客户端数计算一个 AB 对的等权明文参考。"""
    if not pairs:
        raise ValueError("pairs 不能为空")
    first = pairs[0]
    result = np.zeros((first.b.shape[0], first.a.shape[1]), dtype=np.float64)
    for pair in pairs:
        result += (pair.b.astype(np.float64) @ pair.a.astype(np.float64)) * scaling
    return result / n_clients


def error_summary(actual, reference, mask=None):
    """返回全局及可选区域的误差摘要。"""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"误差输入形状不一致：actual={actual.shape}，reference={reference.shape}")
    diff = actual - reference
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != actual.shape:
            raise ValueError(f"mask 形状不一致：mask={mask.shape}，actual={actual.shape}")
        diff = diff[mask]
        ref_values = reference[mask]
    else:
        diff = diff.reshape(-1)
        ref_values = reference.reshape(-1)
    ref_norm = np.linalg.norm(ref_values)
    abs_diff = np.abs(diff)
    return {
        "relative_frobenius_error": float(np.linalg.norm(diff) / ref_norm)
        if ref_norm else float("inf"),
        "max_absolute_error": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_absolute_error": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "element_count": int(diff.size),
    }
