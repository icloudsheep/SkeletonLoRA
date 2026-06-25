"""流水线：同态聚合（A）、明文捷径（B）、部分加密。"""

import time
import numpy as np
import tenseal as ts
from ckks_utils import encrypt_vectors


def pipeline_homomorphic_aggregation(B_list, A_list, ctx, dim):
    """流水线 A：每个客户端本地加密 ΔW_i → 服务端 CKKS 加法聚合。

    服务端全程只接触密文。返回 (enc_cols, enc_rows): list[bytes] × dim。
    """
    n = len(B_list)
    print(f"\n  Homomorphic aggregation ({n} clients, dual encoding, dim={dim})")
    agg_cols, agg_rows = None, None
    t_enc, t_add = 0.0, 0.0

    for ci, (Bi, Ai) in enumerate(zip(B_list, A_list)):
        print(f"\n  Client {ci}:")
        dWi = Bi.astype(np.float64) @ Ai.astype(np.float64)

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

    print(f"\n  Encrypt: {t_enc:.1f}s, add: {t_add:.1f}s")
    t0 = time.time()
    ec = [v.serialize() for v in agg_cols]
    er = [v.serialize() for v in agg_rows]
    bc = sum(len(s) for s in ec); br = sum(len(s) for s in er)
    print(f"  Serialized: {(bc+br)/1024/1024:.1f} MB, {time.time()-t0:.2f}s")
    return ec, er


def pipeline_plaintext_shortcut(B_list, A_list, ctx, dim):
    """流水线 B：明文聚合后加密 ΔW（对照基线，隔离同态噪声）。"""
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


def pipeline_partial_encryption(B_list, A_list, ctx, dim, encrypt_ratio,
                                 enc_rows_idx, enc_cols_idx):
    """部分加密：所有客户端加密相同行列子集，其余明文累加。

    返回 (enc_cols, enc_rows, plain_cols, plain_rows)。
    enc_*: list[bytes | None]; plain_*: np.ndarray，加密位置为 0。
    """
    n = len(B_list)
    k = len(enc_cols_idx)
    print(f"\n  Partial encryption ({n} clients, ratio={encrypt_ratio:.0%}, "
          f"k={k}/{dim}, dual encoding)")

    agg_enc_cols = [None] * dim
    agg_enc_rows = [None] * dim
    agg_plain_cols = np.zeros((dim, dim), dtype=np.float64)
    agg_plain_rows = np.zeros((dim, dim), dtype=np.float64)
    t_enc, t_add_enc, t_add_plain = 0.0, 0.0, 0.0

    enc_col_set = set(enc_cols_idx)
    enc_row_set = set(enc_rows_idx)

    for ci, (Bi, Ai) in enumerate(zip(B_list, A_list)):
        dWi = Bi.astype(np.float64) @ Ai.astype(np.float64)

        t0 = time.time()
        client_enc_cols = {j: ts.ckks_vector(ctx, dWi[:, j].tolist())
                           for j in enc_cols_idx}
        client_enc_rows = {k: ts.ckks_vector(ctx, dWi[k, :].tolist())
                           for k in enc_rows_idx}
        dt_enc = time.time() - t0; t_enc += dt_enc

        t0 = time.time()
        for j, enc_val in client_enc_cols.items():
            if agg_enc_cols[j] is None: agg_enc_cols[j] = enc_val
            else: agg_enc_cols[j] += enc_val
        for k, enc_val in client_enc_rows.items():
            if agg_enc_rows[k] is None: agg_enc_rows[k] = enc_val
            else: agg_enc_rows[k] += enc_val
        dt_add_e = time.time() - t0; t_add_enc += dt_add_e

        t0 = time.time()
        for j in range(dim):
            if j not in enc_col_set: agg_plain_cols[:, j] += dWi[:, j]
        for k in range(dim):
            if k not in enc_row_set: agg_plain_rows[k, :] += dWi[k, :]
        t_add_plain += time.time() - t0

        if ci == 0:
            print(f"    Client 0: enc {len(enc_rows_idx)}r/{len(enc_cols_idx)}c "
                  f"({dt_enc:.2f}s enc)")

    print(f"  Encrypt: {t_enc:.1f}s, add_enc: {t_add_enc:.1f}s, "
          f"add_plain: {t_add_plain:.3f}s")

    t0 = time.time(); bc, br = 0, 0
    enc_cols_ser = [None] * dim; enc_rows_ser = [None] * dim
    for j in enc_cols_idx:
        ser = agg_enc_cols[j].serialize(); enc_cols_ser[j] = ser; bc += len(ser)
    for k in enc_rows_idx:
        ser = agg_enc_rows[k].serialize(); enc_rows_ser[k] = ser; br += len(ser)
    print(f"  Serialized: {bc/1024/1024:.1f}(col)+{br/1024/1024:.1f}(row) MB, "
          f"{time.time()-t0:.2f}s")
    return enc_cols_ser, enc_rows_ser, agg_plain_cols, agg_plain_rows
