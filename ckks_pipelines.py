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


def pipeline_homomorphic_factor_mult(B_list, A_list, ctx, dim, I_idx, J_idx,
                                     n_clients, rank):
    """流水线 C：客户端只发全加密 A/B 因子，服务端密文乘法 + 聚合 + 平均。

    客户端不再本地计算 ΔW_i = B_i A_i，也不做任何聚合：只把 A_i、B_i 加密上传。
    服务端用外积分解在密文域重建骨架（仅 I_idx 行 + J_idx 列）：

        ΔW_i = Σ_c b_c ⊗ a_c   (b_c=B_i[:,c], a_c=A_i[c,:], c=0..rank-1)
        列 j: ΔW_i[:,j] = Σ_c A_i[c,j]·b_c
        行 k: ΔW_i[k,:] = Σ_c B_i[k,c]·a_c

    其中加密标量 A_i[c,j]/B_i[k,c] 由客户端以"全槽广播"形式加密上传，服务端只做
    逐元素 ct×ct 乘法（无需 galois 旋转）。跨客户端密文相加后 ×(1/N) 得密文均值。

    安全边界：本函数模拟客户端 + 服务端，二者都**不应持有私钥**。传入的 ctx 必须是
    公开 context（make_public_context 派生，is_private()==False）——它能加密、能同态
    乘加但无法解密。私钥只留在调用方的解密阶段。下面的断言强制这一边界，防止真实
    部署时误把带私钥的 context 交给服务端、导致服务端可直接解密、HE 隐私保证归零。

    返回 (enc_cols, enc_rows): list[bytes|None] × dim，骨架位置为密文，其余 None
    （格式与部分加密一致，可直接喂给 run_decryption_comparison）。
    """
    assert not ctx.is_private(), \
        "Pipeline C 的 ctx 必须是公开 context（无私钥）：客户端/服务端不得持有私钥"

    print(f"\n  Homomorphic factor multiplication "
          f"({n_clients} clients, rank={rank}, dim={dim})")
    print(f"  Skeleton: {len(I_idx)} rows + {len(J_idx)} cols (cipher ct×ct)")

    agg_cols = {j: None for j in J_idx}   # 服务端：跨客户端聚合的密文列
    agg_rows = {k: None for k in I_idx}
    t_enc, t_mul, t_add = 0.0, 0.0, 0.0

    for ci, (Bi, Ai) in enumerate(zip(B_list, A_list)):
        Bi = Bi.astype(np.float64)        # (dim, rank)
        Ai = Ai.astype(np.float64)        # (rank, dim)

        # ── 客户端本地：仅加密 A/B 因子（向量打包 + 广播标量），不做乘法/聚合 ──
        t0 = time.time()
        encB = [ts.ckks_vector(ctx, Bi[:, c].tolist()) for c in range(rank)]   # b_c
        encA = [ts.ckks_vector(ctx, Ai[c, :].tolist()) for c in range(rank)]   # a_c
        # 广播标量：列需 A_i[c,j]，行需 B_i[k,c]
        bcast_A = {j: [ts.ckks_vector(ctx, [Ai[c, j]] * dim) for c in range(rank)]
                   for j in J_idx}
        bcast_B = {k: [ts.ckks_vector(ctx, [Bi[k, c]] * dim) for c in range(rank)]
                   for k in I_idx}
        t_enc += time.time() - t0

        # ── 服务端：密文域外积重建骨架 + 跨客户端聚合 ──
        t0 = time.time()
        for j in J_idx:
            col = encB[0] * bcast_A[j][0]
            for c in range(1, rank):
                col += encB[c] * bcast_A[j][c]
            if agg_cols[j] is None: agg_cols[j] = col
            else: agg_cols[j] += col
        for k in I_idx:
            row = encA[0] * bcast_B[k][0]
            for c in range(1, rank):
                row += encA[c] * bcast_B[k][c]
            if agg_rows[k] is None: agg_rows[k] = row
            else: agg_rows[k] += row
        t_mul += time.time() - t0

        if ci == 0:
            print(f"    Client 0: enc {2*rank} factor vecs + "
                  f"{(len(I_idx)+len(J_idx))*rank} bcast scalars")

    # ── 服务端：密文域平均 ×(1/N) ──
    t0 = time.time()
    inv_n = 1.0 / n_clients
    for j in J_idx: agg_cols[j] *= inv_n
    for k in I_idx: agg_rows[k] *= inv_n
    t_add += time.time() - t0
    print(f"  Encrypt: {t_enc:.1f}s, cipher mult+add: {t_mul:.1f}s, avg: {t_add:.2f}s")

    # ── 序列化骨架密文（其余位置 None）──
    t0 = time.time(); bc = 0
    enc_cols_ser = [None] * dim
    enc_rows_ser = [None] * dim
    for j in J_idx:
        ser = agg_cols[j].serialize(); enc_cols_ser[j] = ser; bc += len(ser)
    for k in I_idx:
        ser = agg_rows[k].serialize(); enc_rows_ser[k] = ser; bc += len(ser)
    print(f"  Serialized skeleton: {bc/1024/1024:.1f} MB, {time.time()-t0:.2f}s")
    return enc_cols_ser, enc_rows_ser
