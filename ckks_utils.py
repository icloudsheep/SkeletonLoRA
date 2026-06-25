"""工具：数据加载、误差计算、CKKS 上下文、加解密。"""

import os, time
import numpy as np
import tenseal as ts
from safetensors import safe_open
from ckks_config import TEMP_OUTPUT_DIR, POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES, GLOBAL_SCALE


def load_weights(n_clients, layer="q_proj", dim=None):
    """返回 (B_list, A_list)。每项 shape (dim,4) 和 (4,dim)。"""
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


def create_ckks_context():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=POLY_MODULUS_DEGREE,
                     coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    return ctx


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


def resolve_col(col_idx, enc_cols, plain_cols, ctx):
    """获取一列：加密列解密，明文列直接读。"""
    if enc_cols is not None and enc_cols[col_idx] is not None:
        rows, t, b = decrypt_selected(enc_cols, [col_idx], ctx)
        return rows[0], t, b
    return plain_cols[:, col_idx], 0.0, 0


def resolve_row(row_idx, enc_rows, plain_rows, ctx):
    """获取一行：加密行解密，明文行直接读。"""
    if enc_rows is not None and enc_rows[row_idx] is not None:
        rows, t, b = decrypt_selected(enc_rows, [row_idx], ctx)
        return rows[0], t, b
    return plain_rows[row_idx, :], 0.0, 0
