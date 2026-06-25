"""解密对比：全量 vs 骨架（多策略 + 部分加密兼容）。"""

import time
import numpy as np
from ckks_config import R_VALUES, TAU
from ckks_utils import decrypt_selected, resolve_col, resolve_row, relative_error
from ckks_indices import select_indices_mincond


def run_decryption_comparison(delta_w_plain, enc_cols, enc_rows, ctx,
                               label, true_rank, dim, strategies=None,
                               r_values=None, plain_cols=None, plain_rows=None):
    """全量 + 骨架解密对比。

    plain_cols/plain_rows: 部分加密模式（None = 纯 CKKS 解密）。
    strategies: dict of {name: callable(dw, r)->(I, J, cond)}.
    """
    if strategies is None:
        strategies = {"mincond": lambda dw, r: select_indices_mincond(dw, r)}
    if r_values is None:
        r_values = R_VALUES
    is_partial = plain_cols is not None

    print(f"\n{'─'*60}")
    tag = " (partial enc)" if is_partial else ""
    print(f"  [{label}] Decryption comparison{tag}  (dim={dim})")
    print(f"{'─'*60}")

    # ── 全量解密基线 ──
    print("  Full decryption (all columns) ...")
    t0 = time.time()
    if is_partial:
        cols = [resolve_col(j, enc_cols, plain_cols, ctx)[0] for j in range(dim)]
        _full_mat = np.column_stack(cols)
        t_full = time.time() - t0; b_full = 0  # partial 不计 bytes
    else:
        _colmat, t_full, b_full = decrypt_selected(enc_cols, list(range(dim)), ctx)
        _full_mat = _colmat.T
    eps_full = relative_error(_full_mat, delta_w_plain)
    print(f"    t={t_full:.2f}s, bytes={b_full/1024/1024:.1f}MB, ε_full={eps_full:.2e}")

    # ── 骨架解密 ──
    max_r = min(dim, true_rank)
    rs = [r for r in r_values if r <= max_r]
    all_results = {}

    for sname, sfunc in strategies.items():
        print(f"\n  Index strategy: {sname}")
        results = []
        for r_val in rs:
            I_r, J_r, cond = sfunc(delta_w_plain, r_val)

            C_cols, t_cols, b_cols = [], 0.0, 0
            for j in J_r:
                vec, t, b = resolve_col(j, enc_cols, plain_cols, ctx)
                C_cols.append(vec); t_cols += t; b_cols += b
            C_r = np.column_stack(C_cols)

            R_rows, t_rows, b_rows = [], 0.0, 0
            for k in I_r:
                vec, t, b = resolve_row(k, enc_rows, plain_rows, ctx)
                R_rows.append(vec); t_rows += t; b_rows += b
            R_r = np.array(R_rows)

            M_r = R_r[:, J_r]
            if np.linalg.matrix_rank(M_r) < r_val:
                print(f"    r={r_val:2d}: rank(M_r) < {r_val}, skip")
                continue

            dW_rec = C_r @ np.linalg.inv(M_r) @ R_r
            eps = relative_error(dW_rec, delta_w_plain)
            t_skel = max(t_cols + t_rows, 1e-9)
            b_skel = b_cols + b_rows

            n_enc_hit = sum(1 for j in J_r if (enc_cols is not None and enc_cols[j] is not None)) + \
                        sum(1 for k in I_r if (enc_rows is not None and enc_rows[k] is not None))
            n_hit = len(J_r) + len(I_r)

            results.append(dict(
                r=r_val, cond_Mr=cond, error=eps,
                time_skeleton=t_skel, time_full=t_full,
                bytes_skeleton=b_skel, bytes_full=b_full,
                speedup=t_full / t_skel if t_skel > 1e-6 else float("inf"),
                comm_saving=(1 - b_skel / b_full) * 100 if b_full > 0 else 0,
                enc_hit_ratio=n_enc_hit / n_hit if n_hit > 0 else 0,
            ))
            status = "✓" if eps < TAU else "✗"
            enc_info = f"enc_hit={n_enc_hit}/{n_hit}" if is_partial else ""
            spd = t_full / t_skel if t_skel > 1e-6 else float("inf")
            print(f"    r={r_val:2d}: ε={eps:.2e} {status}  cond={cond:.0f}  "
                  f"t_skel={t_skel:.3f}s  speedup={spd:.0f}×  {enc_info}")
        all_results[sname] = results

    return all_results, eps_full
