#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""联邦 LoRA · CKKS 密文聚合实验 — 主编排入口。

全流程由本文件的 main 独占编排，其余模块只提供无状态工具，参数一律由 main 传入。
审查者只需关注本文件、结果 CSV 与结果图。

四个正交实验维度（笛卡尔积 16 组）：
  计算方法 method  : 外积（逐元素乘加重建 ΔW，无 galois）/ 内积（标准矩阵乘法，
                     matmul/dot 跨槽位求和，需 galois）
  打包方式 packing : packed（多列共享一条密文槽位，省网络）/ unpacked（每列独立密文）
  加密程度 enc     : full（A、B 都加密 → 密文×密文）/ half（仅加密 A → 密文×明文）
  骨架优化 skeleton: True（仅重建 uniform 骨架行列 + CUR）/ False（重建完整 d×d）

对比重点：
  - 每种方法内部：打包/骨架/全半加密的时间与网络开销对比并作图。
  - 内积 vs 外积：量化外积法用「因子 repeat/tile 到 dim 长」的空间冗余换掉「跨槽位
    求和」所付出的代价——上传冗余比 = 外积上传/内积上传、服务端时间比 = 外积/内积。
  - 内积-全加密单格若超时（O(d²·N) 次 ct×ct dot），如实标注「不可行」，绝不编数字。

用法：
  python main.py --sweep                       # 跑全部 16 组 + 对比 CSV/图 + 冗余表（推荐）
  python main.py --method 外积 --packing packed --enc full --skeleton   # 单组配置
  python main.py --sweep --real                # 用 temp_output_dir 真实权重
"""

import time
import argparse
import numpy as np

from fe_config import (N_CLIENTS, RANK, DIM, TEMP_OUTPUT_DIR,
                       POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES, GLOBAL_SCALE,
                       RES_DIR, CSV_PATH, SWEEP_CSV_PATH, REDUNDANCY_CSV_PATH,
                       METHODS, PACKING_MODES, ENC_LEVELS, SKELETON_MODES, SKELETON_R,
                       INNER_FULL_TIME_BUDGET,
                       PLOT_PACKED, PLOT_UNPACKED, PLOT_REDUNDANCY)
from fe_context import create_secret_context, derive_public_context
from fe_client import encrypt_upload, upload_bytes, decrypt_groups
from fe_server import aggregate, download_bytes
from fe_inner import (encrypt_upload_inner, upload_bytes_inner,
                      aggregate_inner, decrypt_cols_inner, decrypt_rows_inner)
from fe_skeleton import union_indices, select_uniform_indices, cur_reconstruct
from fe_metrics import MetricsCollector, SweepTable, RedundancyTable
from fe_plot import plot_baseline, plot_redundancy


# ── 数据准备 ────────────────────────────────────────────────────────────

def gen_random_factors(n_clients, dim, rank, seed=42):
    """生成随机 LoRA 因子（无权重时验证全流程）。幅度 0.05 贴近真实 LoRA 增量量级。"""
    rng = np.random.RandomState(seed)
    B_list = [rng.randn(dim, rank) * 0.05 for _ in range(n_clients)]
    A_list = [rng.randn(rank, dim) * 0.05 for _ in range(n_clients)]
    return B_list, A_list


def load_real_factors(n_clients, dim, layer="q_proj"):
    """从 temp_output_dir 加载真实 LoRA 权重并截断到 dim 维。"""
    import os
    from safetensors import safe_open
    B_list, A_list = [], []
    for cid in range(n_clients):
        p = os.path.join(TEMP_OUTPUT_DIR,
                         f"client_{cid}_output/final_lora/adapter_model.safetensors")
        with safe_open(p, framework="np") as f:
            bk = ak = None
            for k in f.keys():
                if layer in k:
                    if "lora_B" in k:
                        bk = k
                    elif "lora_A" in k:
                        ak = k
            B_list.append(f.get_tensor(bk)[:dim, :])
            A_list.append(f.get_tensor(ak)[:, :dim])
    return B_list, A_list


def plaintext_reference(B_list, A_list, n_clients):
    """明文参考 ΔW_mean = (1/N)·Σ_i B_i·A_i，作为精度真值。"""
    dim = B_list[0].shape[0]
    dw = np.zeros((dim, dim), dtype=np.float64)
    for Bi, Ai in zip(B_list, A_list):
        dw += Bi.astype(np.float64) @ Ai.astype(np.float64)
    dw /= n_clients
    return dw, int(np.linalg.matrix_rank(dw))


def relative_error(rec, ref):
    """Frobenius 相对误差 ‖rec-ref‖/‖ref‖。"""
    denom = np.linalg.norm(ref, "fro")
    return float(np.linalg.norm(rec - ref, "fro") / denom) if denom else float("inf")


# ── 单配置运行（供 sweep 与单跑复用） ───────────────────────────────────────

def _skeleton_indices(dim, skeleton):
    """骨架开：uniform 骨架行列并集；关：仅列全索引（行不用重建）。

    骨架关时 _reconstruct_error 直接用列拼完整 d×d 矩阵，行阶段全部产物
    （上传/服务端聚合/下载/解密）都会被丢弃，故此处令 I_idx=空，让 client/
    server 天然跳过整条行链路——不改动服务端聚合逻辑即可省一半时间和网络。
    """
    if skeleton:
        return union_indices(dim, [min(SKELETON_R, min(dim, N_CLIENTS * RANK))])
    return np.array([], dtype=int), np.arange(dim)


def _reconstruct_error(dec_cols, dec_rows, delta_w_mean, dim, skeleton, I_idx, J_idx):
    """由解密的行列重建并算相对误差。骨架经 CUR，完整直接拼列。

    :return: (error, feasible)。CUR 交叉块降秩时 (None, False)。
    """
    if skeleton:
        # 按预定骨架索引 I_idx/J_idx 选取，而非取解密字典全部键：packed 模式服务端会
        # 重建全部列（块对角覆盖整组），dec_cols 含超出骨架的列，直接全取会使 C_r 宽于
        # M_r 行数、交叉块非方阵而重建失败。CUR 只需骨架 r 行 + r 列。
        I_r = np.array(sorted(int(k) for k in I_idx))
        J_r = np.array(sorted(int(j) for j in J_idx))
        C_r = np.column_stack([dec_cols[int(j)] for j in J_r])
        R_r = np.array([dec_rows[int(k)] for k in I_r])
        dW_rec, ok = cur_reconstruct(C_r, R_r, I_r, J_r)
        if not ok:
            return None, False
        return relative_error(dW_rec, delta_w_mean), True
    full_mat = np.column_stack([dec_cols[j] for j in range(dim)])
    return relative_error(full_mat, delta_w_mean), True


def _run_outer(B_list, A_list, delta_w_mean, dim, n_slots, secret_ctx, public_ctx,
               packing, enc, skeleton):
    """外积法：客户端 repeat/tile 因子加密上传 → 服务端逐元素乘加聚合 → 解密重建。"""
    I_idx, J_idx = _skeleton_indices(dim, skeleton)

    t0 = time.time()
    uploads = [encrypt_upload(B, A, public_ctx, RANK, dim, I_idx, J_idx, enc, packing,
                              n_slots)
               for B, A in zip(B_list, A_list)]
    t_enc = (time.time() - t0) / len(B_list)
    up_one = upload_bytes(uploads[0])

    t0 = time.time()
    col_bytes, row_bytes = aggregate(uploads, public_ctx, len(B_list))
    t_server = time.time() - t0
    down_one = download_bytes(col_bytes) + download_bytes(row_bytes)

    t0 = time.time()
    dec_cols = decrypt_groups(col_bytes, secret_ctx, dim)
    dec_rows = decrypt_groups(row_bytes, secret_ctx, dim)
    t_dec = time.time() - t0

    error, feasible = _reconstruct_error(dec_cols, dec_rows, delta_w_mean,
                                         dim, skeleton, I_idx, J_idx)
    return dict(t_client_enc=t_enc, t_client_dec=t_dec, t_server=t_server,
                up_one=up_one, down_one=down_one, error=error, feasible=feasible,
                note="骨架 CUR 重建" if skeleton else "完整重建")


def _run_inner(B_list, A_list, delta_w_mean, dim, n_slots, secret_ctx, public_ctx,
               packing, enc, skeleton):
    """内积法：客户端加密 A/B → 服务端 matmul/dot 跨槽位求和 → 解密重建。

    打包（防退化，仅 half 可行）：把 A 全列拼进少量密文，服务端用块对角明文矩阵一次
    matmul 还原整组，列/行两阶段共用打包 A，上传密文条数从 dim 降到 ceil(dim/g)。
    full+packed 不可行：ct×ct 无块内求和手段，直接标注不可行、不做假打包。
    full 内积需 O(d²·N) 次 ct×ct dot，设时间预算，超则标注不可行。
    """

    # full + packed：物理不可行，如实标注，不编造数字、不做假打包。
    if enc == "full" and packing == "packed":
        return dict(t_client_enc=0.0, t_client_dec=0.0, t_server=0.0,
                    up_one=0, down_one=0, error=None, feasible=False,
                    note="full 加密 ct×ct 无块内求和手段，块对角打包不可行")

    I_idx, J_idx = _skeleton_indices(dim, skeleton)

    t0 = time.time()
    uploads = [encrypt_upload_inner(B, A, public_ctx, RANK, dim, enc, packing, n_slots)
               for B, A in zip(B_list, A_list)]
    t_enc = (time.time() - t0) / len(B_list)
    up_one = upload_bytes_inner(uploads[0])

    budget = INNER_FULL_TIME_BUDGET if enc == "full" else None
    t0 = time.time()
    agg = aggregate_inner(uploads, public_ctx, len(B_list), RANK, dim,
                          I_idx, J_idx, enc, packing, time_budget=budget)
    t_server = time.time() - t0

    if not agg["feasible"]:
        return dict(t_client_enc=t_enc, t_client_dec=0.0, t_server=t_server,
                    up_one=up_one, down_one=0, error=None, feasible=False,
                    note=agg["note"])

    down_one = agg["down_bytes"]
    n = len(B_list)
    t0 = time.time()
    dec_cols = decrypt_cols_inner(agg["col_bytes"], secret_ctx, dim, n)
    dec_rows = decrypt_rows_inner(agg["row_bytes"], secret_ctx, dim, n)
    t_dec = time.time() - t0

    error, feasible = _reconstruct_error(dec_cols, dec_rows, delta_w_mean,
                                         dim, skeleton, I_idx, J_idx)
    note = "块对角打包" if packing == "packed" else "逐列不打包"
    return dict(t_client_enc=t_enc, t_client_dec=t_dec, t_server=t_server,
                up_one=up_one, down_one=down_one, error=error, feasible=feasible,
                note=note)


def run_config(B_list, A_list, delta_w_mean, dim, n_slots, secret_ctx, public_ctx,
               method, packing, enc, skeleton, metrics=None):
    """按方法分派跑一组 (method, packing, enc, skeleton) 配置，返回端到端指标 dict。

    :param method: "外积" / "内积"
    :param n_slots: 单条密文槽位数（= poly_modulus_degree/2），由 _prepare 依 dim 决定
    :param metrics: 可选 MetricsCollector，非空时记录分阶段明细
    :return: dict(t_client_enc, t_client_dec, t_server, up_one, down_one,
                  error, feasible, note)
    """
    if method == "外积":
        res = _run_outer(B_list, A_list, delta_w_mean, dim, n_slots,
                         secret_ctx, public_ctx, packing, enc, skeleton)
    else:
        res = _run_inner(B_list, A_list, delta_w_mean, dim, n_slots,
                         secret_ctx, public_ctx, packing, enc, skeleton)

    if metrics is not None:
        tag = f"{method}/{packing}/{enc}/骨架{'开' if skeleton else '关'}"
        metrics.add(f"客户端加密-{tag}", seconds=res["t_client_enc"],
                    bytes_transferred=res["up_one"], note="单客户端平均加密 + 单次上传")
        metrics.add(f"服务端聚合-{tag}", seconds=res["t_server"],
                    bytes_transferred=res["down_one"], note="密文域聚合 + 单次下载")
        eps_note = "N/A" if res["error"] is None else f"{res['error']:.3e}"
        metrics.add(f"客户端解密-{tag}", seconds=res["t_client_dec"],
                    note=f"误差 ε={eps_note}")
    return res


# ── 主编排 ────────────────────────────────────────────────────────────────

def _pick_ckks_params(dim):
    """依据 dim 选 CKKS 参数：dim ≤ 4096 用 8192 度小 context 省时；否则升到 16384。

    tenseal 要求 dim ≤ slots = poly_modulus_degree/2；且外积法上传是 tile(B,m) 放进
    单条密文，长度 = m·dim，故实际约束是 m·dim ≤ slots。这里 m 取 1 保守估算，仅
    保证向量本身放得下——外积 packed 在 dim ≥ slots 时会自然退化到 m=1（即 unpacked）。
    """
    if dim <= 4096:
        return POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES, GLOBAL_SCALE
    # dim>4096（真实 3200 不进这条，但为将来 7B 模型 4096 维预留）：升到 16384 度、加深模数链。
    return 16384, [60, 40, 40, 40, 60], 2 ** 40


def _prepare(use_real, dim):
    """建密钥 + 备数据。

    建两套密钥：外积法用无 galois 的 context（逐元素乘加，省开销）；内积法用带 galois
    的 context（matmul/dot 跨槽位求和所需）。二者各自派生公开 context 交服务端。

    :return: (ctx_outer, pub_outer, ctx_inner, pub_inner, B_list, A_list, dW_mean,
              dim, n_slots)
    """
    poly_deg, mod_bits, scale = _pick_ckks_params(dim)
    n_slots = poly_deg // 2
    print(f"  CKKS 参数：poly_modulus_degree={poly_deg} → 单密文 {n_slots} slot；"
          f"coeff_mod_bit_sizes={mod_bits}")
    ctx_outer = create_secret_context(poly_deg, mod_bits, scale, galois=False)
    ctx_inner = create_secret_context(poly_deg, mod_bits, scale, galois=True)
    pub_outer = derive_public_context(ctx_outer)
    pub_inner = derive_public_context(ctx_inner)
    assert ctx_outer.is_private() and not pub_outer.is_private()
    assert ctx_inner.is_private() and not pub_inner.is_private()

    if use_real:
        B_list, A_list = load_real_factors(N_CLIENTS, dim)
    else:
        B_list, A_list = gen_random_factors(N_CLIENTS, dim, RANK)
    dW_mean, true_rank = plaintext_reference(B_list, A_list, N_CLIENTS)
    print(f"  数据就绪：B={B_list[0].shape}, A={A_list[0].shape}, "
          f"ΔW_mean 秩={true_rank}, 维度 d={dim}")
    return (ctx_outer, pub_outer, ctx_inner, pub_inner,
            B_list, A_list, dW_mean, dim, n_slots)


def _ctx_for(method, ctx_outer, pub_outer, ctx_inner, pub_inner):
    """按方法选对应的 (secret_ctx, public_ctx)。"""
    if method == "外积":
        return ctx_outer, pub_outer
    return ctx_inner, pub_inner


def _prefix_name(name, use_real, dim=None):
    """按数据来源与维度给产物加后缀，避免覆盖 demo/其他 dim 的结果。

    demo 64（默认 dim）保留原名；其余按 `_真实_d{dim}` 或 `_d{dim}` 命名。

    :param name: 基础文件名，可能含扩展名（.csv）也可能没有
    :param use_real: True 表示当前跑真实数据
    :param dim: 当前维度；None 表示默认 DIM，不加维度后缀
    :return: 带来源/维度标记的文件名
    """
    suffix = ""
    if use_real:
        suffix += "_真实"
    if dim is not None and dim != DIM:
        suffix += f"_d{dim}"
    if not suffix:
        return name
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return f"{stem}{suffix}.{ext}"
    return f"{name}{suffix}"


def main_single(method, packing, enc, skeleton, use_real=False, dim=None):
    """跑单组配置，输出分阶段明细 CSV。dim 缺省用 fe_config.DIM。"""
    dim = dim if dim is not None else DIM
    print("=" * 70)
    tag_data = "真实权重" if use_real else "随机 demo"
    print(f"  单配置：方法={method} 打包={packing} 加密={enc} "
          f"骨架={'开' if skeleton else '关'}  数据={tag_data}  d={dim}")
    print("=" * 70)
    (ctx_o, pub_o, ctx_i, pub_i, B_list, A_list, dW_mean, dim, n_slots
     ) = _prepare(use_real, dim)
    secret_ctx, public_ctx = _ctx_for(method, ctx_o, pub_o, ctx_i, pub_i)

    metrics = MetricsCollector()
    res = run_config(B_list, A_list, dW_mean, dim, n_slots, secret_ctx, public_ctx,
                     method, packing, enc, skeleton, metrics=metrics)
    eps_str = "N/A" if res["error"] is None else f"{res['error']:.3e}"
    feas = "可行" if res["feasible"] else f"不可行（{res['note']}）"
    print(f"\n  客户端加密 {res['t_client_enc']:.3f}s，解密 {res['t_client_dec']:.3f}s，"
          f"服务端聚合 {res['t_server']:.3f}s")
    print(f"  单次上传 {res['up_one']/1024/1024:.2f}MB，"
          f"单次下载 {res['down_one']/1024/1024:.2f}MB，误差 ε={eps_str}，{feas}")
    path = metrics.to_csv(RES_DIR, _prefix_name(CSV_PATH, use_real, dim))
    print(f"\n[完成] 分阶段明细 → {path}")


def _skip_costly(method, packing, enc, skeleton, dim, n_slots, threshold=512):
    """全尺寸跑时跳过物理不现实的格子，如实标注跳过原因。

    dim < 阈值时全跑；否则按估算筛：
      - 外积（任 packing）：按 ct 数量估上传（N 客户端同时驻留 > 8GB 视作 OOM）跳过。
        dim 接近或超过 n_slots 时 packed 退化 g=1、n_groups≈dim，估算跟 unpacked 相当；
        故不是「所有 unpacked 都跳」也不是「所有 packed 都不跳」，具体看内存估算过阈。
      - 内积 packed full：ct×ct 无块内求和，结构性不可行（`_run_inner` 会自动拦下）。
      - 内积 unpacked full：d²·N 次 ct×ct dot 预估数十小时，超预算。
      内积 half 两个 packing 都保留（<2 GB 内存 & 分钟级）。
    :return: 跳过原因字符串，或 None 表示不跳。
    """
    if dim < threshold:
        return None
    if method == "外积":
        g = max(1, n_slots // dim) if packing == "packed" else 1
        n_groups = (dim + g - 1) // g
        # 一个客户端一次上传 ≈ 2 * rank * n_groups * ct_bytes @327KB；N 客户端同时驻留。
        approx_up_gb = 2 * 4 * n_groups * 0.327 / 1024
        total_gb = approx_up_gb * 4
        if total_gb > 8:
            return (f"外积/{packing} dim={dim}：单次上传≈{approx_up_gb:.1f}GB，"
                    f"N 客户端同时驻留内存约 {total_gb:.0f}GB，普通开发机 OOM")
    if method == "内积" and packing == "unpacked" and enc == "full":
        return f"内积 unpacked full dim={dim}：d²·N 次 ct×ct dot 数十小时，超预算"
    return None


def main_sweep(use_real=False, dim=None, skip_costly=False):
    """跑全部 16 组配置，输出对比 CSV + 基线图 + 内外积冗余表/图。

    :param dim: 覆盖 fe_config.DIM 的维度；None 用 fe_config 默认。
    :param skip_costly: True 时按 _skip_costly 规则跳过物理不现实的格子，节省数小时。
    """
    dim = dim if dim is not None else DIM
    print("=" * 70)
    tag_data = "真实权重" if use_real else "随机 demo"
    print(f"  sweep：方法×打包×加密×骨架 = 16 组配置  数据={tag_data}  d={dim}"
          f"  {'（跳过昂贵格）' if skip_costly else ''}")
    print("=" * 70)
    (ctx_o, pub_o, ctx_i, pub_i, B_list, A_list, dW_mean, dim, n_slots
     ) = _prepare(use_real, dim)

    table = SweepTable()
    print(f"\n  {'配置':38s}{'客户端加密':>10s}{'解密':>8s}{'服务端':>9s}"
          f"{'上传MB':>9s}{'下载MB':>9s}{'误差':>12s}{'可行':>6s}")
    print("  " + "-" * 101)
    for method in METHODS:
        secret_ctx, public_ctx = _ctx_for(method, ctx_o, pub_o, ctx_i, pub_i)
        for packing in PACKING_MODES:
            for enc in ENC_LEVELS:
                for skeleton in SKELETON_MODES:
                    skip_reason = (_skip_costly(method, packing, enc, skeleton,
                                                dim, n_slots)
                                   if skip_costly else None)
                    if skip_reason is not None:
                        res = dict(t_client_enc=0.0, t_client_dec=0.0, t_server=0.0,
                                   up_one=0, down_one=0, error=None, feasible=False,
                                   note=skip_reason)
                    else:
                        res = run_config(B_list, A_list, dW_mean, dim, n_slots,
                                         secret_ctx, public_ctx,
                                         method, packing, enc, skeleton)
                    table.add(method, packing, enc, skeleton, n_clients=N_CLIENTS,
                              t_client_enc=res["t_client_enc"],
                              t_client_dec=res["t_client_dec"],
                              t_server=res["t_server"],
                              up_bytes_one=res["up_one"],
                              down_bytes_one=res["down_one"],
                              error=res["error"], feasible=res["feasible"],
                              note=res["note"])
                    tag = f"{method}/{packing}/{enc}/骨架{'开' if skeleton else '关'}"
                    eps = "N/A" if res["error"] is None else f"{res['error']:.3e}"
                    fe = "✓" if res["feasible"] else "✗"
                    print(f"  {tag:38s}{res['t_client_enc']:>9.3f}s"
                          f"{res['t_client_dec']:>7.3f}s{res['t_server']:>8.3f}s"
                          f"{res['up_one']/1024/1024:>8.2f}"
                          f"{res['down_one']/1024/1024:>9.2f}{eps:>12s}{fe:>6s}")

    csv_path = table.to_csv(RES_DIR, _prefix_name(SWEEP_CSV_PATH, use_real, dim))
    print(f"\n[完成] 16 组对比表 → {csv_path}")

    # 内外积冗余对比表。
    red = RedundancyTable().build(table.rows())
    red_csv = red.to_csv(RES_DIR,
                        _prefix_name(REDUNDANCY_CSV_PATH, use_real, dim))
    print(f"[完成] 内外积冗余表 → {red_csv}")

    # 作图：外积法按打包各一张基线图 + 一张内外积冗余图。
    rows = table.rows()
    outer_packed = [r for r in rows if r["方法"] == "外积" and r["打包方式"] == "packed"]
    outer_unpacked = [r for r in rows if r["方法"] == "外积" and r["打包方式"] == "unpacked"]
    for imgs in (plot_baseline(outer_packed, "packed", RES_DIR,
                               _prefix_name(PLOT_PACKED, use_real, dim)),
                 plot_baseline(outer_unpacked, "unpacked", RES_DIR,
                               _prefix_name(PLOT_UNPACKED, use_real, dim)),
                 plot_redundancy(red.rows(), RES_DIR,
                                 _prefix_name(PLOT_REDUNDANCY, use_real, dim))):
        for p in imgs:
            print(f"[完成] 图 → {p}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="联邦 LoRA CKKS 密文聚合实验")
    p.add_argument("--sweep", action="store_true",
                   help="跑全部 16 组配置并出对比 CSV/图 + 冗余表")
    p.add_argument("--method", choices=METHODS, default="外积",
                   help="单跑：计算方法")
    p.add_argument("--packing", choices=PACKING_MODES, default="packed",
                   help="单跑：打包方式")
    p.add_argument("--enc", choices=ENC_LEVELS, default="full",
                   help="单跑：加密程度")
    p.add_argument("--skeleton", action="store_true",
                   help="单跑：启用骨架优化（默认关=完整重建）")
    p.add_argument("--real", action="store_true",
                   help="用 temp_output_dir 真实 LoRA 权重（默认随机 demo）")
    p.add_argument("--dim", type=int, default=None,
                   help="覆盖 fe_config.DIM 的矩阵维度；缺省用 DIM（64）")
    p.add_argument("--skip-costly", action="store_true",
                   help="sweep 时跳过物理不现实的格子（外积按内存估算过阈跳过、"
                        "内积 unpacked full 全部跳过），如实标注跳过原因")
    args = p.parse_args()

    if args.sweep:
        main_sweep(use_real=args.real, dim=args.dim,
                   skip_costly=args.skip_costly)
    else:
        main_single(args.method, args.packing, args.enc, args.skeleton,
                    use_real=args.real, dim=args.dim)
