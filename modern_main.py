#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SkeletonLoRA 新实验协议入口。"""

import argparse
import csv
import json
from pathlib import Path
import re
import time

import numpy as np

from fe_config import (
    COEFF_MOD_BIT_SIZES,
    DIM,
    GLOBAL_SCALE,
    INNER_FULL_TIME_BUDGET,
    LORA_ALPHA,
    METHODS,
    N_CLIENTS,
    PARTIAL_RATIOS,
    POLY_MODULUS_DEGREE,
    RANK,
    RELATIVE_ERROR_FAILURE,
    RELATIVE_ERROR_WARNING,
    RUNS_DIR,
    SCALING,
    SKELETON_R,
    TEMP_OUTPUT_DIR,
)
from fe_context import create_secret_context, derive_public_context
from fe_data import (
    ABIdentifier,
    ABPair,
    ClientABCollection,
    client_paths,
    discover_all_clients,
    infer_shapes,
    materialize_pair,
)
from fe_logging import RunLogger, create_run_paths, environment_snapshot, write_json
from fe_runner import run_inner_pair, run_outer_pair, run_plain_pair


def _progress(message):
    print(message, flush=True)


def gen_demo_collections(n_clients, dim, rank):
    """生成一个固定随机种子的矩形 demo AB 集合。"""
    rng = np.random.RandomState(42)
    identifier = ABIdentifier("demo.layer.q_proj", "default", "lora")
    pairs = []
    for _ in range(n_clients):
        pairs.append(
            ABPair(
                identifier=identifier,
                a=rng.randn(rank, dim) * 0.05,
                b=rng.randn(dim, rank) * 0.05,
                a_key=None,
                b_key=None,
            )
        )
    return identifier, pairs, (dim, dim)


def _task_id(identifier, method, mode, ratio, skeleton):
    ratio_label = "none" if ratio is None else str(ratio)
    return "/".join((identifier.text, method, mode, ratio_label, str(int(skeleton))))


def _tensorboard_prefix(method, mode, ratio, skeleton):
    ratio_label = "none" if ratio is None else str(ratio)
    matrix_label = "skeleton" if skeleton else "full_matrix"
    return "/".join((method, mode, ratio_label, matrix_label))


def _row_base(run_id, identifier, ab_index, method, mode, ratio, skeleton, shape):
    return {
        "运行 ID": run_id,
        "AB index": ab_index,
        "层/投影/AB 标识": identifier.text,
        "方法": method,
        "实验模式": mode,
        "部分加密比例": "N/A" if ratio is None else ratio,
        "骨架优化": "开" if skeleton else "关",
        "输出维度": shape[0],
        "输入维度": shape[1],
        "客户端数量": N_CLIENTS,
        "LoRA 秩": RANK,
        "LoRA alpha": LORA_ALPHA,
        "LoRA scaling": SCALING,
        "skeleton rank": SKELETON_R,
        "实际 skeleton rank": min(SKELETON_R, N_CLIENTS * RANK),
        "初始化 context 引用": "N/A" if method == "明文参考" else f"context-{method}",
    }


def _result_row(base, result):
    row = dict(base)
    row.update(
        {
            "可行性": "可行" if result.get("feasible") else "不可行",
            "备注": result.get("note", ""),
            "相对 Frobenius 误差": "N/A",
            "最大绝对误差": "N/A",
            "平均绝对误差": "N/A",
            "客户端加密秒": "N/A",
            "服务端聚合秒": "N/A",
            "客户端解密秒": "N/A",
            "CUR 秒": "N/A",
            "上传密文字节": "N/A",
            "上传明文字节": "N/A",
            "上传元数据字节": "N/A",
            "下发密文字节": "N/A",
            "下发明文字节": "N/A",
            "下发元数据字节": "N/A",
            "上传合计字节": "N/A",
            "下发合计字节": "N/A",
            "聚合阶段总网络字节": "N/A",
            "密文元素数": "N/A",
            "明文元素数": "N/A",
            "交叉块数值秩": "N/A",
            "交叉块条件数": "N/A",
            "逆方法": "N/A",
            "精度状态": "N/A",
            "CKKS 直接结果误差": "N/A",
            "骨架行列误差": "N/A",
            "CUR 重建误差": "N/A",
            "密文区域误差": "N/A",
            "明文区域误差": "N/A",
        }
    )
    error = result.get("error")
    if error:
        row["相对 Frobenius 误差"] = error["relative_frobenius_error"]
        row["最大绝对误差"] = error["max_absolute_error"]
        row["平均绝对误差"] = error["mean_absolute_error"]
        relative_error = error["relative_frobenius_error"]
        if relative_error >= RELATIVE_ERROR_FAILURE:
            row["精度状态"] = "failure"
        elif relative_error >= RELATIVE_ERROR_WARNING:
            row["精度状态"] = "warning"
        else:
            row["精度状态"] = "正常"
    timing = result.get("timing", {})
    for key, output_key in (
        ("client_encrypt_seconds", "客户端加密秒"),
        ("server_seconds", "服务端聚合秒"),
        ("client_decrypt_seconds", "客户端解密秒"),
        ("cur_seconds", "CUR 秒"),
    ):
        if key in timing:
            row[output_key] = timing[key]
    for source, prefix in ((result.get("upload", {}), "上传"), (result.get("download", {}), "下发")):
        for key, suffix in (
            ("ciphertext_bytes", "密文字节"),
            ("plaintext_bytes", "明文字节"),
            ("metadata_bytes", "元数据字节"),
        ):
            if key in source:
                row[f"{prefix}{suffix}"] = source[key]
        if "total_bytes" in source:
            row[f"{prefix}合计字节"] = source["total_bytes"]
    upload_total = _as_number(row["上传合计字节"])
    download_total = _as_number(row["下发合计字节"])
    if upload_total is not None or download_total is not None:
        row["聚合阶段总网络字节"] = (upload_total or 0) + (download_total or 0)
    skeleton = result.get("skeleton", {})
    if skeleton:
        row["密文元素数"] = skeleton.get("ciphertext_elements", "N/A")
        row["明文元素数"] = skeleton.get("plaintext_elements", "N/A")
    cur = result.get("cur") or {}
    if cur:
        row["交叉块数值秩"] = cur.get("numerical_rank", "N/A")
        row["交叉块条件数"] = cur.get("condition_number", "N/A")
        row["逆方法"] = cur.get("inverse_method", "N/A")
    if result.get("direct_error"):
        row["CKKS 直接结果误差"] = result["direct_error"]["relative_frobenius_error"]
    if result.get("skeleton_error"):
        row["骨架行列误差"] = result["skeleton_error"]["relative_frobenius_error"]
    if result.get("cur") and result.get("error"):
        row["CUR 重建误差"] = result["error"]["relative_frobenius_error"]
    regions = result.get("error_regions") or {}
    if regions.get("encrypted"):
        row["密文区域误差"] = regions["encrypted"]["relative_frobenius_error"]
    if regions.get("plaintext"):
        row["明文区域误差"] = regions["plaintext"]["relative_frobenius_error"]
    return row


def _as_number(value):
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


IDENTITY_COLUMNS = [
    "运行 ID",
    "AB index",
    "层/投影/AB 标识",
    "方法",
    "实验模式",
    "部分加密比例",
    "骨架优化",
    "输出维度",
    "输入维度",
]

CONFIG_COLUMNS = [
    "客户端数量",
    "LoRA 秩",
    "LoRA alpha",
    "LoRA scaling",
    "skeleton rank",
    "实际 skeleton rank",
    "初始化 context 引用",
]

STATUS_COLUMNS = [
    "可行性",
    "精度状态",
    "备注",
]

SUMMARY_COLUMNS = [
    *IDENTITY_COLUMNS,
    *STATUS_COLUMNS,
    "相对 Frobenius 误差",
    "客户端加密秒",
    "服务端聚合秒",
    "客户端解密秒",
    "CUR 秒",
    "上传合计字节",
    "下发合计字节",
    "聚合阶段总网络字节",
]

PRECISION_COLUMNS = [
    *IDENTITY_COLUMNS,
    *STATUS_COLUMNS,
    "相对 Frobenius 误差",
    "最大绝对误差",
    "平均绝对误差",
    "CKKS 直接结果误差",
    "骨架行列误差",
    "CUR 重建误差",
    "密文区域误差",
    "明文区域误差",
]

TIMING_COLUMNS = [
    *IDENTITY_COLUMNS,
    *STATUS_COLUMNS,
    "客户端加密秒",
    "服务端聚合秒",
    "客户端解密秒",
    "CUR 秒",
]

COMMUNICATION_COLUMNS = [
    *IDENTITY_COLUMNS,
    *STATUS_COLUMNS,
    "上传密文字节",
    "上传明文字节",
    "上传元数据字节",
    "上传合计字节",
    "下发密文字节",
    "下发明文字节",
    "下发元数据字节",
    "下发合计字节",
    "聚合阶段总网络字节",
]

SKELETON_COLUMNS = [
    *IDENTITY_COLUMNS,
    *STATUS_COLUMNS,
    "密文元素数",
    "明文元素数",
    "交叉块数值秩",
    "交叉块条件数",
    "逆方法",
]

CONFIG_DETAIL_COLUMNS = [
    *IDENTITY_COLUMNS,
    *CONFIG_COLUMNS,
    *STATUS_COLUMNS,
]


def _select_columns(rows, columns):
    derived_rows = [_with_derived_metrics(row) for row in rows]
    return [
        {column: row.get(column, "N/A") for column in columns}
        for row in derived_rows
    ]


def _sum_metric_fields(row, columns):
    values = [_as_number(row.get(column)) for column in columns]
    if all(value is None for value in values):
        return None
    total = sum(value or 0 for value in values)
    return int(total) if total.is_integer() else total


def _with_derived_metrics(row):
    result = dict(row)
    if _as_number(result.get("上传合计字节")) is None:
        upload_total = _sum_metric_fields(
            result, ("上传密文字节", "上传明文字节", "上传元数据字节")
        )
        if upload_total is not None:
            result["上传合计字节"] = upload_total
    if _as_number(result.get("下发合计字节")) is None:
        download_total = _sum_metric_fields(
            result, ("下发密文字节", "下发明文字节", "下发元数据字节")
        )
        if download_total is not None:
            result["下发合计字节"] = download_total
    if _as_number(result.get("聚合阶段总网络字节")) is None:
        network_total = _sum_metric_fields(result, ("上传合计字节", "下发合计字节"))
        if network_total is not None:
            result["聚合阶段总网络字节"] = network_total
    return result


def _sort_by_number(rows, column):
    derived_rows = [_with_derived_metrics(row) for row in rows]
    return sorted(
        (row for row in derived_rows if _as_number(row.get(column)) is not None),
        key=lambda row: _as_number(row.get(column)),
        reverse=True,
    )


def _write_metric_tables(root, rows):
    rows = [_with_derived_metrics(row) for row in rows]
    _write_csv(root / "ab_metrics.csv", _select_columns(rows, SUMMARY_COLUMNS), SUMMARY_COLUMNS)
    _write_csv(
        root / "ab_config_metrics.csv",
        _select_columns(rows, CONFIG_DETAIL_COLUMNS),
        CONFIG_DETAIL_COLUMNS,
    )
    _write_csv(
        root / "precision_metrics.csv",
        _select_columns(rows, PRECISION_COLUMNS),
        PRECISION_COLUMNS,
    )
    _write_csv(
        root / "timing_metrics.csv",
        _select_columns(rows, TIMING_COLUMNS),
        TIMING_COLUMNS,
    )
    _write_csv(
        root / "communication_metrics.csv",
        _select_columns(rows, COMMUNICATION_COLUMNS),
        COMMUNICATION_COLUMNS,
    )
    _write_csv(
        root / "skeleton_cur_metrics.csv",
        _select_columns(rows, SKELETON_COLUMNS),
        SKELETON_COLUMNS,
    )


def _status_counts(rows, column):
    counts = {}
    for row in rows:
        value = row.get(column, "N/A")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _max_metric(rows, column):
    values = [_as_number(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _write_artifacts(paths, rows, context_rows):
    rows = [_with_derived_metrics(row) for row in rows]
    summary = {
        "run_id": paths.root.name,
        "total_tasks": len(rows),
        "feasibility": _status_counts(rows, "可行性"),
        "precision_status": _status_counts(rows, "精度状态"),
        "max_relative_frobenius_error": _max_metric(rows, "相对 Frobenius 误差"),
        "max_upload_total_bytes": _max_metric(rows, "上传合计字节"),
        "max_download_total_bytes": _max_metric(rows, "下发合计字节"),
        "max_network_total_bytes": _max_metric(rows, "聚合阶段总网络字节"),
        "context_count": len(context_rows),
    }
    write_json(paths.artifacts / "run_summary.json", summary)
    (paths.artifacts / "run_summary.md").write_text(
        "\n".join(
            [
                f"# {paths.root.name}",
                "",
                f"- 任务数：{summary['total_tasks']}",
                f"- 可行性：{summary['feasibility']}",
                f"- 精度状态：{summary['precision_status']}",
                f"- 最大相对 Frobenius 误差：{summary['max_relative_frobenius_error']}",
                f"- 最大聚合阶段总网络字节：{summary['max_network_total_bytes']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_csv(
        paths.artifacts / "top_errors.csv",
        _select_columns(_sort_by_number(rows, "相对 Frobenius 误差")[:20], PRECISION_COLUMNS),
        PRECISION_COLUMNS,
    )
    _write_csv(
        paths.artifacts / "top_communication.csv",
        _select_columns(_sort_by_number(rows, "聚合阶段总网络字节")[:20], COMMUNICATION_COLUMNS),
        COMMUNICATION_COLUMNS,
    )
    timing_rows = []
    for row in rows:
        timing_total = sum(
            _as_number(row.get(column)) or 0
            for column in ("客户端加密秒", "服务端聚合秒", "客户端解密秒", "CUR 秒")
        )
        copied = dict(row)
        copied["任务总耗时秒"] = timing_total
        timing_rows.append(copied)
    timing_columns = [*TIMING_COLUMNS, "任务总耗时秒"]
    _write_csv(
        paths.artifacts / "top_timing.csv",
        _select_columns(_sort_by_number(timing_rows, "任务总耗时秒")[:20], timing_columns),
        timing_columns,
    )


def _ab_profile_rows(identifiers, collections, shapes):
    rows = []
    for ab_index, identifier in enumerate(identifiers):
        shape = shapes[identifier]
        pairs = [
            materialize_pair(collection, identifier, shape, RANK)
            for collection in collections
        ]
        a_norms = [float(np.linalg.norm(pair.a)) for pair in pairs]
        b_norms = [float(np.linalg.norm(pair.b)) for pair in pairs]
        a_max_abs = [float(np.max(np.abs(pair.a))) if pair.a.size else 0.0 for pair in pairs]
        b_max_abs = [float(np.max(np.abs(pair.b))) if pair.b.size else 0.0 for pair in pairs]
        rows.append(
            {
                "AB index": ab_index,
                "层/投影/AB 标识": identifier.text,
                "输出维度": shape[0],
                "输入维度": shape[1],
                "A 平均 Frobenius 范数": float(np.mean(a_norms)),
                "B 平均 Frobenius 范数": float(np.mean(b_norms)),
                "A 平均最大绝对值": float(np.mean(a_max_abs)),
                "B 平均最大绝对值": float(np.mean(b_max_abs)),
            }
        )
    return rows


def _write_ab_profile(paths, logger, profile_rows):
    if not profile_rows:
        return
    logger.text(
        "ab_profile/index_map",
        json.dumps(
            [
                {"ab_index": row["AB index"], "identifier": row["层/投影/AB 标识"]}
                for row in profile_rows
            ],
            ensure_ascii=False,
            indent=2,
        ),
    )
    for row in profile_rows:
        ab_index = row["AB index"]
        metric_fields = {"identifier": row["层/投影/AB 标识"]}
        for tag, column in (
            ("ab_profile/A/frobenius_norm_mean", "A 平均 Frobenius 范数"),
            ("ab_profile/B/frobenius_norm_mean", "B 平均 Frobenius 范数"),
            ("ab_profile/A/max_abs_mean", "A 平均最大绝对值"),
            ("ab_profile/B/max_abs_mean", "B 平均最大绝对值"),
            ("ab_profile/shape/output_dim", "输出维度"),
            ("ab_profile/shape/input_dim", "输入维度"),
        ):
            logger.scalar(tag, row[column], ab_index, **metric_fields)
    _write_csv(paths.artifacts / "ab_profile.csv", profile_rows)


def _load_inputs(use_real, dim):
    if not use_real:
        _progress(f"[输入] 生成 demo 数据：客户端={N_CLIENTS}，维度={dim}，rank={RANK}")
        identifier, pairs, shape = gen_demo_collections(N_CLIENTS, dim, RANK)
        collections = [
            ClientABCollection(
                client_id=client_id,
                path="demo",
                pairs={identifier: pair},
                warnings=[],
            )
            for client_id, pair in enumerate(pairs)
        ]
        return [identifier], collections, {identifier: shape}, []
    _progress(f"[输入] 扫描真实 LoRA 权重：配置客户端数={N_CLIENTS}，目录={TEMP_OUTPUT_DIR}")
    available_client_ids = []
    for path in Path(TEMP_OUTPUT_DIR).glob("client_*_output/final_lora/adapter_model.safetensors"):
        match = re.fullmatch(r"client_(\d+)_output", path.parent.parent.name)
        if match:
            available_client_ids.append(int(match.group(1)))
    available_client_ids.sort()
    _progress(f"[输入] 发现权重文件={len(available_client_ids)}，客户端编号={available_client_ids[:8]}"
              + ("..." if len(available_client_ids) > 8 else ""))
    if len(available_client_ids) != N_CLIENTS:
        _progress(
            f"[警告] 配置只读取 client_0 到 client_{N_CLIENTS - 1}；"
            f"实际发现 {len(available_client_ids)} 个客户端文件"
        )
    paths = client_paths(TEMP_OUTPUT_DIR, N_CLIENTS)
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("缺少客户端权重文件：" + ", ".join(missing))
    _progress(f"[输入] 正在读取 {len(paths)} 个客户端 safetensors 文件，请等待...")
    collections, identifiers = discover_all_clients(paths, RANK)
    _progress(f"[输入] AB 标识发现完成：{len(identifiers)} 个，开始检查形状")
    shapes, shape_warnings = infer_shapes(collections, identifiers, RANK)
    if not shapes:
        raise ValueError("没有发现合法的 LoRA AB 对")
    return list(shapes), collections, shapes, shape_warnings


def run_experiment(use_real=False, dim=None, selected_method=None, selected_mode=None,
                   selected_ratio=None, selected_skeleton=None):
    """执行新实验协议；高维真实实验前由调用方先确认资源。"""
    dim = dim if dim is not None else DIM
    _progress(
        f"[启动] mode={'real' if use_real else 'demo'}，方法={selected_method or '全部'}，"
        f"实验模式={selected_mode or '全部'}"
    )
    identifiers, collections, shapes, warnings = _load_inputs(use_real, dim)
    max_dim = max(max(shape) for shape in shapes.values())
    slots = POLY_MODULUS_DEGREE // 2
    _progress(f"[检查] 合法 AB={len(identifiers)}，最大维度={max_dim}，CKKS 槽位={slots}")
    if max_dim > slots:
        raise ValueError(f"最大矩阵维度 {max_dim} 超过 CKKS 槽位数 {slots}")
    if any(min(shape) < SKELETON_R for shape in shapes.values()):
        raise ValueError(f"存在维度小于 SKELETON_R={SKELETON_R} 的 AB 对")

    label = ("real" if use_real else "demo") + f"-d{max_dim}"
    paths = create_run_paths(RUNS_DIR, label)
    run_id = paths.root.name
    config = {
        "use_real": use_real,
        "dim": dim,
        "max_dim": max_dim,
        "n_clients": N_CLIENTS,
        "rank": RANK,
        "lora_alpha": LORA_ALPHA,
        "scaling": SCALING,
        "skeleton_r": SKELETON_R,
        "poly_modulus_degree": POLY_MODULUS_DEGREE,
        "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
        "global_scale": GLOBAL_SCALE,
        "methods": METHODS,
        "modes": ["plain_baseline", "partial_A", "partial_AB", "full"],
        "partial_ratios": PARTIAL_RATIOS,
        "warnings": warnings,
    }
    write_json(paths.config, config)
    write_json(paths.environment, environment_snapshot())
    logger = RunLogger(paths)
    logger.text("run/config", json.dumps(config, ensure_ascii=False, indent=2))
    ab_profile_rows = _ab_profile_rows(identifiers, collections, shapes)
    _write_ab_profile(paths, logger, ab_profile_rows)
    ab_indices = {identifier: index for index, identifier in enumerate(identifiers)}
    _progress(f"[运行] 输出目录：{paths.root}")
    rows = []
    step = 0
    completed_tasks = 0
    failed_tasks = 0
    contexts = {}
    context_metrics = {}
    try:
        for method in ["明文参考", *METHODS]:
            if selected_method and method != selected_method:
                continue
            if method in METHODS:
                _progress(f"[上下文] 创建 {method} CKKS context...")
                started = time.perf_counter()
                secret_ctx = create_secret_context(
                    POLY_MODULUS_DEGREE,
                    COEFF_MOD_BIT_SIZES,
                    GLOBAL_SCALE,
                    galois=method == "内积",
                )
                context_create = time.perf_counter() - started
                started = time.perf_counter()
                public_ctx = derive_public_context(secret_ctx)
                context_derive = time.perf_counter() - started
                contexts[method] = (secret_ctx, public_ctx)
                context_metrics[method] = {
                    "create_seconds": context_create,
                    "derive_seconds": context_derive,
                    "public_context_bytes": len(public_ctx.serialize()),
                }
                _progress(
                    f"[上下文] {method} 完成：创建={context_create:.2f}s，"
                    f"派生={context_derive:.2f}s，公钥上下文={len(public_ctx.serialize())} bytes"
                )
            for identifier in identifiers:
                shape = shapes[identifier]
                pairs = [
                    materialize_pair(collection, identifier, shape, RANK)
                    for collection in collections
                ]
                B_list = [pair.b for pair in pairs]
                A_list = [pair.a for pair in pairs]
                task_modes = [("plain_baseline", None)] if method == "明文参考" else []
                if method in METHODS:
                    task_modes += [("full", None)]
                    task_modes += [(mode, ratio) for mode in ("partial_A", "partial_AB") for ratio in PARTIAL_RATIOS]
                for mode, ratio in task_modes:
                    if selected_mode and mode != selected_mode:
                        continue
                    if selected_ratio is not None and ratio != selected_ratio:
                        continue
                    skeleton_values = [False, True]
                    if selected_skeleton is not None:
                        skeleton_values = [selected_skeleton]
                    for skeleton in skeleton_values:
                        task_id = _task_id(identifier, method, mode, ratio, skeleton)
                        _progress(f"[任务 {step + 1}] {task_id}")
                        logger.task(task_id, "started")
                        ab_index = ab_indices[identifier]
                        base = _row_base(run_id, identifier, ab_index, method, mode, ratio, skeleton, shape)
                        try:
                            if mode == "plain_baseline":
                                result = run_plain_pair(B_list, A_list, N_CLIENTS, SCALING, skeleton, SKELETON_R)
                            elif method == "外积":
                                secret_ctx, public_ctx = contexts[method]
                                result = run_outer_pair(
                                    B_list, A_list, public_ctx, secret_ctx, N_CLIENTS, RANK,
                                    mode, ratio, SCALING, skeleton, min(SKELETON_R, N_CLIENTS * RANK), slots,
                                )
                            else:
                                secret_ctx, public_ctx = contexts[method]
                                result = run_inner_pair(
                                    B_list,
                                    A_list,
                                    public_ctx,
                                    secret_ctx,
                                    N_CLIENTS,
                                    mode,
                                    ratio,
                                    SCALING,
                                    skeleton,
                                    min(SKELETON_R, N_CLIENTS * RANK),
                                    slots,
                                    INNER_FULL_TIME_BUDGET,
                                )
                            rows.append(_result_row(base, result))
                            tag_prefix = _tensorboard_prefix(method, mode, ratio, skeleton)
                            metric_fields = {
                                "identifier": identifier.text,
                                "method": method,
                                "mode": mode,
                                "ratio": "none" if ratio is None else ratio,
                                "skeleton": "开" if skeleton else "关",
                            }
                            if result.get("error"):
                                logger.scalar(
                                    f"error/{tag_prefix}/relative_frobenius",
                                    result["error"]["relative_frobenius_error"],
                                    ab_index,
                                    **metric_fields,
                                )
                            for timing_name, value in result.get("timing", {}).items():
                                logger.scalar(
                                    f"timing/{tag_prefix}/{timing_name}",
                                    value,
                                    ab_index,
                                    **metric_fields,
                                )
                            for direction in ("upload", "download"):
                                for metric_name, value in result.get(direction, {}).items():
                                    if metric_name == "total_bytes":
                                        continue
                                    logger.metric(
                                        f"communication/{tag_prefix}/{direction}_{metric_name}",
                                        value,
                                        ab_index,
                                        **metric_fields,
                                    )
                            upload_total = result.get("upload", {}).get("total_bytes")
                            download_total = result.get("download", {}).get("total_bytes")
                            if upload_total is not None:
                                logger.scalar(
                                    f"communication/{tag_prefix}/upload_total_bytes",
                                    upload_total,
                                    ab_index,
                                    **metric_fields,
                                )
                            if download_total is not None:
                                logger.scalar(
                                    f"communication/{tag_prefix}/download_total_bytes",
                                    download_total,
                                    ab_index,
                                    **metric_fields,
                                )
                            if upload_total is not None or download_total is not None:
                                logger.scalar(
                                    f"communication/{tag_prefix}/network_total_bytes",
                                    (upload_total or 0) + (download_total or 0),
                                    ab_index,
                                    **metric_fields,
                                )
                            completed_tasks += 1
                            logger.task(task_id, "completed", feasible=result.get("feasible"), note=result.get("note"))
                        except Exception as exc:
                            result = {"feasible": False, "note": repr(exc)}
                            rows.append(_result_row(base, result))
                            failed_tasks += 1
                            logger.task(task_id, "failed", error=repr(exc))
                        logger.scalar("progress/completed_tasks", completed_tasks, step)
                        logger.scalar("progress/failed_tasks", failed_tasks, step)
                        logger.scalar("progress/processed_tasks", completed_tasks + failed_tasks, step)
                        step += 1
                        logger.flush()
    finally:
        logger.close()
    _write_metric_tables(paths.root, rows)
    context_rows = [
        {"方法": method, **values} for method, values in context_metrics.items()
    ]
    _write_csv(paths.root / "context_metrics.csv", context_rows)
    _write_artifacts(paths, rows, context_rows)
    return paths.root


def main():
    parser = argparse.ArgumentParser(description="SkeletonLoRA 新实验协议")
    parser.add_argument("--real", action="store_true", help="加载真实 LoRA 权重")
    parser.add_argument("--dim", type=int, default=None, help="demo 矩阵维度")
    parser.add_argument("--method", choices=["明文参考", "外积", "内积"], default=None)
    parser.add_argument("--mode", choices=["plain_baseline", "partial_A", "partial_AB", "full"], default=None)
    parser.add_argument("--ratio", type=int, choices=PARTIAL_RATIOS, default=None)
    parser.add_argument("--skeleton", action="store_true", default=None)
    args = parser.parse_args()
    path = run_experiment(
        use_real=args.real,
        dim=args.dim,
        selected_method=args.method,
        selected_mode=args.mode,
        selected_ratio=args.ratio,
        selected_skeleton=args.skeleton,
    )
    print(f"[完成] run → {path}")


if __name__ == "__main__":
    main()
