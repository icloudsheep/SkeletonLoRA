"""单个矩形 AB 对的明文、外积 CKKS 与 skeleton 运行器。"""

import time

import numpy as np

from fe_modes import (
    build_partition,
    encrypted_output_mask,
    error_summary,
    mean_reference,
)
from fe_inner_hybrid import (
    aggregate as aggregate_inner,
    decrypt_result as decrypt_inner_result,
    download_bytes as download_inner_bytes,
    encrypt_upload as encrypt_inner_upload,
    upload_bytes as upload_inner_bytes,
)
from fe_outer_hybrid import (
    aggregate,
    build_blocks,
    decrypt_result,
    download_bytes,
    encrypt_upload,
    upload_bytes,
)
from fe_skeleton import cur_reconstruct_with_stats, select_uniform_rect_indices


def _sum_counts(counts):
    keys = {key for item in counts for key in item}
    return {key: sum(item.get(key, 0) for item in counts) for key in keys}


def _region_errors(output, reference, partition):
    encrypted_mask = encrypted_output_mask(partition, reference.shape)
    return {
        "encrypted": error_summary(output, reference, encrypted_mask)
        if encrypted_mask.any() else None,
        "plaintext": error_summary(output, reference, ~encrypted_mask)
        if (~encrypted_mask).any() else None,
    }


def _skeleton_mask(shape, row_idx, col_idx):
    mask = np.zeros(shape, dtype=bool)
    mask[row_idx, :] = True
    mask[:, col_idx] = True
    return mask


def run_plain_pair(B_list, A_list, n_clients, scaling, skeleton, skeleton_r):
    """运行一个 AB 对的纯明文 baseline。"""
    started = time.perf_counter()
    reference = mean_reference(
        [type("Pair", (), {"a": A, "b": B}) for A, B in zip(A_list, B_list)],
        n_clients,
        scaling,
    )
    reference_seconds = time.perf_counter() - started
    if not skeleton:
        return {
            "matrix": reference,
            "reference": reference,
            "error": error_summary(reference, reference),
            "feasible": True,
            "note": "纯明文参考",
            "timing": {"reference_seconds": reference_seconds, "cur_seconds": 0.0},
            "skeleton": {"ciphertext_elements": 0, "plaintext_elements": reference.size},
            "error_regions": {"encrypted": None, "plaintext": error_summary(reference, reference)},
        }
    row_idx, col_idx = select_uniform_rect_indices(*reference.shape, skeleton_r)
    cur_started = time.perf_counter()
    reconstructed, ok, cur_stats = cur_reconstruct_with_stats(
        reference[:, col_idx], reference[row_idx, :], row_idx, col_idx
    )
    cur_seconds = time.perf_counter() - cur_started
    return {
        "matrix": reconstructed,
        "reference": reference,
        "error": error_summary(reconstructed, reference) if ok else None,
        "feasible": ok,
        "note": "明文 skeleton CUR" if ok else cur_stats["failure_reason"],
        "timing": {"reference_seconds": reference_seconds, "cur_seconds": cur_seconds},
        "skeleton": {
            "ciphertext_elements": 0,
            "plaintext_elements": row_idx.size * reference.shape[1]
            + col_idx.size * (reference.shape[0] - row_idx.size),
        },
        "cur": cur_stats,
        "skeleton_error": error_summary(
            reconstructed,
            reference,
            _skeleton_mask(reference.shape, row_idx, col_idx),
        ) if ok else None,
        "error_regions": {
            "encrypted": None,
            "plaintext": error_summary(reconstructed, reference) if ok else None,
        },
    }


def run_outer_pair(
    B_list,
    A_list,
    public_ctx,
    secret_ctx,
    n_clients,
    rank,
    mode,
    ratio,
    scaling,
    skeleton,
    skeleton_r,
    n_slots=None,
):
    """运行一个 AB 对的外积 CKKS 配置。"""
    B_list = [np.asarray(B, dtype=np.float64) for B in B_list]
    A_list = [np.asarray(A, dtype=np.float64) * scaling for A in A_list]
    out_features, in_features = B_list[0].shape[0], A_list[0].shape[1]
    partition = build_partition(out_features, in_features, mode, ratio)
    if skeleton:
        row_idx, col_idx = select_uniform_rect_indices(
            out_features, in_features, skeleton_r
        )
    else:
        row_idx = col_idx = None
    blocks = build_blocks(
        out_features,
        in_features,
        partition,
        skeleton,
        skeleton_rows=row_idx,
        skeleton_cols=col_idx,
        max_slots=n_slots,
    )

    started = time.perf_counter()
    uploads = [
        encrypt_upload(B, A, public_ctx, rank, partition, blocks)
        for B, A in zip(B_list, A_list)
    ]
    client_encrypt_seconds = (time.perf_counter() - started) / n_clients
    upload_sizes = [upload_bytes(upload) for upload in uploads]

    started = time.perf_counter()
    aggregate_result = aggregate(uploads, public_ctx, n_clients)
    server_seconds = time.perf_counter() - started
    download_size = download_bytes(aggregate_result)

    started = time.perf_counter()
    matrix, decrypt_stats = decrypt_result(aggregate_result, secret_ctx)
    client_decrypt_seconds = time.perf_counter() - started
    reference = mean_reference(
        [type("Pair", (), {"a": A, "b": B}) for A, B in zip(A_list, B_list)],
        n_clients,
        scaling=1.0,
    )
    cur_stats = None
    cur_seconds = 0.0
    if skeleton:
        skeleton_error = error_summary(
            matrix,
            reference,
            _skeleton_mask(reference.shape, row_idx, col_idx),
        )
        cur_started = time.perf_counter()
        reconstructed, feasible, cur_stats = cur_reconstruct_with_stats(
            matrix[:, col_idx], matrix[row_idx, :], row_idx, col_idx
        )
        cur_seconds = time.perf_counter() - cur_started
        output = reconstructed
    else:
        feasible = True
        output = matrix
        skeleton_error = None
    return {
        "matrix": output,
        "pre_cur_matrix": matrix,
        "reference": reference,
        "error": error_summary(output, reference) if feasible else None,
        "feasible": feasible,
        "note": "外积 CKKS skeleton" if skeleton else "外积 CKKS 完整矩阵",
        "timing": {
            "client_encrypt_seconds": client_encrypt_seconds,
            "server_seconds": server_seconds,
            "client_decrypt_seconds": client_decrypt_seconds,
            "cur_seconds": cur_seconds,
        },
        "upload": _sum_counts(upload_sizes),
        "download": download_size,
        "skeleton": decrypt_stats,
        "cur": cur_stats,
        "direct_error": error_summary(matrix, reference) if not skeleton else None,
        "skeleton_error": skeleton_error,
        "error_regions": _region_errors(output, reference, partition) if feasible else None,
    }


def run_inner_pair(
    B_list,
    A_list,
    public_ctx,
    secret_ctx,
    n_clients,
    mode,
    ratio,
    scaling,
    skeleton,
    skeleton_r,
    n_slots,
    time_budget,
):
    """运行一个 AB 对的内积 CKKS 配置。"""
    B_list = [np.asarray(B, dtype=np.float64) for B in B_list]
    A_list = [np.asarray(A, dtype=np.float64) * scaling for A in A_list]
    out_features, in_features = B_list[0].shape[0], A_list[0].shape[1]
    partition = build_partition(out_features, in_features, mode, ratio)
    if skeleton:
        row_idx, col_idx = select_uniform_rect_indices(
            out_features, in_features, skeleton_r
        )
    else:
        row_idx = col_idx = None

    started = time.perf_counter()
    uploads = [
        encrypt_inner_upload(B, A, public_ctx, partition)
        for B, A in zip(B_list, A_list)
    ]
    client_encrypt_seconds = (time.perf_counter() - started) / n_clients
    upload_sizes = [upload_inner_bytes(upload) for upload in uploads]

    started = time.perf_counter()
    aggregate_result = aggregate_inner(
        uploads,
        public_ctx,
        partition,
        skeleton,
        row_idx,
        col_idx,
        n_clients,
        n_slots,
        time_budget,
    )
    server_seconds = time.perf_counter() - started
    if not aggregate_result["feasible"]:
        return {
            "feasible": False,
            "note": aggregate_result["note"],
            "timing": {
                "client_encrypt_seconds": client_encrypt_seconds,
                "server_seconds": server_seconds,
            },
            "upload": _sum_counts(upload_sizes),
        }
    download_size = download_inner_bytes(aggregate_result)

    started = time.perf_counter()
    matrix, decrypt_stats = decrypt_inner_result(aggregate_result, secret_ctx)
    client_decrypt_seconds = time.perf_counter() - started
    reference = mean_reference(
        [type("Pair", (), {"a": A, "b": B}) for A, B in zip(A_list, B_list)],
        n_clients,
        scaling=1.0,
    )
    cur_stats = None
    cur_seconds = 0.0
    if skeleton:
        skeleton_error = error_summary(
            matrix,
            reference,
            _skeleton_mask(reference.shape, row_idx, col_idx),
        )
        cur_started = time.perf_counter()
        reconstructed, feasible, cur_stats = cur_reconstruct_with_stats(
            matrix[:, col_idx], matrix[row_idx, :], row_idx, col_idx
        )
        cur_seconds = time.perf_counter() - cur_started
        output = reconstructed
    else:
        feasible = True
        output = matrix
        skeleton_error = None
    return {
        "matrix": output,
        "pre_cur_matrix": matrix,
        "reference": reference,
        "error": error_summary(output, reference) if feasible else None,
        "feasible": feasible,
        "note": "内积 CKKS skeleton" if skeleton else "内积 CKKS 完整矩阵",
        "timing": {
            "client_encrypt_seconds": client_encrypt_seconds,
            "server_seconds": server_seconds,
            "client_decrypt_seconds": client_decrypt_seconds,
            "cur_seconds": cur_seconds,
        },
        "upload": _sum_counts(upload_sizes),
        "download": download_size,
        "skeleton": decrypt_stats,
        "cur": cur_stats,
        "direct_error": error_summary(matrix, reference) if not skeleton else None,
        "skeleton_error": skeleton_error,
        "error_regions": _region_errors(output, reference, partition) if feasible else None,
    }
