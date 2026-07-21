"""矩形内积 CKKS 的 full/partial 混合聚合协议。"""

import time

import numpy as np
import tenseal as ts

from fe_outer_hybrid import build_blocks


def _operand(values, encrypted, public_ctx):
    values = np.asarray(values, dtype=np.float64)
    if encrypted:
        return ("ct", ts.ckks_vector(public_ctx, values.tolist()).serialize())
    return ("plain", values)


def encrypt_upload(B, A, public_ctx, partition):
    """按行/列选择生成紧凑的内积上传包。"""
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    encrypted_rows = set(int(item) for item in partition.encrypted_rows)
    encrypted_cols = set(int(item) for item in partition.encrypted_cols)
    return {
        "shape": (B.shape[0], A.shape[1]),
        "A": [_operand(A[:, col], col in encrypted_cols, public_ctx) for col in range(A.shape[1])],
        "B": [_operand(B[row, :], row in encrypted_rows, public_ctx) for row in range(B.shape[0])],
    }


def upload_bytes(upload):
    """统计内积上传包的明文、密文和基础元数据。"""
    counts = {"metadata_bytes": 16, "ciphertext_bytes": 0, "plaintext_bytes": 0}
    for operand in (*upload["A"], *upload["B"]):
        kind, payload = operand
        if kind == "ct":
            counts["ciphertext_bytes"] += len(payload)
        else:
            counts["plaintext_bytes"] += np.asarray(payload).nbytes
    counts["payload_bytes"] = counts["ciphertext_bytes"] + counts["plaintext_bytes"]
    counts["total_bytes"] = counts["metadata_bytes"] + counts["payload_bytes"]
    return counts


def _load(operand, public_ctx):
    kind, payload = operand
    if kind == "ct":
        return ts.ckks_vector_from(public_ctx, payload)
    return np.asarray(payload, dtype=np.float64)


def _aggregate_plain_block(uploads, rows, cols, n_clients):
    value = None
    for upload in uploads:
        B = np.vstack([upload["B"][row][1] for row in rows])
        A = np.column_stack([upload["A"][col][1] for col in cols])
        client = B @ A
        value = client if value is None else value + client
    return value / n_clients


def _aggregate_a_encrypted(uploads, public_ctx, rows, cols, n_clients):
    columns = []
    for col in cols:
        accumulated = None
        for upload in uploads:
            encrypted_a = _load(upload["A"][col], public_ctx)
            plain_b = np.vstack([upload["B"][row][1] for row in rows])
            current = encrypted_a.matmul(plain_b.T)
            accumulated = current if accumulated is None else accumulated + current
        accumulated *= 1.0 / n_clients
        columns.append(accumulated.serialize())
    return columns


def _aggregate_b_encrypted(uploads, public_ctx, rows, cols, n_clients, expired):
    scalars = []
    loaded_b = [
        {row: _load(upload["B"][row], public_ctx) for row in rows}
        for upload in uploads
    ]
    loaded_a = [
        {col: _load(upload["A"][col], public_ctx) for col in cols}
        for upload in uploads
    ]
    for col in cols:
        column = []
        for row in rows:
            accumulated = None
            for client_index in range(len(uploads)):
                b = loaded_b[client_index][row]
                a = loaded_a[client_index][col]
                current = b.dot(a)
                accumulated = current if accumulated is None else accumulated + current
            accumulated *= 1.0 / n_clients
            column.append(accumulated.serialize())
            if expired():
                return None
        scalars.append(column)
    return scalars


def aggregate(uploads, public_ctx, partition, skeleton, skeleton_rows, skeleton_cols,
              n_clients, n_slots, time_budget=None):
    """服务端以内积方式聚合请求的完整矩阵或 skeleton 块。"""
    shape = tuple(uploads[0]["shape"])
    blocks = build_blocks(
        *shape,
        partition,
        skeleton,
        skeleton_rows=skeleton_rows,
        skeleton_cols=skeleton_cols,
        max_slots=n_slots,
    )
    started = time.perf_counter()

    def expired():
        return time_budget is not None and time.perf_counter() - started > time_budget

    results = []
    for block in blocks:
        rows = block.row_indices
        cols = block.col_indices
        if not block.encrypted:
            kind = "plain"
            payload = _aggregate_plain_block(uploads, rows, cols, n_clients)
        elif block.a_encrypted and not block.b_encrypted:
            kind = "ct_columns"
            payload = _aggregate_a_encrypted(uploads, public_ctx, rows, cols, n_clients)
        else:
            kind = "ct_scalars"
            payload = _aggregate_b_encrypted(
                uploads, public_ctx, rows, cols, n_clients, expired
            )
            if payload is None:
                return {
                    "feasible": False,
                    "note": f"内积逐元素 dot 超过时间预算 {time_budget}s",
                    "shape": shape,
                    "blocks": [],
                }
        results.append(
            {
                "row_indices": rows.tolist(),
                "col_indices": cols.tolist(),
                "kind": kind,
                "payload": payload,
            }
        )
    return {"feasible": True, "note": "", "shape": shape, "blocks": results}


def download_bytes(result):
    """统计内积聚合结果的明文、密文和元数据。"""
    counts = {"metadata_bytes": 0, "ciphertext_bytes": 0, "plaintext_bytes": 0}
    for block in result["blocks"]:
        counts["metadata_bytes"] += 8 * (
            len(block["row_indices"]) + len(block["col_indices"])
        )
        if block["kind"] == "plain":
            counts["plaintext_bytes"] += np.asarray(block["payload"]).nbytes
        elif block["kind"] == "ct_columns":
            counts["ciphertext_bytes"] += sum(len(item) for item in block["payload"])
        else:
            counts["ciphertext_bytes"] += sum(
                len(item) for column in block["payload"] for item in column
            )
    counts["payload_bytes"] = counts["ciphertext_bytes"] + counts["plaintext_bytes"]
    counts["total_bytes"] = counts["metadata_bytes"] + counts["payload_bytes"]
    return counts


def decrypt_result(result, secret_ctx):
    """客户端解密内积结果并按坐标组装矩阵。"""
    matrix = np.zeros(tuple(result["shape"]), dtype=np.float64)
    ciphertext_elements = 0
    plaintext_elements = 0
    for block in result["blocks"]:
        rows = np.asarray(block["row_indices"], dtype=int)
        cols = np.asarray(block["col_indices"], dtype=int)
        if block["kind"] == "plain":
            matrix[np.ix_(rows, cols)] = np.asarray(block["payload"], dtype=np.float64)
            plaintext_elements += rows.size * cols.size
        elif block["kind"] == "ct_columns":
            for col, payload in zip(cols, block["payload"]):
                values = np.asarray(ts.ckks_vector_from(secret_ctx, payload).decrypt())
                matrix[rows, col] = values[:rows.size]
            ciphertext_elements += rows.size * cols.size
        else:
            for col, column in zip(cols, block["payload"]):
                for row, payload in zip(rows, column):
                    value = ts.ckks_vector_from(secret_ctx, payload).decrypt()[0]
                    matrix[row, col] = value
            ciphertext_elements += rows.size * cols.size
    return matrix, {
        "ciphertext_elements": ciphertext_elements,
        "plaintext_elements": plaintext_elements,
        "total_elements": int(np.prod(result["shape"])),
    }
