"""服务端工具：密文域聚合 A/B 因子，下发聚合后的密文。

服务端只持公开 context，全程不解密。对每个客户端上传包中的每个分组，用逐元素乘
（密文×密文或密文×明文）累加 rank 项外积，得到该组的 ΔW_i 重建段；再跨客户端相加、
乘 1/N 求平均，最后序列化下发。

聚合语义（正确性关键）：先对每个客户端各自重建 B_i·A_i、再跨客户端相加，
而非 (Σ_i B_i)·(Σ_i A_i)——后者会引入客户端间交叉项 B_i·A_j (i≠j)。本模块逐客户端
独立重建后累加，天然满足联邦平均聚合 ΔW_mean = (1/N)·Σ_i B_i·A_i。

打包/加密程度差异全部封装在上传包的 term 结构里（("ct",bytes)/("plain",ndarray)），
服务端按标记加载并选择 ct×ct 或 ct×plain，无需区分调用路径。
"""

import numpy as np
import tenseal as ts


def _load_op(kind_payload, public_ctx):
    """把上传包里的操作数还原为可运算对象。

    :param kind_payload: ("ct", bytes) 或 ("plain", ndarray)
    :return: CKKSVector（密文）或 np.ndarray（明文）
    """
    kind, payload = kind_payload
    if kind == "ct":
        return ts.ckks_vector_from(public_ctx, payload)
    return np.asarray(payload, dtype=np.float64)


def _mul(op1, op2):
    """逐元素乘两个操作数，至少一方为密文时结果为密文。

    tenseal 的 CKKSVector 同时支持 ct×ct 与 ct×plain(ndarray)，故只需保证密文在左。
    """
    a_ct = isinstance(op1, ts.CKKSVector)
    b_ct = isinstance(op2, ts.CKKSVector)
    if a_ct:
        return op1 * op2
    if b_ct:
        return op2 * op1
    # 两个明文相乘不应出现：每组必有一个 A 派生的加密操作数。
    raise ValueError("分组中缺少密文操作数，结果无法保持加密")


def _aggregate_groups(uploads, public_ctx, which, inv_n):
    """对所有客户端的某一类分组（列或行）做密文域重建 + 跨客户端平均。

    :param which: "col_groups" 或 "row_groups"
    :param inv_n: 平均系数 1/N（明文标量）
    :return: list[{"indices": [...], "ct": CKKSVector}]，顺序与分组一致
    """
    n_groups = len(uploads[0][which])
    out = []
    for gi in range(n_groups):
        indices = uploads[0][which][gi]["indices"]
        acc = None
        for up in uploads:
            grp = up[which][gi]
            # 累加本组 rank 项外积，得到该客户端在此组的重建段。
            client_seg = None
            for op1_raw, op2_raw in grp["terms"]:
                term = _mul(_load_op(op1_raw, public_ctx),
                            _load_op(op2_raw, public_ctx))
                client_seg = term if client_seg is None else client_seg + term
            acc = client_seg if acc is None else acc + client_seg
        acc *= inv_n   # 密文 × 明文标量，跨客户端求平均
        out.append({"indices": indices, "ct": acc})
    return out


def aggregate(uploads, public_ctx, n_clients):
    """密文域聚合：重建上传包覆盖的行列分组并跨客户端平均。

    覆盖范围（骨架 or 完整）由客户端上传时的索引决定，服务端不感知区别。

    :param uploads: 各客户端上传包列表（fe_client.encrypt_upload 的返回）
    :param public_ctx: 公开 context
    :param n_clients: 客户端数 N
    :return: (col_group_bytes, row_group_bytes)，
             每项为 list[{"indices": [...], "ct": 序列化 bytes}]
    """
    inv_n = 1.0 / n_clients
    col_out = _aggregate_groups(uploads, public_ctx, "col_groups", inv_n)
    row_out = _aggregate_groups(uploads, public_ctx, "row_groups", inv_n)

    col_bytes = [{"indices": g["indices"], "ct": g["ct"].serialize()} for g in col_out]
    row_bytes = [{"indices": g["indices"], "ct": g["ct"].serialize()} for g in row_out]
    return col_bytes, row_bytes


def download_bytes(group_bytes):
    """统计一批下发分组密文的总序列化字节。"""
    return sum(len(g["ct"]) for g in group_bytes)
