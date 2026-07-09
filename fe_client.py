"""客户端工具：加密上传 A/B 因子、私钥解密下发密文。

职责边界（由 main 编排调用，所有参数 main 传入）：客户端本地不计算 ΔW_i=B_i·A_i、
不聚合，只把重建所需的 A/B 因子素材加密（或按半加密留明文）上传；解密阶段用私钥
还原服务端下发的密文。

重建原理（全程逐元素运算，不需 galois 旋转）：
  列 j：ΔW_i[:,j] = Σ_c A_i[c,j]·b_c    （b_c = B_i[:,c]，长 dim）
  行 k：ΔW_i[k,:] = Σ_c B_i[k,c]·a_c    （a_c = A_i[c,:]，长 dim）
把「标量×向量」用「广播标量向量 ⊙ 因子向量」实现，即可只靠逐元素乘完成。

打包（packing）：一条密文槽位数 = poly_modulus_degree/2，可容纳 g = 槽位数/dim 个
长 dim 的段。把 g 个列的重建结果并排放进同一条密文的相邻段，密文条数从「每列一条」
降到「每 g 列一条」，显著省网络；dim 越大 g 越小，dim≥槽位数时 g=1、打包自动退化为
不打包（物理必然，非人为）。

加密程度（enc level）：
  full —— A、B 因子都加密，服务端做密文×密文；
  half —— 只加密 A 派生量、B 派生量走明文，服务端做密文×明文（更省时省钥）。
两种模式下「A 派生的操作数」恒为密文，故乘积恒为密文、结果始终可由私钥解密。
"""

import numpy as np
import tenseal as ts


def group_size(n_slots, dim, packing):
    """打包时每条密文并排容纳的段数 g。

    packed 取 g = 槽位数 // dim（至少 1）；unpacked 恒为 1（每段独立一段一条密文）。
    dim ≥ 槽位数时 packed 也只能取 1，打包自然退化为不打包。

    防静默降级：dim 超过槽位数时，tenseal 不报错而是打印 WARNING、把向量拆进多条
    密文并禁用 matmul 等操作，导致外积槽位对齐失效（结果可能错）或内积延迟报错。
    故此处显式拦截 dim > n_slots，把隐蔽的静默降级提前暴露为明确错误。

    :param n_slots: 一条密文可用明文槽位数（= poly_modulus_degree/2），由 main 传入；
                    tenseal 0.3.16 的 Context 未暴露该值的读取接口，故显式传参。
    :raises ValueError: dim 超过单条密文槽位数，任何打包方式都无法保证槽位对齐
    """
    if dim > n_slots:
        raise ValueError(
            f"dim={dim} 超过单条密文槽位数 {n_slots}，无法保证槽位对齐："
            f"请增大 POLY_MODULUS_DEGREE（当前槽位=degree/2）使 2×dim≤degree，"
            f"或改用分块打包。")
    if packing == "unpacked":
        return 1
    return max(1, n_slots // dim)


def _chunks(indices, g):
    """把索引序列按每组 g 个切成若干组，返回 list[list[int]]。"""
    idx = [int(i) for i in indices]
    return [idx[s:s + g] for s in range(0, len(idx), g)]


def _term(vec_part, scalar_part, enc_vec, public_ctx):
    """构造一个待服务端相乘的因子对（vec_part ⊙ scalar_part 中的一个操作数组合）。

    :param vec_part: 因子向量拼接后的明文 ndarray（长 dim×组段数）
    :param scalar_part: 广播标量拼接后的明文 ndarray（同长）
    :param enc_vec: True 表示「向量来自 A 的加密派生量」——此时 vec 加密、scalar 明文；
                    False 表示向量来自 B、在 half 下为明文，scalar 来自 A 需加密。
    :param public_ctx: 公开 context
    :return: (op1, op2)，每个 op 为 ("ct", bytes) 或 ("plain", ndarray)
    """
    if enc_vec:
        # A 派生的向量加密，B 派生的标量明文。
        return (("ct", ts.ckks_vector(public_ctx, vec_part.tolist()).serialize()),
                ("plain", scalar_part))
    # B 派生的向量明文，A 派生的标量加密。
    return (("plain", vec_part),
            ("ct", ts.ckks_vector(public_ctx, scalar_part.tolist()).serialize()))


def _term_full(vec_part, scalar_part, public_ctx):
    """full 加密：向量与标量都加密，服务端做密文×密文。"""
    return (("ct", ts.ckks_vector(public_ctx, vec_part.tolist()).serialize()),
            ("ct", ts.ckks_vector(public_ctx, scalar_part.tolist()).serialize()))


def encrypt_upload(B_i, A_i, public_ctx, rank, dim, row_idx, col_idx,
                   enc_level, packing, n_slots):
    """按加密程度与打包方式，生成一个客户端的上传包。

    :param B_i: LoRA 因子 B，形状 (dim, rank)
    :param A_i: LoRA 因子 A，形状 (rank, dim)
    :param public_ctx: 公开 context（加密用，无私钥）
    :param rank: LoRA 因子秩
    :param dim: 矩阵维度
    :param row_idx: 服务端将重建的行索引集合
    :param col_idx: 服务端将重建的列索引集合
    :param enc_level: "full" 或 "half"
    :param packing: "packed" 或 "unpacked"
    :param n_slots: 一条密文可用槽位数（= poly_modulus_degree/2）
    :return: dict 上传包，含 col_groups / row_groups，每组为
             {"indices": [...], "terms": [(op1, op2), ...共 rank 项...]}
    """
    B_i = B_i.astype(np.float64)
    A_i = A_i.astype(np.float64)
    g = group_size(n_slots, dim, packing)

    col_groups = []
    for grp in _chunks(col_idx, g):
        m = len(grp)
        terms = []
        for c in range(rank):
            # 向量段：b_c 在每个列段重复出现 → tile(b_c, m)。
            vec_part = np.tile(B_i[:, c], m)
            # 标量段：列段 s 用 A_i[c, grp[s]] 广播满 dim。
            scalar_part = np.concatenate([[A_i[c, j]] * dim for j in grp])
            if enc_level == "full":
                terms.append(_term_full(vec_part, scalar_part, public_ctx))
            else:
                # half：列的向量来自 B（明文），标量来自 A（加密）→ enc_vec=False。
                terms.append(_term(vec_part, scalar_part, enc_vec=False,
                                   public_ctx=public_ctx))
        col_groups.append({"indices": grp, "terms": terms})

    row_groups = []
    for grp in _chunks(row_idx, g):
        m = len(grp)
        terms = []
        for c in range(rank):
            # 行的向量段来自 a_c（A），在每个行段重复。
            vec_part = np.tile(A_i[c, :], m)
            # 标量段来自 B_i[grp[s], c] 广播。
            scalar_part = np.concatenate([[B_i[k, c]] * dim for k in grp])
            if enc_level == "full":
                terms.append(_term_full(vec_part, scalar_part, public_ctx))
            else:
                # half：行的向量来自 A（加密）→ enc_vec=True，标量来自 B（明文）。
                terms.append(_term(vec_part, scalar_part, enc_vec=True,
                                   public_ctx=public_ctx))
        row_groups.append({"indices": grp, "terms": terms})

    return dict(col_groups=col_groups, row_groups=row_groups, group_size=g)


def upload_bytes(upload):
    """统计一个上传包的总字节：密文按序列化长度，明文按 float64 原始字节。"""
    total = 0
    for groups in (upload["col_groups"], upload["row_groups"]):
        for grp in groups:
            for op1, op2 in grp["terms"]:
                for kind, payload in (op1, op2):
                    if kind == "ct":
                        total += len(payload)
                    else:
                        total += payload.astype(np.float64).nbytes
    return total


def decrypt_groups(group_bytes, secret_ctx, dim):
    """解密服务端下发的分组密文，拆回 {索引: 长 dim 向量}。

    :param group_bytes: list[{"indices": [...], "ct": bytes}]
    :param secret_ctx: 含私钥 context
    :param dim: 每段长度
    :return: {索引: np 一维向量}
    """
    out = {}
    for grp in group_bytes:
        flat = np.array(ts.ckks_vector_from(secret_ctx, grp["ct"]).decrypt())
        for s, idx in enumerate(grp["indices"]):
            out[int(idx)] = flat[s * dim:(s + 1) * dim]
    return out
