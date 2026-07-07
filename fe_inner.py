"""内积法（标准矩阵乘法）工具：客户端加密上传、服务端内积聚合、客户端解密。

与外积法（fe_client/fe_server）对照。同一聚合 ΔW_mean = (1/N)·Σ_i B_i·A_i，本模块用
标准矩阵乘法的「内积」视角实现，即把收缩维 rank 放进密文槽位，靠跨槽位求和得到结果：

    列 j：ΔW_i[:,j] = B_i · A_i[:,j]        （A_i[:,j] 长 rank）
    行 k：ΔW_i[k,:] = B_i[k,:] · A_i        （B_i[k,:] 长 rank）

每个输出元 ΔW[k,j] = ⟨B_i[k,:], A_i[:,j]⟩ 是长 rank 的内积；内积 = 逐元素乘后把 rank
个槽位加起来，这是**跨槽位求和**，CKKS 必须借 galois 旋转（matmul/dot 内部完成）实现。
因此本模块的 context 必须带 galois key，且客户端只需上传「收缩维在槽位」的 rank 长密文，
与外积法把因子 repeat/tile 到 dim 长形成鲜明的空间冗余对比。

加密程度：
  half —— 只加密 A（按列，长 rank），B 明文。列走 matmul(明文 B)，行走 dot(明文 B[k,:])。
  full —— A、B 都加密（A 按列、B 按行，均长 rank）。每个输出元用一次 ct×ct dot。

计算量：full 内积每个输出元一次 ct×ct dot，完整矩阵需 O(d²·N) 次，代价高；main 对
过大的格设超时保护并如实标注「不可行」，绝不编造数字。
"""

import time
import numpy as np
import tenseal as ts


# ── 客户端：加密上传 ────────────────────────────────────────────────────────

def encrypt_upload_inner(B_i, A_i, public_ctx, rank, dim, enc_level):
    """内积法上传包：A 按列加密（长 rank）；full 时 B 也按行加密（长 rank）。

    收缩维 rank 落在槽位里，故每条密文只用 rank 个槽位——这正是与外积法（因子被
    repeat/tile 到 dim 长）对比空间冗余的关键。

    :param B_i: (dim, rank)
    :param A_i: (rank, dim)
    :param enc_level: "half"（仅加密 A）/ "full"（A、B 都加密）
    :return: dict(encA_cols=[dim 条长 rank 密文 bytes],
                  encB_rows=[dim 条 或 None], plain_B=ndarray 或 None)
    """
    B_i = B_i.astype(np.float64)
    A_i = A_i.astype(np.float64)

    # A 按列加密：encA_cols[j] = Enc(A[:,j])，长 rank。
    encA_cols = [ts.ckks_vector(public_ctx, A_i[:, j].tolist()).serialize()
                 for j in range(dim)]

    if enc_level == "full":
        # B 按行加密：encB_rows[k] = Enc(B[k,:])，长 rank。
        encB_rows = [ts.ckks_vector(public_ctx, B_i[k, :].tolist()).serialize()
                     for k in range(dim)]
        plain_B = None
    else:
        # half：B 保持明文，供服务端做 matmul/dot 的明文操作数。
        encB_rows = None
        plain_B = B_i

    return dict(encA_cols=encA_cols, encB_rows=encB_rows, plain_B=plain_B)


def upload_bytes_inner(upload):
    """统计内积法单客户端上传字节：密文按序列化长度，明文 B 按 float64 字节。"""
    total = sum(len(s) for s in upload["encA_cols"])
    if upload["encB_rows"] is not None:
        total += sum(len(s) for s in upload["encB_rows"])
    if upload["plain_B"] is not None:
        total += upload["plain_B"].astype(np.float64).nbytes
    return total


# ── 服务端：内积聚合 ────────────────────────────────────────────────────────

def aggregate_inner(uploads, public_ctx, n_clients, rank, dim,
                    row_idx, col_idx, enc_level, time_budget=None):
    """内积法密文域聚合，返回骨架/完整所需的行列（跨客户端平均）。

    :param row_idx: 需重建的行索引；:param col_idx: 需重建的列索引
    :param enc_level: "half"/"full"
    :param time_budget: 秒；超过则中止并返回 feasible=False（不编造数字）
    :return: dict(col_map={j:ndarray}, row_map={k:ndarray},
                  down_bytes, feasible, note)
    """
    inv_n = 1.0 / n_clients
    col_idx = [int(j) for j in col_idx]
    row_idx = [int(k) for k in row_idx]
    t0 = time.time()

    # 预加载各客户端密文/明文操作数。
    loaded = []
    for up in uploads:
        encA = [ts.ckks_vector_from(public_ctx, s) for s in up["encA_cols"]]
        if enc_level == "full":
            encB = [ts.ckks_vector_from(public_ctx, s) for s in up["encB_rows"]]
            loaded.append((encA, encB, None))
        else:
            loaded.append((encA, None, up["plain_B"]))

    col_ct = {}   # j -> 该列的密文表示（仅跨客户端求和，未乘 1/N）
    row_ct = {}   # k -> 该行的密文表示

    def _expired():
        return time_budget is not None and (time.time() - t0) > time_budget

    # 密文表示有两种形态（见 _serialize_vec）：
    #   "packed"：一条长 dim 的密文（half 列的 matmul 输出）——省下发字节；
    #   "scalars"：dim 条标量密文（dot 逐元素输出）——避开 pack_vectors 的精度损失。
    # 为何不用 pack_vectors 合并标量：实测 pack_vectors 合并大量标量密文会把误差抬到
    # 1e-3 量级，再经 uniform 骨架病态交叉块 M_r 求逆（cond~1e6）放大成发散。逐标量
    # 各自解密可保持 1e-7 精度，代价是下发字节增多——本模块如实统计该字节，不掩盖。
    #
    # 平均系数 1/N 不在密文域乘：matmul/dot 已消耗乘法深度，再乘明文会触发
    # 「scale out of bounds」。N 是公开常数，除 N 放到解密后完成，不损隐私、省一层深度。
    for j in col_idx:
        if enc_level == "half":
            # matmul(明文 B.T)：encA[j] 长 rank → dim，精度高，可用单条打包密文。
            acc = None
            for (encA, encB, plainB) in loaded:
                cj = encA[j].matmul(plainB.T)
                acc = cj if acc is None else acc + cj
            col_ct[j] = ("packed", acc)
        else:
            # full：列 j 第 k 元 = Σ_i dot(encB_i[k], encA_i[j])，逐标量保留精度。
            scal = []
            for k in range(dim):
                acc = None
                for (encA, encB, plainB) in loaded:
                    d = encB[k].dot(encA[j])
                    acc = d if acc is None else acc + d
                scal.append(acc)
            col_ct[j] = ("scalars", scal)
        if _expired():
            return dict(feasible=False, note=f"超时>{time_budget}s（列阶段）",
                        col_bytes={}, row_bytes={}, down_bytes=0)

    # 行 k：ΔW_i[k,:] = B_i[k,:] · A_i。行的每个元都是 dot，逐标量保精度。
    for k in row_idx:
        scal = []
        for j in range(dim):
            acc = None
            for (encA, encB, plainB) in loaded:
                if enc_level == "half":
                    # 行 k 第 j 元 = dot(明文 B[k,:], encA[j])，跨 rank 槽位求和。
                    d = encA[j].dot(plainB[k, :].tolist())
                else:
                    d = encB[k].dot(encA[j])
                acc = d if acc is None else acc + d
            scal.append(acc)
        row_ct[k] = ("scalars", scal)
        if _expired():
            return dict(feasible=False, note=f"超时>{time_budget}s（行阶段）",
                        col_bytes={}, row_bytes={}, down_bytes=0)

    col_bytes = {j: _serialize_vec(v) for j, v in col_ct.items()}
    row_bytes = {k: _serialize_vec(v) for k, v in row_ct.items()}
    down = (sum(_vec_nbytes(v) for v in col_bytes.values())
            + sum(_vec_nbytes(v) for v in row_bytes.values()))
    return dict(feasible=True, note="", col_bytes=col_bytes, row_bytes=row_bytes,
                down_bytes=down)


def _serialize_vec(vec_repr):
    """把 ("packed", ct) 或 ("scalars", [ct,...]) 序列化为可下发形态。"""
    kind, payload = vec_repr
    if kind == "packed":
        return ("packed", payload.serialize())
    return ("scalars", [ct.serialize() for ct in payload])


def _vec_nbytes(ser_vec):
    """统计一个序列化向量表示的下发字节。"""
    kind, payload = ser_vec
    if kind == "packed":
        return len(payload)
    return sum(len(s) for s in payload)


# ── 客户端：解密 ────────────────────────────────────────────────────────────

def decrypt_map_inner(bytes_map, secret_ctx, dim, n_clients):
    """解密 {索引: 序列化向量表示} → {索引: 长 dim 向量}，并完成 1/N 平均。

    向量表示分两形态：
      "packed"  —— 一条长 dim 密文，解密后取前 dim 槽；
      "scalars" —— dim 条标量密文，逐条解密取槽 0 拼成向量（避开 pack_vectors 精度损失）。
    服务端只做了跨客户端求和（未乘 1/N），故此处统一除以 N 得平均。

    :param n_clients: 客户端数 N
    """
    out = {}
    for idx, ser_vec in bytes_map.items():
        kind, payload = ser_vec
        if kind == "packed":
            vec = np.array(ts.ckks_vector_from(secret_ctx, payload).decrypt())[:dim]
        else:
            vec = np.array([np.array(ts.ckks_vector_from(secret_ctx, s).decrypt())[0]
                            for s in payload])
        out[int(idx)] = vec / n_clients
    return out
