"""内积法（标准矩阵乘法）工具：客户端加密上传、服务端内积聚合、客户端解密。

与外积法（fe_client/fe_server）对照。同一聚合 ΔW_mean = (1/N)·Σ_i B_i·A_i，本模块用
标准矩阵乘法的「内积」视角实现，把收缩维 rank 放进密文槽位，靠跨槽位求和得到结果：

    列 j：ΔW_i[:,j] = B_i · A_i[:,j]        （A_i[:,j] 长 rank）
    行 k：ΔW_i[k,:] = B_i[k,:] · A_i        （B_i[k,:] 长 rank）

每个输出元 ΔW[k,j] = ⟨B_i[k,:], A_i[:,j]⟩ 是长 rank 的内积；内积 = 逐元素乘后把 rank
个槽位加起来，属**跨槽位求和**，CKKS 借 galois 旋转（matmul 内部完成）实现，故 context
必须带 galois key。

加密程度：
  half —— 只加密 A，B 明文。列/行都用 matmul(明文 B) 完成，精度高（~1e-9）。
  full —— A、B 都加密，每个输出元一次 ct×ct dot。O(d²·N) 次，代价高，设超时保护。

打包（防退化，仅 half 可行）：
  内积段长是收缩维 rank（远小于 dim），一条密文可容纳 slots//rank 列（如 4096//4=1024），
  故把 A 的多列拼进一条密文、服务端用**块对角明文矩阵**一次 matmul 还原整组，块对角结构
  保证各列内积只在自己 rank 段内求和、互不串扰。A 全 dim 列打包后既供列重建、又供行重建
  （行 k 用 B[k,:] 的块对角），上传密文条数从 dim 降到 ceil(dim/g)，真正省网络。
  full 打包不可行：ct×ct 无块内求和手段（详见 aggregate_inner 调用方对 full+packed 的
  不可行标注），tenseal 0.3.16 只有全局 sum/dot，会把各列内积串扰成一个数。
"""

import time
import numpy as np
import tenseal as ts


def inner_group_size(n_slots, rank, packing, col_count, dim):
    """内积打包时一条密文并排容纳的列数 g（块对角打包）。

    unpacked 恒为 1；packed 需同时满足两条槽位约束：
      ① A 打包段长 rank，g·rank ≤ n_slots（上传密文装得下）；
      ② 服务端列还原后 g 列并排（每列 dim 元素），g·dim ≤ n_slots（结果密文装得下）；
    故 g = min(n_slots//rank, n_slots//dim, col_count)。当 dim ≥ n_slots 时 g=1，
    打包退化为不打包（物理必然）——例如 dim=3200、n_slots=4096 → g=1。
    这条约束若被忽略，服务端块对角矩阵尺寸 (g·rank × g·dim) 会以 GB 级 numpy 爆内存。

    :param n_slots: 一条密文槽位数（= poly_modulus_degree/2）
    :param rank: LoRA 因子秩，即每列内积段长
    :param packing: "packed"/"unpacked"
    :param col_count: 待打包列数，g 不超过它
    :param dim: 矩阵维度，用于约束 ②
    """
    if packing == "unpacked":
        return 1
    return max(1, min(n_slots // rank, n_slots // dim, col_count))


# ── 客户端：加密上传 ────────────────────────────────────────────────────────

def encrypt_upload_inner(B_i, A_i, public_ctx, rank, dim, enc_level,
                         packing, n_slots):
    """内积法上传包。

    统一约定：A 覆盖全部 dim 列（打包时供列/行两阶段共用），B 视加密程度决定明文/密文。

    :param packing: "packed"/"unpacked"；full 不打包（打包不可行，调用方另标不可行）
    :param n_slots: 密文槽位数
    :return: dict，含 enc_level/packing/group_size 与密文载荷
    """
    B_i = B_i.astype(np.float64)
    A_i = A_i.astype(np.float64)
    all_cols = list(range(dim))

    if enc_level == "full":
        # full：不打包，A 按列 + B 按行各加密长 rank 密文。
        encA_cols = [ts.ckks_vector(public_ctx, A_i[:, j].tolist()).serialize()
                     for j in range(dim)]
        encB_rows = [ts.ckks_vector(public_ctx, B_i[k, :].tolist()).serialize()
                     for k in range(dim)]
        return dict(enc_level="full", packing=packing, group_size=1,
                    encA_cols=encA_cols, encB_rows=encB_rows, plain_B=None)

    # half：B 明文
    if packing == "unpacked":
        # encA_cols（长 rank，dim 条）供列阶段 matmul(明文 B.T) 复用；
        # encA_rows（长 dim，rank 条）供行阶段 ct×标量 累加使用——绕开 tenseal
        # 0.3.16 中 ct.dot(plain_list) 结果 pack_vectors 会串扰的坑（size=1 但底层
        # 槽位残留），rank 条长 dim 密文额外上传只增加 ~1.3 MB 但换来干净行阶段。
        encA_cols = [ts.ckks_vector(public_ctx, A_i[:, j].tolist()).serialize()
                     for j in range(dim)]
        encA_rows = [ts.ckks_vector(public_ctx, A_i[c, :].tolist()).serialize()
                     for c in range(rank)]
        return dict(enc_level="half", packing="unpacked", group_size=1,
                    encA_cols=encA_cols, encA_rows=encA_rows,
                    encB_rows=None, plain_B=B_i)

    # half + packed：A 全列按每组 g 列拼进一条密文（供列阶段块对角 matmul）；
    # 额外上传 rank 条长 dim 的 encA_rows（供行阶段 ct×标量 累加，跟 unpacked 行阶段
    # 共用同一套 _agg_rows_matmul_half）——行阶段不再复用打包 A，避免为每组小 matmul
    # 白算无骨架列的组。rank 条附加密文约 rank×ct_bytes ~ 1.3 MB，可忽略。
    g = inner_group_size(n_slots, rank, "packed", dim, dim)
    col_groups = []
    for s in range(0, dim, g):
        grp = all_cols[s:s + g]
        packed = np.concatenate([A_i[:, j] for j in grp])   # 长 len(grp)×rank
        col_groups.append({
            "indices": grp,
            "ct": ts.ckks_vector(public_ctx, packed.tolist()).serialize(),
        })
    encA_rows = [ts.ckks_vector(public_ctx, A_i[c, :].tolist()).serialize()
                 for c in range(rank)]
    return dict(enc_level="half", packing="packed", group_size=g,
                col_groups=col_groups, encA_rows=encA_rows, plain_B=B_i)


def upload_bytes_inner(upload):
    """统计内积法单客户端上传字节：密文按序列化长度，明文 B 按 float64 字节。"""
    total = 0
    if upload.get("col_groups") is not None:
        total += sum(len(g["ct"]) for g in upload["col_groups"])
    if upload.get("encA_cols") is not None:
        total += sum(len(s) for s in upload["encA_cols"])
    if upload.get("encA_rows") is not None:
        total += sum(len(s) for s in upload["encA_rows"])
    if upload.get("encB_rows") is not None:
        total += sum(len(s) for s in upload["encB_rows"])
    if upload.get("plain_B") is not None:
        total += upload["plain_B"].astype(np.float64).nbytes
    return total


# ── 服务端：内积聚合 ────────────────────────────────────────────────────────

def aggregate_inner(uploads, public_ctx, n_clients, rank, dim,
                    row_idx, col_idx, enc_level, packing, time_budget=None):
    """内积法密文域聚合（跨客户端求和，未乘 1/N，除 N 移至解密后）。

    - half+packed  ：列走块对角 matmul、按 col_idx 过滤无骨架列的组（骨架开时
                     dim≫g 场景可省 dim/|J| 倍白算）；行走 encA_rows ct×标量累加
                     （跟 unpacked 行阶段同一套 _agg_rows_matmul_half，避免为每组
                     跑小 matmul）。
    - half+unpacked：列每 j 一次 matmul(明文 B.T)；行用 encA_rows ct×标量累加（避开
                     tenseal 0.3.16 中 ct.dot(plain_list) 结果 pack_vectors 的串扰坑）。
    - full         ：逐元素 ct×ct dot（列、行皆是），dim 标量 pack_vectors 合到一条 ct
                     下发；full+packed 由调用方标不可行、不入此函数。

    :param n_clients: 当前实现里 1/N 移至解密后完成（避免消耗额外乘法深度），本参数
                      保留在签名里供调用点显式传参、留作未来密文域求平均的入口。
    :param packing: "packed"/"unpacked"
    :param time_budget: 秒；超时返回 feasible=False
    :return: dict(feasible, note, col_bytes, row_bytes, down_bytes)
    """
    col_idx = [int(j) for j in col_idx]
    row_idx = [int(k) for k in row_idx]
    t0 = time.time()

    def _expired():
        return time_budget is not None and (time.time() - t0) > time_budget

    if enc_level == "half" and packing == "packed":
        # 列走块对角 matmul、按 col_idx 过滤组；行走 encA_rows（跟 unpacked 一致）。
        col_bytes = _agg_packed_cols(uploads, public_ctx, rank, dim, col_idx)
        row_bytes = _agg_rows_matmul_half(uploads, public_ctx, rank, row_idx)
    elif enc_level == "half":
        col_bytes = _agg_cols_matmul(uploads, public_ctx, col_idx)
        row_bytes = _agg_rows_matmul_half(uploads, public_ctx, rank, row_idx)
    else:
        col_bytes = _agg_cols_scalar(uploads, public_ctx, dim, col_idx, _expired)
        if col_bytes is None:
            return dict(feasible=False, note=f"超时>{time_budget}s（列阶段）",
                        col_bytes={}, row_bytes={}, down_bytes=0)
        row_bytes = _agg_rows_scalar(uploads, public_ctx, dim, row_idx, _expired)
        if row_bytes is None:
            return dict(feasible=False, note=f"超时>{time_budget}s（行阶段）",
                        col_bytes={}, row_bytes={}, down_bytes=0)

    down = (sum(_vec_nbytes(v) for v in col_bytes.values())
            + sum(_vec_nbytes(v) for v in row_bytes.values()))
    return dict(feasible=True, note="", col_bytes=col_bytes, row_bytes=row_bytes,
                down_bytes=down)


def _agg_packed_cols(uploads, public_ctx, rank, dim, col_idx):
    """half+packed 列：只算「组内含骨架列」的组，跳过其余组的 matmul 与下发。

    dim 接近 n_slots 时 g=1 打包退化，n_groups=dim；骨架开只需 r=|J| 列，若不过滤会把
    dim/r 倍计算与下发白算——dim=3200/r=16 时白算 200 倍。demo（dim=64、g=64、n_groups=1）
    时唯一那组本就覆盖全列，此过滤不改变行为、无副作用。

    :return: col_bytes 以 ("grp", gi) 为键，值 ("packed_group", indices, 序列化密文)
    """
    n_groups = len(uploads[0]["col_groups"])
    col_set = set(int(j) for j in col_idx)
    col_bytes = {}
    for gi in range(n_groups):
        indices = uploads[0]["col_groups"][gi]["indices"]
        if not any(int(j) in col_set for j in indices):
            continue   # 无骨架列，跳过整段 matmul + 下发
        g = len(indices)
        acc = None
        for up in uploads:
            ev = ts.ckks_vector_from(public_ctx, up["col_groups"][gi]["ct"])
            B = up["plain_B"]
            M = np.zeros((g * rank, g * dim), dtype=np.float64)
            for s in range(g):
                M[s * rank:(s + 1) * rank, s * dim:(s + 1) * dim] = B.T
            seg = ev.matmul(M)
            acc = seg if acc is None else acc + seg
        col_bytes[("grp", gi)] = ("packed_group", indices, acc.serialize())
    return col_bytes


def _agg_cols_matmul(uploads, public_ctx, col_idx):
    """half+unpacked 列：每列一次 matmul(明文 B.T)，单条长 dim 密文。"""
    out = {}
    for j in col_idx:
        acc = None
        for up in uploads:
            ev = ts.ckks_vector_from(public_ctx, up["encA_cols"][j])
            cj = ev.matmul(up["plain_B"].T)
            acc = cj if acc is None else acc + cj
        out[j] = ("packed", acc.serialize())
    return out


def _agg_cols_scalar(uploads, public_ctx, dim, col_idx, expired):
    """full 列：每列逐元素 ct×ct dot，dim 个标量 pack 成一条打包 ct 下发。

    下发压缩：从每列 dim 条独立 ct（dim×327KB）压到每列 1 条打包 ct（1×327KB），
    等于 dim 倍减负。pack_vectors 引入的额外精度损失实测优于逐标量解密（1e-5 vs 1e-4）
    ——因为 pack 后只做一次 decrypt 的舍入。超时返回 None。
    """
    loaded = _load_full(uploads, public_ctx)
    out = {}
    for j in col_idx:
        scal = []
        for k in range(dim):
            acc = None
            for (encA, encB) in loaded:
                d = encB[k].dot(encA[j])
                acc = d if acc is None else acc + d
            scal.append(acc)
        packed_ct = ts.CKKSVector.pack_vectors(scal)
        out[j] = ("packed", packed_ct.serialize())
        if expired():
            return None
    return out


def _agg_rows_matmul_half(uploads, public_ctx, rank, row_idx):
    """half 行阶段（packed/unpacked 共用）：用 encA_rows（rank 条长 dim 密文）做
    ct×标量 累加。

    行 k：ΔW_i[k,:] = Σ_c B_i[k,c] · A_i[c,:] = Σ_c B_i[k,c] · encA_rows[c]。
    每客户端每行 rank 次 ct×plain-scalar（不消耗深度层数、结果保持长 dim 干净密文），
    避开 ct.dot(plain_list)→size=1→pack 串扰的 tenseal 坑；对 packed 而言，
    还避免为每组重复跑小 matmul、把无骨架列的组也白算。
    每行下发一条长 dim packed ct。
    """
    loaded_rows = [[ts.ckks_vector_from(public_ctx, s) for s in up["encA_rows"]]
                   for up in uploads]
    out = {}
    for k in row_idx:
        acc = None
        for ci, up in enumerate(uploads):
            plainB = up["plain_B"]
            for c in range(rank):
                term = loaded_rows[ci][c] * float(plainB[k, c])
                acc = term if acc is None else acc + term
        out[k] = ("packed", acc.serialize())
    return out


def _agg_rows_scalar(uploads, public_ctx, dim, row_idx, expired):
    """full 行：逐标量 ct×ct dot，dim 个标量 pack 到一条 ct 下发。

    仅 full 分支使用：ct.dot(ct) 经重线性化+rescale 后是干净单槽 ct，pack_vectors 可安全
    组装。下发字节从 dim×327KB 压到 1×327KB。超时返回 None。
    """
    loaded = _load_full(uploads, public_ctx)
    out = {}
    for k in row_idx:
        scal = []
        for j in range(dim):
            acc = None
            for (encA, encB) in loaded:
                d = encB[k].dot(encA[j])
                acc = d if acc is None else acc + d
            scal.append(acc)
        packed_ct = ts.CKKSVector.pack_vectors(scal)
        out[k] = ("packed", packed_ct.serialize())
        if expired():
            return None
    return out


def _load_full(uploads, public_ctx):
    """full：加载 A 列密文（dict j->ct）与 B 行密文（list）。"""
    loaded = []
    for up in uploads:
        encA = {j: ts.ckks_vector_from(public_ctx, s)
                for j, s in enumerate(up["encA_cols"])}
        encB = [ts.ckks_vector_from(public_ctx, s) for s in up["encB_rows"]]
        loaded.append((encA, encB))
    return loaded


def _vec_nbytes(ser_vec):
    """统计一个序列化向量表示的下发字节。

    当前活跃的下发形态有三种：
      "packed"        —— 单条打包 ct（half 列/行 matmul、full 列/行 pack_vectors 合批）；
      "packed_group"  —— 一组多列打包 ct（half+packed 列阶段块对角 matmul 输出）；
      "scalars"       —— dim 条标量 ct 组成的列表（当前无写入方，保留兼容）。
    """
    kind = ser_vec[0]
    if kind == "packed":
        return len(ser_vec[1])
    if kind == "packed_group":
        return len(ser_vec[2])
    # scalars
    return sum(len(s) for s in ser_vec[1])


# ── 客户端：解密 ────────────────────────────────────────────────────────────

def decrypt_cols_inner(col_bytes, secret_ctx, dim, n_clients):
    """解密列聚合结果 → {列索引: 长 dim 向量}，除 N 求平均。

    支持三种列形态：("packed", ct) 单列；("packed_group", indices, ct) 一组多列；
    ("scalars", [ct]) 逐标量。
    """
    out = {}
    for key, ser in col_bytes.items():
        kind = ser[0]
        if kind == "packed":
            v = np.array(ts.ckks_vector_from(secret_ctx, ser[1]).decrypt())[:dim]
            out[int(key)] = v / n_clients
        elif kind == "packed_group":
            _, indices, ct = ser
            flat = np.array(ts.ckks_vector_from(secret_ctx, ct).decrypt())
            for s, j in enumerate(indices):
                out[int(j)] = flat[s * dim:(s + 1) * dim] / n_clients
        else:  # scalars
            v = np.array([np.array(ts.ckks_vector_from(secret_ctx, s).decrypt())[0]
                          for s in ser[1]])
            out[int(key)] = v / n_clients
    return out


def decrypt_rows_inner(row_bytes, secret_ctx, dim, n_clients):
    """解密行聚合结果 → {行索引: 长 dim 向量}，除 N 求平均。

    当前行阶段两种下发形态（跟 aggregate_inner 三条分支对应）：
      ("packed", ct)      —— 单条长 dim 打包 ct（half 走 encA_rows 累加、full 走
                             pack_vectors 合批）；
      ("scalars", [ct])   —— 逐标量 ct（保留兼容，当前无写入方）。
    """
    out = {}
    for k, ser in row_bytes.items():
        kind = ser[0]
        if kind == "packed":
            v = np.array(ts.ckks_vector_from(secret_ctx, ser[1]).decrypt())[:dim]
            out[int(k)] = v / n_clients
        else:  # scalars
            v = np.array([np.array(ts.ckks_vector_from(secret_ctx, s).decrypt())[0]
                          for s in ser[1]])
            out[int(k)] = v / n_clients
    return out
