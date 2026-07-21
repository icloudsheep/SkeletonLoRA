"""全局配置常量。

集中管理实验规模、CKKS 参数与输出路径，供 main 读取后传递给各工具函数。
工具函数自身不读本模块，所有参数均由 main 显式传入。
"""

# ── 数据来源 ────────────────────────────────────────────────────────────
# 真实 LoRA 适配器权重目录；每个客户端一个子目录，内含 adapter_model.safetensors。
TEMP_OUTPUT_DIR = "./temp_output_dir"

# ── 联邦规模 ────────────────────────────────────────────────────────────
N_CLIENTS = 4          # 参与聚合的客户端数量 N
RANK = 4               # 每个客户端 LoRA 因子的秩 R（B 为 dim×R，A 为 R×dim）
LORA_ALPHA = 8         # 全局 LoRA 缩放参数；SCALING = LORA_ALPHA / RANK
SCALING = LORA_ALPHA / RANK

# 实验维度 d（ΔW 为 d×d）。
# 取舍：无骨架基线需在密文域算完整 d×d 矩阵，代价约 O(d^2·R·N) 次密文乘，
# d=3200 时上传密文可达数 GB、耗时分钟级，实测不可行；故默认取小维度让两条
# 路径都能跑完并直接对比。真实 3200 维只跑骨架路径时另行指定。
DIM = 64

# ── 骨架规模扫描 ──────────────────────────────────────────────────────────
# 骨架路径逐个 r 值重建并统计误差；r 需 ≤ min(dim, N·RANK)。
R_VALUES = [2, 4, 6, 8, 10, 12, 14, 16]

# ── CKKS 参数 ──────────────────────────────────────────────────────────
# poly_modulus_degree 决定单条密文可打包的槽位数（= degree/2）；向量长度 dim 需 ≤ 槽位数。
POLY_MODULUS_DEGREE = 8192
# 系数模数链：首尾为特殊素数，中间两个 40bit 提供 2 层乘法深度，
# 恰好容纳「密文×密文外积」+「密文×明文求平均」两次带 rescale 的乘法。
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE = 2 ** 40

# ── 基线对比网格（sweep 三轴） ─────────────────────────────────────────────
# 三个正交维度，笛卡尔积共 8 组配置，全部由 main 统一编排：
#   打包方式 packing : "packed"（多列共享一条密文槽位，省网络）
#                      "unpacked"（每行/列一条独立密文，聚合直接相加，省服务端时间）
#   加密程度 enc     : "full"（A、B 均加密 → 服务端密文×密文）
#                      "half"（仅加密 A、B 走明文 → 服务端密文×明文，更省）
#   骨架优化 skeleton: True（仅重建 uniform 骨架行列 + CUR）/ False（重建完整 d×d）
PACKING_MODES = ["packed", "unpacked"]
ENC_LEVELS = ["full", "half"]
SKELETON_MODES = [True, False]

# 计算方法轴：外积法（fe_client/fe_server，逐元素乘加，无 galois）与内积法（fe_inner，
# 标准矩阵乘法，靠 matmul/dot 跨槽位求和，需 galois）。二者笛卡尔积其余三轴 = 16 组。
METHODS = ["外积", "内积"]

# 骨架路径统一取的骨架规模（并集口径）。取真实秩量级 N·RANK 保证 CUR 可无损重建。
SKELETON_R = 16
CUR_CONDITION_NUMBER_THRESHOLD = 1e12
RELATIVE_ERROR_WARNING = 1e-3
RELATIVE_ERROR_FAILURE = 1e-1

# ── 新实验模式 ───────────────────────────────────────────────────────────
# plain_baseline 绕过 CKKS；full 加密全部 A/B；partial_A 只加密 A 的前 p% 列；
# partial_AB 同时加密 A 的前 p% 列和 B 的前 p% 行。
EXPERIMENT_MODES = ["plain_baseline", "partial_A", "partial_AB", "full"]
PARTIAL_RATIOS = [1, 5, 25, 50]
MODE_FULL_RATIO = 100

# 结果口径：packed/unpacked 不再作为对比轴，内部实现使用 packed 策略。
PACKING = "packed"

# 运行产物与持久化策略。
RUNS_DIR = "_res/runs"
SAVE_COMMUNICATION_PAYLOADS = False
SAVE_MATRIX_ARTIFACTS = False
SAVE_PRIVATE_CONTEXT = False

# 实验重复：demo 预热 1 次、正式 3 次；真实权重只执行 1 次。
DEMO_WARMUP_RUNS = 1
DEMO_REPEATS = 3
REAL_REPEATS = 1

# 内积法-全加密单格的服务端时间预算（秒）。full 内积每个输出元一次 ct×ct dot，
# 完整重建需 O(d²·N) 次，可能极慢；超过预算则如实标注「不可行」，绝不编造数字。
INNER_FULL_TIME_BUDGET = 180

# ── 输出 ────────────────────────────────────────────────────────────────
RES_DIR = "_res"
CSV_PATH = "聚合实验结果.csv"          # 单配置运行的分阶段明细（中文表头）
SWEEP_CSV_PATH = "基线对比结果.csv"     # sweep 的 16 组配置对比表（中文表头）
REDUNDANCY_CSV_PATH = "内外积冗余对比.csv"  # 外积/内积配对的空间冗余量化表
PLOT_PACKED = "基线1_打包_对比"         # 打包基线图文件名（自动加 .png/.pdf）
PLOT_UNPACKED = "基线2_不打包_对比"     # 不打包基线图文件名
PLOT_REDUNDANCY = "内外积冗余对比"      # 内外积冗余图文件名
