# SkeletonLoRA

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

联邦 LoRA 场景下，基于 **CKKS 同态加密** 的密文域聚合实验。客户端加密上传 LoRA
因子 A/B，服务端在不解密的情况下完成聚合并下发，客户端用私钥解密。当前协议同时支持
方阵和矩形 LoRA 投影，并以每个 `(层名, 投影名, AB 标识)` 作为独立实验单位。

正式实验模式包括：

- **plain baseline**：完全绕过 CKKS 的精确 NumPy 参考；
- **full**：A/B 全部加密；
- **partial_A**：只加密 A 的前 `p%` 列；
- **partial_AB**：同时加密 A 的前 `p%` 列和 B 的前 `p%` 行。

`partial` 的默认比例为 `1%、5%、25%、50%`。每种模式都可选择完整下发或 skeleton
下发。混合模式下，服务端分别处理明文块和密文块，客户端按坐标拼接后再计算误差或 CUR。

完整聚合和骨架聚合分别表示：

- **完整聚合**：服务端重建整个 $d\times d$ 聚合矩阵后下发。
- **骨架聚合**：服务端只重建 uniform 骨架覆盖的 $r$ 行 + $r$ 列，客户端解密后用
  CUR 分解 $\Delta W_{\text{rec}} = C_r M_r^{-1} R_r$ 重建完整矩阵。

参考真值为平均聚合 $\Delta W_{\text{mean}} = \frac{1}{N}\sum_i B_i A_i$，服务端密文域
的 $\times\frac{1}{N}$ 与之对齐。

## 密钥边界

| 角色 | 持有 | 能力 |
|------|------|------|
| 客户端 | 公开 context（加密） + 私钥 context（解密） | 加密上传、解密下发 |
| 服务端 | 公开 context | 密文加/乘/平均，**无法解密** |

服务端全程只接触密文，这是同态加密隐私保证的基础。

## 聚合语义（正确性关键）

联邦聚合必须先对每个客户端各自算 $B_i A_i$、再跨客户端求和，
**不是** $(\sum_i B_i)(\sum_i A_i)$——后者会引入客户端间交叉项 $B_i A_j\ (i\neq j)$。
服务端对每个客户端独立在密文域重建外积后累加，满足该语义。

## 项目结构

```
SkeletonLoRA/
├── main.py          # 兼容旧实验入口
├── modern_main.py   # 新协议入口：多层、多 AB、矩形、full/partial、TensorBoard
├── fe_config.py     # 配置常量（规模、CKKS 参数、输出路径）
├── fe_data.py       # 全部 LoRA AB 发现、配对、校验与零补
├── fe_modes.py      # full/partial 分区与误差摘要
├── fe_outer_hybrid.py # 矩形外积 CKKS 明文/密文混合块协议
├── fe_runner.py     # 单 AB 对的 baseline/外积/skeleton 运行器
├── fe_logging.py    # run 目录、配置快照、任务状态与 TensorBoard
├── fe_context.py    # CKKS 私钥/公开 context 创建与派生
├── fe_client.py     # 客户端：加密上传 A/B 因子、私钥解密
├── fe_server.py     # 服务端：密文域骨架聚合 / 完整聚合
├── fe_skeleton.py   # uniform 骨架索引 + CUR 重建
├── fe_metrics.py    # 耗时/传输量收集，导出中文 CSV
├── _res/            # 实验结果（聚合实验结果.csv）
├── temp_output_dir/ # 真实 LoRA 权重（gitignored，--real 模式需要）
└── __SHE-LoRA-main/, _scripts/   # 参考实现，不参与主流程
```

## 环境

```shell
conda env create -f environment.yaml
conda activate skeleton_lora_fe   # 注意环境名为 skeleton_lora_fe
```

依赖：Python 3.10 + tenseal 0.3.16 + safetensors + numpy + matplotlib + tensorboard + psutil。

## 用法

```shell
python main.py          # demo 模式：随机 A/B，无需权重，小维度秒级跑通
python main.py --real   # 真实模式：从 temp_output_dir 加载 LoRA 权重

# 新协议：小维度单配置验证
python modern_main.py --dim 16 --method 外积 --mode partial_A --ratio 25 --skeleton

# 新协议：完整 demo 配置矩阵
python modern_main.py --dim 64

# 新协议：真实权重；高维运行前先评估内存和耗时
python modern_main.py --real --method 外积 --mode full --skeleton
```

维度 `DIM` 在 `fe_config.py` 配置。**注意**：完整聚合需在密文域算整个 $d\times d$
矩阵，代价随 $d$ 平方增长，$d=3200$ 时上传密文可达数 GB、耗时分钟级，实测不可行；
故 demo 默认取小维度让两条路径都能跑完对比。真实大维度建议只跑骨架路径。

## 输出

新协议每次运行都写入 `_res/runs/<时间戳>-<数据来源>-d<最大维度>/`，包括配置快照、
环境快照、任务状态、主题 CSV、摘要 artifacts 和 TensorBoard event。TensorBoard 按
`方法/模式/比例/skeleton` 归并结果指标 tag，同一配置下的 AB 结果以 step 展示在同一张图；
另有 `AB画像/*` 用 AB index 展示 A/B 范数与形状，并用 `AB画像/索引映射`
记录 index 到 AB 标识的映射。中文字段解释见 [`CSV字段说明.md`](CSV字段说明.md)。
旧入口仍写入原有 `_res/` 结果文件。

## License

Apache License 2.0 — 见 [LICENSE](LICENSE)。
