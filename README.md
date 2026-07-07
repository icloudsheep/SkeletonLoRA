# SkeletonLoRA

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

联邦 LoRA 场景下，基于 **CKKS 同态加密** 的密文域聚合实验。客户端只加密上传 LoRA
因子 A/B，服务端在密文域完成聚合并下发，客户端用私钥解密。对比两种聚合方式：

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
├── main.py          # 唯一编排入口：建密钥→加密上传→密文聚合→解密→误差分析→写CSV
├── fe_config.py     # 配置常量（规模、CKKS 参数、输出路径）
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

依赖：Python 3.10 + tenseal 0.3.16 + safetensors + numpy + matplotlib。

## 用法

```shell
python main.py          # demo 模式：随机 A/B，无需权重，小维度秒级跑通
python main.py --real   # 真实模式：从 temp_output_dir 加载 LoRA 权重
```

维度 `DIM` 在 `fe_config.py` 配置。**注意**：完整聚合需在密文域算整个 $d\times d$
矩阵，代价随 $d$ 平方增长，$d=3200$ 时上传密文可达数 GB、耗时分钟级，实测不可行；
故 demo 默认取小维度让两条路径都能跑完对比。真实大维度建议只跑骨架路径。

## 输出

`_res/聚合实验结果.csv`（中文表头，utf-8-sig 编码）记录每一步的耗时（秒）、
网络传输字节 / MB 与备注，末尾为完整聚合及各 $r$ 值骨架聚合相对 $\Delta W_{\text{mean}}$
的误差。

## License

Apache License 2.0 — 见 [LICENSE](LICENSE)。
