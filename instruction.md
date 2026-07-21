# 项目实验使用说明书

## 启动 TensorBoard

```shell
tensorboard --logdir _res/runs --port 6006
```

## 真实数据实验范围

- 数据来源固定为 `--real true`，即读取 `fe_config.py` 中的 `TEMP_OUTPUT_DIR`。
- 当前完整矩阵包含：明文参考 2 组；外积 18 组；内积 18 组；合计 38 组。
- `--skeleton false` 表示完整矩阵重建；`--skeleton true` 表示 skeleton/CUR 重建。
- full + `--skeleton false` 在真实 LoRA 维度下内存和耗时可能非常高，运行前应确认服务器资源。
- 以下命令均串行执行；程序本身没有任务级并行。

## 一键串行运行全套 real 实验

```shell
set -euo pipefail

# 1. 明文参考：不经过 CKKS，用于绝对误差基准与 CUR 误差基准。
for skeleton in false true; do
  python modern_main.py --real true --method 明文参考 --mode plain_baseline --skeleton "${skeleton}"
done

# 2. CKKS 全加密：外积/内积 × 全矩阵/skeleton。
for method in 外积 内积; do
  for skeleton in false true; do
    python modern_main.py --real true --method "${method}" --mode full --skeleton "${skeleton}"
  done
done

# 3. CKKS 部分加密：外积/内积 × partial_A/partial_AB × 比例 × 全矩阵/skeleton。
for method in 外积 内积; do
  for mode in partial_A partial_AB; do
    for ratio in 1 5 25 50; do
      for skeleton in false true; do
        python modern_main.py --real true --method "${method}" --mode "${mode}" --ratio "${ratio}" --skeleton "${skeleton}"
      done
    done
  done
done
```

## 后台运行全套 real 实验

如果服务器会断开 SSH，建议先把上一节脚本保存为 `run_all_real.sh`，再执行：

```shell
chmod +x run_all_real.sh
nohup ./run_all_real.sh > real_experiments.log 2>&1 &
tail -f real_experiments.log
```

## 单项 real 实验命令

### 明文参考-无比例-不带骨架

```shell
python modern_main.py --real true --method 明文参考 --mode plain_baseline --skeleton false
```

### 明文参考-无比例-带骨架

```shell
python modern_main.py --real true --method 明文参考 --mode plain_baseline --skeleton true
```

### 外积-全加密-无比例-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode full --skeleton false
```

### 外积-全加密-无比例-带骨架

```shell
python modern_main.py --real true --method 外积 --mode full --skeleton true
```

### 外积-部分加密A-1%-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_A --ratio 1 --skeleton false
```

### 外积-部分加密A-1%-带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_A --ratio 1 --skeleton true
```

### 外积-部分加密A-5%-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_A --ratio 5 --skeleton false
```

### 外积-部分加密A-5%-带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_A --ratio 5 --skeleton true
```

### 外积-部分加密A-25%-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_A --ratio 25 --skeleton false
```

### 外积-部分加密A-25%-带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_A --ratio 25 --skeleton true
```

### 外积-部分加密A-50%-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_A --ratio 50 --skeleton false
```

### 外积-部分加密A-50%-带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_A --ratio 50 --skeleton true
```

### 外积-部分加密AB-1%-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_AB --ratio 1 --skeleton false
```

### 外积-部分加密AB-1%-带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_AB --ratio 1 --skeleton true
```

### 外积-部分加密AB-5%-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_AB --ratio 5 --skeleton false
```

### 外积-部分加密AB-5%-带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_AB --ratio 5 --skeleton true
```

### 外积-部分加密AB-25%-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_AB --ratio 25 --skeleton false
```

### 外积-部分加密AB-25%-带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_AB --ratio 25 --skeleton true
```

### 外积-部分加密AB-50%-不带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_AB --ratio 50 --skeleton false
```

### 外积-部分加密AB-50%-带骨架

```shell
python modern_main.py --real true --method 外积 --mode partial_AB --ratio 50 --skeleton true
```

### 内积-全加密-无比例-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode full --skeleton false
```

### 内积-全加密-无比例-带骨架

```shell
python modern_main.py --real true --method 内积 --mode full --skeleton true
```

### 内积-部分加密A-1%-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_A --ratio 1 --skeleton false
```

### 内积-部分加密A-1%-带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_A --ratio 1 --skeleton true
```

### 内积-部分加密A-5%-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_A --ratio 5 --skeleton false
```

### 内积-部分加密A-5%-带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_A --ratio 5 --skeleton true
```

### 内积-部分加密A-25%-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_A --ratio 25 --skeleton false
```

### 内积-部分加密A-25%-带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_A --ratio 25 --skeleton true
```

### 内积-部分加密A-50%-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_A --ratio 50 --skeleton false
```

### 内积-部分加密A-50%-带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_A --ratio 50 --skeleton true
```

### 内积-部分加密AB-1%-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_AB --ratio 1 --skeleton false
```

### 内积-部分加密AB-1%-带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_AB --ratio 1 --skeleton true
```

### 内积-部分加密AB-5%-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_AB --ratio 5 --skeleton false
```

### 内积-部分加密AB-5%-带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_AB --ratio 5 --skeleton true
```

### 内积-部分加密AB-25%-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_AB --ratio 25 --skeleton false
```

### 内积-部分加密AB-25%-带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_AB --ratio 25 --skeleton true
```

### 内积-部分加密AB-50%-不带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_AB --ratio 50 --skeleton false
```

### 内积-部分加密AB-50%-带骨架

```shell
python modern_main.py --real true --method 内积 --mode partial_AB --ratio 50 --skeleton true
```

## 输出查看

- 每条命令会生成一个独立的 `_res/runs/<时间戳>-real-d<最大维度>-<case>/` 目录。
- 主题 CSV 在每个 run 目录根部：`ab_error_metrics.csv`、`ab_timing_metrics.csv`、`ab_communication_metrics.csv`、`skeleton_cur_metrics.csv`、`context_metrics.csv`。
- 摘要文件在每个 run 的 `artifacts/` 目录。
- TensorBoard 读取 `_res/runs` 后，可按中文 tag 查看误差、耗时、通信量和 AB 画像。
