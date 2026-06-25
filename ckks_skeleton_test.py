#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CKKS Skeleton Decryption — 模块化流水线测试
=============================================
入口文件。所有逻辑拆分到以下模块：
  ckks_config    — 常量
  ckks_utils     — 加载/CKKS 加解密/误差
  ckks_indices   — 索引选择（骨架 + 部分加密）
  ckks_pipelines — 流水线 A/B/部分加密
  ckks_compare   — 解密对比
  ckks_plot      — 可视化 + CSV
  ckks_demo      — Demo 实验
  ckks_main      — main / main_partial

用法:
  python ckks_skeleton_test.py            # 3200×3200 主实验
  python ckks_skeleton_test.py --demo     # 2 客户端 10×4 演示
  python ckks_skeleton_test.py --demo-compare  # 浮点 vs 整数
  python ckks_skeleton_test.py --partial  # 部分加密扫描

环境: conda skeleton_lora_fe (py3.10 + tenseal + safetensors + numpy + matplotlib)
"""

import argparse
from ckks_demo import demo_mode, demo_float_vs_int
from ckks_main import main, main_partial

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CKKS Skeleton Decryption Test")
    p.add_argument("--demo", action="store_true",
                   help="2-client 10x4 illustrative demo")
    p.add_argument("--demo-compare", action="store_true",
                   help="Float vs integer comparison")
    p.add_argument("--partial", action="store_true",
                   help="Partial encryption scan (1%..100%)")
    args = p.parse_args()
    if args.demo_compare:
        demo_float_vs_int()
    elif args.demo:
        demo_mode()
    elif args.partial:
        main_partial()
    else:
        main()
