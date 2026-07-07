"""基线对比作图工具。

按打包方式分别出一张对比图：每张图在「全加密/半加密」两组下，对比「骨架优化开/关」
的时间开销（客户端加密、客户端解密、服务端聚合）与网络开销（单次上传、单次下载、
总开销）。同时导出 PNG 与 PDF。

工具只消费 main 传入的结构化结果，不自行运行实验、不读全局配置。
"""

import os
import matplotlib
matplotlib.use("Agg")   # 无界面后端，服务器/CI 环境可用
import matplotlib.pyplot as plt

# 中文字体：优先常见中文字体，缺失时 matplotlib 回退（可能显示方框，不影响数值）。
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti",
    "Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _cfg_label(enc, skeleton):
    """把加密程度 + 骨架开关拼成可读的分组标签。"""
    return f"{'全加密' if enc == 'full' else '半加密'}·骨架{'开' if skeleton else '关'}"


def plot_baseline(rows, packing, res_dir, filename):
    """为某一打包方式画时间与网络两联图并存 PNG/PDF。

    :param rows: 该打包方式下的配置行列表（fe_metrics.SweepTable.rows 的子集），
                 每行含中文键：加密程度/骨架优化/客户端加密秒/客户端解密秒/服务端聚合秒/
                 单次上传MB/单次下载MB/总网络MB
    :param packing: "packed"/"unpacked"，仅用于标题
    :param res_dir: 输出目录
    :param filename: 文件名（不含扩展名）
    :return: [png_path, pdf_path]
    """
    os.makedirs(res_dir, exist_ok=True)

    # 固定分组顺序：全加密开/关、半加密开/关，保证四种配置图例稳定。
    order = [("full", True), ("full", False), ("half", True), ("half", False)]

    def _find(enc, skel):
        tag = "开" if skel else "关"
        enc_cn = "full"
        for r in rows:
            if r["加密程度"] == enc and r["骨架优化"] == tag:
                return r
        return None

    labels, t_enc, t_dec, t_srv, up, down, total = [], [], [], [], [], [], []
    for enc, skel in order:
        r = _find(enc, skel)
        if r is None:
            continue
        labels.append(_cfg_label(enc, skel))
        t_enc.append(r["客户端加密秒"])
        t_dec.append(r["客户端解密秒"])
        t_srv.append(r["服务端聚合秒"])
        up.append(r["单次上传MB"])
        down.append(r["单次下载MB"])
        total.append(r["总网络MB"])

    import numpy as np
    x = np.arange(len(labels))
    w = 0.25
    pack_cn = "打包" if packing == "packed" else "不打包"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"基线对比（{pack_cn}）", fontsize=15, fontweight="bold")

    # 左：时间开销
    ax1.bar(x - w, t_enc, w, label="客户端加密", color="#3498db")
    ax1.bar(x, t_dec, w, label="客户端解密", color="#9b59b6")
    ax1.bar(x + w, t_srv, w, label="服务端聚合", color="#e74c3c")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("耗时（秒）"); ax1.set_title("(a) 时间开销")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    # 右：网络开销
    ax2.bar(x - w, up, w, label="单次上传", color="#2ecc71")
    ax2.bar(x, down, w, label="单次下载", color="#f39c12")
    ax2.bar(x + w, total, w, label="总网络开销", color="#34495e")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("数据量（MB）"); ax2.set_title("(b) 网络开销")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    png = os.path.join(res_dir, filename + ".png")
    pdf = os.path.join(res_dir, filename + ".pdf")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_redundancy(red_rows, res_dir, filename):
    """画内外积「空间冗余」量化图并存 PNG/PDF。

    左：上传冗余比（外积上传/内积上传），量化外积用多少倍上传量换掉跨槽位求和；
    右：外积上传 vs 内积上传的绝对值分组柱，直观看两法的通信量级差。
    上传冗余比可能很大，故左图用对数纵轴；不可行配对（比值 N/A）跳过。

    :param red_rows: fe_metrics.RedundancyTable.rows() 的返回
    :param res_dir: 输出目录
    :param filename: 文件名（不含扩展名）
    :return: [png_path, pdf_path]
    """
    import numpy as np
    os.makedirs(res_dir, exist_ok=True)

    # 只画比值可计算的配对。
    rows = [r for r in red_rows if r["上传冗余比"] != "N/A"]
    labels, ratio, outer_up, inner_up = [], [], [], []
    for r in rows:
        pk = "打包" if r["打包方式"] == "packed" else "不打包"
        labels.append(f"{pk}·{'全' if r['加密程度']=='full' else '半'}·骨{r['骨架优化']}")
        ratio.append(r["上传冗余比"])
        outer_up.append(r["外积上传MB"])
        inner_up.append(r["内积上传MB"])

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("内外积空间冗余量化（外积用空间冗余换掉跨槽位求和）",
                 fontsize=15, fontweight="bold")

    bars = ax1.bar(x, ratio, 0.6, color="#e67e22")
    ax1.set_yscale("log")
    ax1.axhline(1.0, color="gray", ls="--", lw=1, label="冗余比=1（无冗余）")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("上传冗余比（外积/内积，对数轴）")
    ax1.set_title("(a) 上传冗余比")
    for b, v in zip(bars, ratio):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}×",
                 ha="center", va="bottom", fontsize=8)
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    w = 0.38
    ax2.bar(x - w / 2, outer_up, w, label="外积上传", color="#e74c3c")
    ax2.bar(x + w / 2, inner_up, w, label="内积上传", color="#2ecc71")
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("单次上传（MB，对数轴）")
    ax2.set_title("(b) 外积 vs 内积 单次上传量")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    png = os.path.join(res_dir, filename + ".png")
    pdf = os.path.join(res_dir, filename + ".pdf")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]
