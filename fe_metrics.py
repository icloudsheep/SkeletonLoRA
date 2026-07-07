"""指标收集与中文 CSV 导出工具。

由 main 在各步骤前后调用，累积每个阶段的耗时（秒）与网络传输量（字节），
最终一次性导出为中文表头 CSV。工具本身不计时、不判断阶段划分，只被动记录
main 传入的键值。
"""

import os
import csv


class MetricsCollector:
    """按阶段名累积耗时与传输字节的容器。

    传输量单独统计的原因：耗时反映计算开销，传输字节反映通信开销，二者是骨架
    与非骨架方案对比的两个独立维度，需分列记录。
    """

    def __init__(self):
        # 每个元素为一行阶段记录，保持插入顺序以便 CSV 按执行时序排列。
        self._rows = []

    def add(self, stage, seconds=0.0, bytes_transferred=0, note=""):
        """追加一条阶段记录。

        :param stage: 阶段名称（中文），作为 CSV 首列
        :param seconds: 该阶段耗时，单位秒
        :param bytes_transferred: 该阶段需网络传输的数据量，单位字节；无传输填 0
        :param note: 备注，记录该阶段的规模/参数等上下文
        """
        self._rows.append(dict(
            阶段=stage,
            耗时秒=round(seconds, 6),
            传输字节=int(bytes_transferred),
            传输MB=round(bytes_transferred / 1024 / 1024, 4),
            备注=note,
        ))

    def rows(self):
        """返回已记录的全部阶段行（浅拷贝，供 main 汇总或打印）。"""
        return list(self._rows)

    def to_csv(self, res_dir, filename):
        """导出为中文表头 CSV。

        使用 utf-8-sig 编码写 BOM，保证 Excel 打开中文表头不乱码。

        :param res_dir: 输出目录，不存在则创建
        :param filename: 文件名
        :return: 写出的完整路径
        """
        os.makedirs(res_dir, exist_ok=True)
        path = os.path.join(res_dir, filename)
        fields = ["阶段", "耗时秒", "传输字节", "传输MB", "备注"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self._rows)
        return path


def serialized_size(ct_list):
    """统计一组密文序列化后的总字节数。

    传入的每个元素应为已序列化的 bytes；None 占位（非骨架位置）按 0 计。
    用于度量客户端上传或服务端下发的实际通信量。

    :param ct_list: 序列化密文（bytes）或 None 的列表
    :return: 总字节数
    """
    return sum(len(s) for s in ct_list if s is not None)


class SweepTable:
    """基线 sweep 的配置对比表：每组配置一行，汇总时间/网络/误差各项指标。

    与 MetricsCollector 的分阶段明细不同，本表面向「横向对比 8 组配置」的审查视角，
    每行是一个完整配置的端到端结果，便于直接作图与阅读。
    """

    # 传输量拆成三类的原因：单次上传（一个客户端→服务端）、单次下载（服务端→一个
    # 客户端）反映单点通信压力；总开销 = N×上传 + N×下载，反映整轮全网通信量。
    # 首列「方法」区分外积法 / 内积法，是本表横向对比的最外层维度。
    FIELDS = ["方法", "打包方式", "加密程度", "骨架优化",
              "客户端加密秒", "客户端解密秒", "服务端聚合秒",
              "单次上传MB", "单次下载MB", "总网络MB",
              "误差", "可行性", "备注"]

    def __init__(self):
        self._rows = []

    def add(self, method, packing, enc, skeleton, *, n_clients,
            t_client_enc, t_client_dec, t_server,
            up_bytes_one, down_bytes_one, error, feasible=True, note=""):
        """追加一行配置结果。

        :param method: 计算方法 "外积"/"内积"
        :param packing: 打包方式 "packed"/"unpacked"
        :param enc: 加密程度 "full"/"half"
        :param skeleton: 是否启用骨架优化
        :param n_clients: 客户端数 N，用于换算总网络开销
        :param t_client_enc: 单客户端加密耗时（秒）
        :param t_client_dec: 单客户端解密耗时（秒）
        :param t_server: 服务端聚合耗时（秒）
        :param up_bytes_one: 单次上传字节（一个客户端→服务端）
        :param down_bytes_one: 单次下载字节（服务端→一个客户端）
        :param error: 重建相对误差；不可行时可为 None
        :param feasible: 该配置是否数值可行；False 时相关数值可为占位
        :param note: 备注
        """
        total_bytes = n_clients * (up_bytes_one + down_bytes_one)
        self._rows.append({
            "方法": method,
            "打包方式": packing,
            "加密程度": enc,
            "骨架优化": "开" if skeleton else "关",
            "客户端加密秒": round(t_client_enc, 6),
            "客户端解密秒": round(t_client_dec, 6),
            "服务端聚合秒": round(t_server, 6),
            "单次上传MB": round(up_bytes_one / 1024 / 1024, 4),
            "单次下载MB": round(down_bytes_one / 1024 / 1024, 4),
            "总网络MB": round(total_bytes / 1024 / 1024, 4),
            "误差": (f"{error:.3e}" if error is not None else "N/A"),
            "可行性": "可行" if feasible else "不可行",
            "备注": note,
        })

    def rows(self):
        """返回全部配置行（浅拷贝）。"""
        return list(self._rows)

    def to_csv(self, res_dir, filename):
        """导出配置对比表为中文表头 CSV（utf-8-sig）。"""
        os.makedirs(res_dir, exist_ok=True)
        path = os.path.join(res_dir, filename)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self._rows)
        return path


class RedundancyTable:
    """外积 vs 内积「空间冗余」量化表。

    按 (打包方式, 加密程度, 骨架优化) 把外积法与内积法配对，量化外积法用空间冗余
    （因子 repeat/tile 到 dim 长）换掉跨槽位求和所付出的代价：
      上传冗余比 = 外积单次上传 / 内积单次上传   （>1 表示外积多传的倍数）
      服务端时间比 = 外积服务端秒 / 内积服务端秒  （<1 表示外积换来的提速倍数）
    仅当两法在该配置都可行时才配对，否则跳过并不计比值。
    """

    FIELDS = ["打包方式", "加密程度", "骨架优化",
              "外积上传MB", "内积上传MB", "上传冗余比",
              "外积服务端秒", "内积服务端秒", "服务端时间比", "备注"]

    def __init__(self):
        self._rows = []

    def build(self, sweep_rows):
        """从 SweepTable 的行构建配对冗余表。

        :param sweep_rows: SweepTable.rows() 的返回
        :return: self（便于链式）
        """
        def key(r):
            return (r["打包方式"], r["加密程度"], r["骨架优化"])

        outer = {key(r): r for r in sweep_rows if r["方法"] == "外积"}
        inner = {key(r): r for r in sweep_rows if r["方法"] == "内积"}

        for k in outer:
            if k not in inner:
                continue
            o, i = outer[k], inner[k]
            both_ok = (o["可行性"] == "可行" and i["可行性"] == "可行")
            up_ratio = (o["单次上传MB"] / i["单次上传MB"]
                        if both_ok and i["单次上传MB"] > 0 else None)
            t_ratio = (o["服务端聚合秒"] / i["服务端聚合秒"]
                       if both_ok and i["服务端聚合秒"] > 0 else None)
            self._rows.append({
                "打包方式": k[0], "加密程度": k[1], "骨架优化": k[2],
                "外积上传MB": o["单次上传MB"], "内积上传MB": i["单次上传MB"],
                "上传冗余比": (round(up_ratio, 2) if up_ratio is not None else "N/A"),
                "外积服务端秒": o["服务端聚合秒"], "内积服务端秒": i["服务端聚合秒"],
                "服务端时间比": (round(t_ratio, 3) if t_ratio is not None else "N/A"),
                "备注": "" if both_ok else "存在不可行格，未计比值",
            })
        return self

    def rows(self):
        return list(self._rows)

    def to_csv(self, res_dir, filename):
        """导出冗余对比表为中文表头 CSV（utf-8-sig）。"""
        os.makedirs(res_dir, exist_ok=True)
        path = os.path.join(res_dir, filename)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self._rows)
        return path
