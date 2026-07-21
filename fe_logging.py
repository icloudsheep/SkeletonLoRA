"""实验 run 目录、配置快照、TensorBoard 和任务状态记录。"""

from dataclasses import dataclass
from datetime import datetime
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import psutil


@dataclass
class RunPaths:
    """一次实验运行的产物路径。"""

    root: Path
    config: Path
    environment: Path
    tasks: Path
    metrics: Path
    tensorboard: Path
    artifacts: Path


def _git_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_snapshot():
    """收集当前机器、解释器和关键依赖版本。"""
    versions = {}
    for name in ("numpy", "tenseal", "safetensors", "tensorboard"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "available")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "environment_variables": {
            key: value
            for key, value in os.environ.items()
            if key in {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
        },
        "versions": versions,
        "git": _git_metadata(),
    }


def create_run_paths(root, label):
    """创建带人类可读时间戳且不覆盖旧结果的 run 目录。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{timestamp}-{label}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}-{label}-{suffix}"
        suffix += 1
    for directory in (candidate, candidate / "tensorboard", candidate / "artifacts"):
        directory.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        root=candidate,
        config=candidate / "config.json",
        environment=candidate / "environment.json",
        tasks=candidate / "tasks.jsonl",
        metrics=candidate / "metrics.jsonl",
        tensorboard=candidate / "tensorboard",
        artifacts=candidate / "artifacts",
    )


def write_json(path, data):
    """以 UTF-8 JSON 写入配置或环境快照。"""
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


class RunLogger:
    """写入 JSONL 明细和可选 TensorBoard 标量/文本。"""

    def __init__(self, paths):
        self.paths = paths
        self._writer = None
        try:
            from tensorboard.compat.proto import event_pb2, summary_pb2
            from tensorboard.summary.writer.event_file_writer import EventFileWriter

            self._event_pb2 = event_pb2
            self._summary_pb2 = summary_pb2
            self._writer = EventFileWriter(str(paths.tensorboard))
        except Exception:
            self._writer = None

    def _append(self, path, record):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def task(self, task_id, status, **fields):
        """记录任务状态；恢复时只将 completed 视为可跳过。"""
        record = {"task_id": task_id, "status": status, **fields}
        self._append(self.paths.tasks, record)

    def scalar(self, tag, value, step, **fields):
        """写入一个 TensorBoard 标量和 JSONL 指标。"""
        record = {"tag": tag, "value": value, "step": step, **fields}
        self._append(self.paths.metrics, record)
        if self._writer is not None and value is not None:
            summary = self._summary_pb2.Summary(
                value=[self._summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
            )
            self._writer.add_event(
                self._event_pb2.Event(
                    wall_time=time.time(), step=int(step), summary=summary
                )
            )

    def metric(self, tag, value, step, **fields):
        """只写入 JSONL 指标，不生成 TensorBoard 图表。"""
        record = {"tag": tag, "value": value, "step": step, **fields}
        self._append(self.paths.metrics, record)

    def text(self, tag, text, step=0):
        """写入 TensorBoard 文本。"""
        if self._writer is not None:
            from tensorboard.compat.proto import tensor_pb2, types_pb2

            metadata = self._summary_pb2.SummaryMetadata(
                plugin_data=self._summary_pb2.SummaryMetadata.PluginData(
                    plugin_name="text"
                )
            )
            tensor = tensor_pb2.TensorProto(
                dtype=types_pb2.DT_STRING,
                string_val=[str(text).encode("utf-8")],
            )
            summary = self._summary_pb2.Summary(
                value=[
                    self._summary_pb2.Summary.Value(
                        tag=tag, metadata=metadata, tensor=tensor
                    )
                ]
            )
            self._writer.add_event(
                self._event_pb2.Event(
                    wall_time=time.time(), step=int(step), summary=summary
                )
            )

    def flush(self):
        if self._writer is not None:
            self._writer.flush()

    def close(self):
        if self._writer is not None:
            self._writer.close()
