"""真实 LoRA 权重发现、配对与矩形形状校验工具。"""

from dataclasses import dataclass
import os
import re
from typing import Iterable

import numpy as np
from safetensors import safe_open


_AB_KEY_RE = re.compile(
    r"^(?P<prefix>.+)\.(?P<kind>lora_(?:embedding_)?[AB])"
    r"(?:\.(?P<adapter>[^.]+))?\.weight$"
)


@dataclass(frozen=True)
class ABIdentifier:
    """一个 LoRA A/B 对的稳定标识。"""

    prefix: str
    adapter: str
    family: str

    @property
    def text(self):
        return ".".join((self.prefix, self.family, self.adapter))


@dataclass
class ABPair:
    """一个客户端上的 LoRA A/B 对及其来源 key。"""

    identifier: ABIdentifier
    a: np.ndarray | None
    b: np.ndarray | None
    a_key: str | None
    b_key: str | None
    warning: str | None = None

    @property
    def shape(self):
        if self.a is None or self.b is None:
            return None
        return (int(self.b.shape[0]), int(self.a.shape[1]))


@dataclass
class ClientABCollection:
    """一个客户端的 AB 对集合。"""

    client_id: int
    path: str
    pairs: dict[ABIdentifier, ABPair]
    warnings: list[str]


def _parse_key(key):
    match = _AB_KEY_RE.match(key)
    if match is None:
        return None
    family = match.group("kind")
    adapter = match.group("adapter") or "default"
    prefix = match.group("prefix")
    return ABIdentifier(prefix=prefix, adapter=adapter, family=family)


def _pair_identifier(identifier):
    return ABIdentifier(
        prefix=identifier.prefix,
        adapter=identifier.adapter,
        family="lora_embedding" if "embedding" in identifier.family else "lora",
    )


def _read_candidates(path):
    candidates = {}
    with safe_open(path, framework="np") as handle:
        for key in handle.keys():
            parsed = _parse_key(key)
            if parsed is None:
                continue
            pair_id = _pair_identifier(parsed)
            side = "A" if parsed.family.endswith("A") else "B"
            candidates.setdefault(pair_id, {}).setdefault(side, []).append(key)
            candidates[pair_id][side].sort()
    return candidates


def _load_tensor(path, key):
    with safe_open(path, framework="np") as handle:
        return np.asarray(handle.get_tensor(key), dtype=np.float64)


def discover_client_pairs(path, client_id, rank):
    """发现一个客户端文件中的全部标准二维 LoRA AB 对。"""
    warnings = []
    pairs = {}
    candidates = _read_candidates(path)
    for identifier in sorted(candidates, key=lambda item: item.text):
        sides = candidates[identifier]
        a_keys = sides.get("A", [])
        b_keys = sides.get("B", [])
        if len(a_keys) != 1 or len(b_keys) != 1:
            warnings.append(
                f"{identifier.text}: A 候选={len(a_keys)}，B 候选={len(b_keys)}，"
                "按确定性排序选择首个候选或使用缺失零补"
            )
        a_key = a_keys[0] if a_keys else None
        b_key = b_keys[0] if b_keys else None
        a = _load_tensor(path, a_key) if a_key else None
        b = _load_tensor(path, b_key) if b_key else None
        warning = None
        if a is None or b is None:
            warning = "A/B 不完整，使用零矩阵补充"
        elif a.ndim != 2 or b.ndim != 2:
            warning = f"A/B 必须是二维，实际 A.ndim={a.ndim}，B.ndim={b.ndim}"
        elif a.shape[0] != rank or b.shape[1] != rank:
            warning = (
                f"rank 不匹配：A.shape={a.shape}，B.shape={b.shape}，期望 rank={rank}"
            )
        elif a.shape[1] != b.shape[0]:
            warning = f"A/B 中间维度不一致：A.shape={a.shape}，B.shape={b.shape}"
        if warning:
            warnings.append(f"{identifier.text}: {warning}")
        pairs[identifier] = ABPair(identifier, a, b, a_key, b_key, warning)
    return ClientABCollection(client_id, path, pairs, warnings)


def discover_all_clients(paths: Iterable[str], rank):
    """发现所有客户端的 AB 对，并返回全局标识并集。"""
    collections = [
        discover_client_pairs(path, client_id, rank)
        for client_id, path in enumerate(paths)
    ]
    identifiers = sorted(
        {identifier for collection in collections for identifier in collection.pairs},
        key=lambda item: item.text,
    )
    return collections, identifiers


def _zero_pair(identifier, shape, rank, warning):
    out_features, in_features = shape
    return ABPair(
        identifier=identifier,
        a=np.zeros((rank, in_features), dtype=np.float64),
        b=np.zeros((out_features, rank), dtype=np.float64),
        a_key=None,
        b_key=None,
        warning=warning,
    )


def materialize_pair(collection, identifier, shape, rank):
    """返回指定客户端的合法 AB 对；缺失或非法对象使用零矩阵。"""
    pair = collection.pairs.get(identifier)
    if pair is None or pair.warning:
        warning = pair.warning if pair is not None else "客户端缺少该 AB 对"
        return _zero_pair(identifier, shape, rank, warning)
    if pair.shape != shape:
        return _zero_pair(
            identifier,
            shape,
            rank,
            f"矩形形状不一致：实际={pair.shape}，全局={shape}",
        )
    return pair


def infer_shapes(collections, identifiers, rank):
    """为每个全局 AB 标识推断一致的矩形形状。"""
    shapes = {}
    warnings = []
    for identifier in identifiers:
        candidates = [
            collection.pairs[identifier].shape
            for collection in collections
            if identifier in collection.pairs
            and collection.pairs[identifier].warning is None
        ]
        if not candidates:
            warnings.append(f"{identifier.text}: 所有客户端均无合法 AB 对")
            continue
        unique_shapes = sorted(set(candidates))
        if len(unique_shapes) != 1:
            warnings.append(f"{identifier.text}: 客户端矩形形状不一致={unique_shapes}")
            continue
        shapes[identifier] = unique_shapes[0]
    return shapes, warnings


def client_paths(root, n_clients):
    """生成连续客户端的 safetensors 路径。"""
    return [
        os.path.join(
            root,
            f"client_{client_id}_output",
            "final_lora",
            "adapter_model.safetensors",
        )
        for client_id in range(n_clients)
    ]
