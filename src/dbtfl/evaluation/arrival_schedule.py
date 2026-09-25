"""Paired asynchronous client-arrival schedules for fair baseline studies.

用于公平基线研究的配对异步客户端上线调度。

Both DwT-FL and a comparison baseline must receive the same logical arrival
vector for a matching case and repetition. The schedule intentionally has a
first client at the configured minimum and a late client at the configured
anchor (19 seconds in the paper's 0--20 second setting). This preserves an
asynchronous arrival workload without letting a random sample accidentally
remove the required late-arrival condition. DwT-FL 与对比基线在匹配的用例和
重复中必须获得同一逻辑上线向量。该调度刻意将首客户端置于配置下界，并将一个
晚到客户端置于配置锚点（论文的 0--20 秒设置中为 19 秒）；这样既保持异步上线
负载，又避免随机抽样意外抹去必需的晚到条件。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from typing import Any


PAPER_LATE_ARRIVAL_SECONDS = 19.0
"""Late-arrival anchor required by the paper's 0--20 second workload.

论文 0--20 秒工作负载要求的晚到锚点。
"""


def paired_arrival_delays(
    *,
    seed: int,
    case: Any,
    repetition: int,
    client_ids: Iterable[str],
    minimum_delay_seconds: float,
    maximum_delay_seconds: float,
    late_arrival_seconds: float = PAPER_LATE_ARRIVAL_SECONDS,
) -> dict[str, float]:
    """Return one reproducible, shared, anchored arrival schedule.

    返回一个可复现、跨方案共享且带锚点的上线调度。

    The first stable client begins at ``minimum_delay_seconds``. For two or
    more clients, the last stable client begins at the clipped late-arrival
    anchor; all other clients independently draw before that anchor. Therefore
    a 0--20-second formal run always includes a client scheduled exactly at
    19 seconds. 首个稳定客户端在 ``minimum_delay_seconds`` 开始。对于至少两个
    客户端，最后一个稳定客户端在裁剪后的晚到锚点开始；其余客户端在该锚点之前独立
    抽样。因此 0--20 秒正式运行总会包含一个恰好在第 19 秒调度的客户端。
    """
    identifiers = tuple(sorted({str(client_id) for client_id in client_ids}))
    if minimum_delay_seconds < 0.0 or maximum_delay_seconds < minimum_delay_seconds:
        raise ValueError("arrival delay bounds are invalid / 上线延迟边界无效")
    if late_arrival_seconds < 0.0:
        raise ValueError("late arrival anchor must be non-negative / 晚到锚点必须非负")
    if not identifiers:
        return {}

    anchor = min(maximum_delay_seconds, max(minimum_delay_seconds, late_arrival_seconds))
    if len(identifiers) == 1:
        return {identifiers[0]: anchor}

    result = {
        identifiers[0]: float(minimum_delay_seconds),
        identifiers[-1]: float(anchor),
    }
    # The identity excludes implementation-specific worker settings. This
    # makes matching DwT-FL and NDSS cases draw exactly the same intermediate
    # delays while retaining independent random arrival positions. 标识不包含
    # 实现特有的工作线程设置，使匹配的 DwT-FL 和 NDSS 用例抽取完全相同的中间
    # 延迟，同时保留独立的随机上线位置。
    material = json.dumps(
        {
            "seed": seed,
            "repetition": repetition,
            "suite": str(getattr(case, "suite")),
            "variable": str(getattr(case, "variable")),
            "value": getattr(case, "value"),
            "clients": int(getattr(case, "clients")),
            "records_per_client": int(getattr(case, "records_per_client")),
            "duplicate_ratio": float(getattr(case, "duplicate_ratio")),
            "backend_workers": int(getattr(case, "backend_workers")),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    generator = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
    for identifier in identifiers[1:-1]:
        result[identifier] = generator.uniform(minimum_delay_seconds, anchor)
    return result


def arrival_schedule_contract(
    delays: Mapping[str, float],
    *,
    late_arrival_seconds: float = PAPER_LATE_ARRIVAL_SECONDS,
) -> dict[str, object]:
    """Describe the schedule invariant in persisted experiment provenance.

    在持久化实验来源信息中描述调度不变量。
    """
    if not delays:
        return {
            "scheme": "paired_anchored_async_arrival_v1",
            "late_arrival_anchor_seconds": late_arrival_seconds,
            "scheduled_late_client_ids": [],
        }
    latest = max(delays.values())
    return {
        "scheme": "paired_anchored_async_arrival_v1",
        "late_arrival_anchor_seconds": late_arrival_seconds,
        "scheduled_late_client_ids": sorted(
            client_id
            for client_id, delay in delays.items()
            if abs(delay - latest) <= 1e-9
        ),
    }
