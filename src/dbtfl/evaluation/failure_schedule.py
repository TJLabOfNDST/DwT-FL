"""Shared, auditable client-disconnect schedules for fault experiments.

用于故障实验的共享、可审计客户端断连调度。

The schedule deliberately uses a deterministic client order and fixed spacing.
This removes random-order noise while still ensuring that clients do not fail as
one simultaneous population.  NDSS25 imports this module through the existing
compatibility bridge so paired runs use the same event offsets.
调度刻意使用确定的客户端顺序和固定间隔。这样既避免随机顺序噪声，又确保客户端
不会作为一个同时离线的群体发生故障。NDSS25 通过已有兼容桥导入本模块，因此配对
运行使用完全相同的事件偏移。
"""

from __future__ import annotations

from typing import Iterable


def staggered_disconnect_schedule(
    client_ids: Iterable[str],
    *,
    initial_delay_seconds: float,
    interval_seconds: float,
) -> tuple[dict[str, float | int | str], ...]:
    """Return one fixed post-connection disconnect event per client.

    返回每个客户端在连接后的固定断连事件。

    Offsets are measured after the fault injector is armed, not after process
    startup or client arrival.  Every listed client has therefore connected
    before its event can occur. 偏移从故障注入器启用后开始计算，而不是从进程启动
    或客户端上线开始；因此每个列出的客户端均已连接后才会发生其事件。
    """
    if initial_delay_seconds < 0.0:
        raise ValueError("initial_delay_seconds must be non-negative / 首次断连延迟必须非负")
    if interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be positive / 断连间隔必须为正数")
    identifiers = tuple(sorted(set(client_ids)))
    if not identifiers:
        raise ValueError("client_ids must not be empty / 客户端标识不能为空")
    return tuple(
        {
            "client_id": identifier,
            "ordinal": ordinal,
            "scheduled_disconnect_after_seconds": (
                initial_delay_seconds + ordinal * interval_seconds
            ),
        }
        for ordinal, identifier in enumerate(identifiers)
    )
