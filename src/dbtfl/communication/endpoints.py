"""Stable path constants for future AS and KS implementations. / 未来 AS 与 KS 实现的稳定路径常量。"""

from __future__ import annotations

from enum import Enum


class AggregationServerPath(str, Enum):
    """AS routes reserved by the DwT-FL protocol. / DwT-FL 协议保留的 AS 路由。"""

    HEALTH = "/v1/health"
    REGISTER_CLIENT = "/v1/clients/register"
    HEARTBEAT = "/v1/clients/heartbeat"
    REGISTER_LABELS = "/v1/labels/register"
    CLAIM_TASK = "/v1/tasks/claim"
    COMMIT_TASK = "/v1/tasks/commit"
    SUBMIT_MODEL_UPDATE = "/v1/model-updates"
    AGGREGATE_MODEL_UPDATES = "/v1/model-updates/aggregate"
    DOWNLOAD_GLOBAL_MODEL = "/v1/global-models/download"
    CONFIGURE_ROUND = "/v1/rounds/configure"
    METRICS = "/v1/evaluation/metrics"
    EVALUATION_RESET = "/v1/evaluation/reset"
    EVALUATION_LEASE = "/v1/evaluation/lease"


class KeyServerPath(str, Enum):
    """KS routes reserved by the DwT-FL protocol. / DwT-FL 协议保留的 KS 路由。"""

    HEALTH = "/v1/health"
    EVALUATE_OPRF = "/v1/oprf/evaluate"
    METRICS = "/v1/evaluation/metrics"
