"""Federated model-update serialization and FedAvg aggregation helpers.

联邦模型更新序列化与 FedAvg 聚合辅助工具。
"""

from .fedavg import FedAvgError, aggregate_safetensors, fedavg_arrays
from .updates import ModelUpdateDescriptor, ModelUpdateStore, chunk_file, sha256_file

__all__ = [
    "FedAvgError",
    "ModelUpdateDescriptor",
    "ModelUpdateStore",
    "aggregate_safetensors",
    "chunk_file",
    "fedavg_arrays",
    "sha256_file",
]
