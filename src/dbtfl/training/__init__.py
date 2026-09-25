"""Local, model-specific training tools kept outside DwT-FL protocol roles.

与 DwT-FL 协议角色解耦的、本地且模型专用的训练工具。
"""

from .data import (
    PreparedDatasetManifest,
    PreparedRecord,
    iter_prepared_records,
    materialize_hot_training_split,
    prepare_csv_dataset,
)
from .gpt import DistilledGptTrainingConfig, LocalTrainingResult, train_distilled_gpt
from .scheduler import (
    ClientTrainingJob,
    ClientTrainingResult,
    mps_partitioning_status,
    run_client_training_jobs,
)

__all__ = [
    "ClientTrainingJob",
    "ClientTrainingResult",
    "DistilledGptTrainingConfig",
    "LocalTrainingResult",
    "PreparedDatasetManifest",
    "PreparedRecord",
    "iter_prepared_records",
    "materialize_hot_training_split",
    "mps_partitioning_status",
    "prepare_csv_dataset",
    "run_client_training_jobs",
    "train_distilled_gpt",
]
