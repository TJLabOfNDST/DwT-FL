"""Numerically checked weighted FedAvg for model state dictionaries.

面向模型状态字典、带数值校验的加权 FedAvg 实现。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy


class FedAvgError(ValueError):
    """Raised when client updates cannot safely participate in FedAvg.

    当客户端更新无法安全参与 FedAvg 时引发。
    """


def fedavg_arrays(
    state_dicts: Sequence[Mapping[str, numpy.ndarray]],
    sample_counts: Sequence[int],
) -> dict[str, numpy.ndarray]:
    """Compute sample-count-weighted FedAvg with strict tensor compatibility.

    使用样本数加权的 FedAvg，并严格检查张量兼容性。

    For floating tensors, ``w = Σ(n_i * w_i) / Σn_i`` is accumulated in
    float64 before casting back to the original dtype.  Integer or Boolean
    buffers must match exactly because averaging them changes model semantics.
    对浮点张量，``w = Σ(n_i * w_i) / Σn_i`` 先在 float64 中累积，再转换回原始
    数据类型；整数或布尔缓冲区必须严格一致，因为对它们求平均会改变模型语义。
    """
    if not state_dicts or len(state_dicts) != len(sample_counts):
        raise FedAvgError("updates and sample counts must be non-empty and aligned / 更新与样本数必须非空且对齐")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in sample_counts
    ):
        raise FedAvgError("sample counts must be positive integers / 样本数必须为正整数")
    expected_keys = tuple(sorted(state_dicts[0]))
    if not expected_keys:
        raise FedAvgError("state dictionaries must not be empty / 状态字典不得为空")
    for state_dict in state_dicts:
        if tuple(sorted(state_dict)) != expected_keys:
            raise FedAvgError("all updates must have identical tensor keys / 所有更新必须具有相同张量键")

    total_samples = sum(sample_counts)
    result: dict[str, numpy.ndarray] = {}
    for name in expected_keys:
        tensors = [numpy.asarray(state_dict[name]) for state_dict in state_dicts]
        reference = tensors[0]
        if any(
            tensor.shape != reference.shape or tensor.dtype != reference.dtype
            for tensor in tensors[1:]
        ):
            raise FedAvgError(
                f"tensor {name} has incompatible shape or dtype / "
                f"张量 {name} 的形状或类型不兼容"
            )
        if numpy.issubdtype(reference.dtype, numpy.floating):
            accumulator = numpy.zeros(reference.shape, dtype=numpy.float64)
            for tensor, count in zip(tensors, sample_counts, strict=True):
                accumulator += tensor.astype(numpy.float64, copy=False) * count
            result[name] = (accumulator / total_samples).astype(reference.dtype)
        elif all(numpy.array_equal(reference, tensor) for tensor in tensors[1:]):
            result[name] = reference.copy()
        else:
            raise FedAvgError(
                f"non-floating tensor {name} differs across updates / 非浮点张量 {name} 在更新间不同"
            )
    return result


def aggregate_safetensors(
    update_paths: Sequence[Path],
    sample_counts: Sequence[int],
    output_path: Path,
) -> Path:
    """Load full checkpoints, apply FedAvg, and atomically save a global model.

    加载完整检查点、执行 FedAvg 并原子保存全局模型。
    """
    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise RuntimeError(
            "FedAvg checkpoint aggregation requires safetensors and PyTorch / "
            "FedAvg 检查点聚合需要 safetensors 与 PyTorch"
        ) from error
    normalized_paths = tuple(Path(path).resolve() for path in update_paths)
    if not normalized_paths or any(not path.is_file() for path in normalized_paths):
        raise FileNotFoundError("every model update path must exist / 每个模型更新路径必须存在")
    numpy_states = [
        {name: tensor.detach().cpu().numpy() for name, tensor in load_file(str(path)).items()}
        for path in normalized_paths
    ]
    averaged_arrays = fedavg_arrays(numpy_states, sample_counts)
    averaged_tensors = {
        name: torch.from_numpy(array.copy())
        for name, array in averaged_arrays.items()
    }
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(averaged_tensors, str(temporary_path))
    temporary_path.replace(output_path)
    return output_path
