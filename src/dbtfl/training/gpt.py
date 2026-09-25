'Efficient optional-PyTorch training for a small distilled GPT model.'

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .data import iter_prepared_records


DEFAULT_STUDENT_MODEL: Final[str] = "EleutherAI/pythia-14m"
"""Small general-purpose causal LM with a SafeTensors checkpoint. /  SafeTensors """


@dataclass(frozen=True, slots=True)
class DistilledGptTrainingConfig:
    'Model-specific local-training configuration, independent of AS and KS.'

    output_directory: Path
    initial_checkpoint_path: Path | None = None
    student_model_name: str = DEFAULT_STUDENT_MODEL
    teacher_model_name: str | None = None
    max_length: int = 128
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    num_workers: int = 2
    seed: int = 17
    precision: str = "bf16"
    checkpoint_precision: str = "fp16"
    distillation_alpha: float = 0.5
    distillation_temperature: float = 2.0
    enable_tf32: bool = True
    enable_gradient_checkpointing: bool = False
    gpu_memory_fraction: float | None = None
    require_cuda: bool = False

    def __post_init__(self) -> None:
        'Reject values that would make a run non-reproducible or invalid.'
        if not self.student_model_name:
            raise ValueError("student_model_name must not be empty / ")
        if self.teacher_model_name == "":
            raise ValueError("teacher_model_name must be None or non-empty / ")
        if self.max_length < 8 or self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch and sequence settings are invalid / ")
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid / ")
        if self.num_workers < 0 or self.warmup_steps < 0:
            raise ValueError("worker or warmup settings are invalid / ")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16 /  fp32fp16  bf16")
        if self.checkpoint_precision not in {"fp32", "fp16"}:
            raise ValueError(
                "checkpoint_precision must be fp32 or fp16 / "
                " fp32  fp16"
            )
        if not 0.0 <= self.distillation_alpha <= 1.0 or self.distillation_temperature <= 0:
            raise ValueError("distillation settings are invalid / ")
        if self.gpu_memory_fraction is not None and not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be in (0, 1] / GPU  (0, 1]")


@dataclass(frozen=True, slots=True)
class LocalTrainingResult:
    'Artifacts and measured values from one finished local training run.'

    checkpoint_path: Path
    metrics_path: Path
    sample_count: int
    optimizer_steps: int
    mean_loss: float
    elapsed_seconds: float
    device: str


def train_distilled_gpt(
    prepared_jsonl: Path,
    config: DistilledGptTrainingConfig,
) -> LocalTrainingResult:
    'Fine-tune a distilled GPT checkpoint on locally retained text only.\n    PyTorch, Transformers, and Safetensors are deliberately imported here, not\n    at package import time.  Thus AS/KS services remain runnable on machines\n    without training dependencies.  PyTorchTransformers'
    torch, transformers, safetensors_torch = _training_dependencies()
    records = tuple(iter_prepared_records(prepared_jsonl))
    if not records:
        raise ValueError("training split is empty / ")
    device = _select_device(torch)
    if config.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for this training run but no CUDA device is visible / "
            " CUDA CUDA "
        )
    gpu_memory_fraction = _resolve_gpu_memory_fraction(config)
    _configure_runtime(torch, config, device, gpu_memory_fraction)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.student_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(config.student_model_name)
    if config.initial_checkpoint_path is not None:
        initial_path = Path(config.initial_checkpoint_path).resolve()
        if not initial_path.is_file():
            raise FileNotFoundError("initial checkpoint was not found / ")
        initial_state = safetensors_torch.load_file(str(initial_path))
        model.load_state_dict(initial_state, strict=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if config.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    teacher = _load_teacher(torch, transformers, config, device)

    dataset = _TokenizedPreparedDataset(records, tokenizer, config.max_length)
    collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    optimizer = _create_optimizer(torch, model, config, device)
    scaler = _create_scaler(torch, config, device)
    scheduler = _create_scheduler(torch, optimizer, len(loader), config)
    autocast_context = _autocast_context(torch, config, device)

    model.train()
    started = time.perf_counter()
    cumulative_loss = 0.0
    optimizer_steps = 0
    batch_count = 0
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(config.epochs):
        for batch_index, batch in enumerate(loader, start=1):
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            with autocast_context():
                student_output = model(**batch)
                loss = student_output.loss
                if teacher is not None:
                    loss = _distillation_loss(
                        torch,
                        student_output.logits,
                        teacher,
                        batch,
                        loss,
                        config,
                    )
                scaled_loss = loss / config.gradient_accumulation_steps
            _backward_and_step(
                torch,
                scaled_loss,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                should_step=(
                    batch_index % config.gradient_accumulation_steps == 0
                    or batch_index == len(loader)
                ),
            )
            if batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(loader):
                optimizer_steps += 1
            cumulative_loss += float(loss.detach().float().cpu())
            batch_count += 1

    elapsed_seconds = time.perf_counter() - started
    output_directory = Path(config.output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "local_model.safetensors"
    # Preserve integer/bool buffers exactly, while serializing floating model
    # parameters in the explicitly recorded transfer precision. Training still
    # runs in ``precision`` above; this boundary only reduces AS upload and
    # global-distribution bytes.
    
    
    state_dict = {
        name: _checkpoint_tensor(torch, tensor, config.checkpoint_precision)
        for name, tensor in model.state_dict().items()
    }
    safetensors_torch.save_file(state_dict, str(checkpoint_path))
    tokenizer.save_pretrained(output_directory / "tokenizer")
    metrics_path = output_directory / "training_metrics.json"
    metrics = {
        "schema_version": "1.0",
        "sample_count": len(records),
        "optimizer_steps": optimizer_steps,
        "mean_loss": cumulative_loss / batch_count,
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "gpu_memory_fraction": gpu_memory_fraction,
        "cuda": _cuda_metrics(torch, device, gpu_memory_fraction),
        "config": _json_safe_config(config),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return LocalTrainingResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        sample_count=len(records),
        optimizer_steps=optimizer_steps,
        mean_loss=float(metrics["mean_loss"]),
        elapsed_seconds=elapsed_seconds,
        device=str(device),
    )


class _TokenizedPreparedDataset:
    'Minimal tokenization dataset to avoid coupling training to CSV details.'

    def __init__(self, records: tuple[Any, ...], tokenizer: Any, max_length: int) -> None:
        'Store local records and tokenizer configuration.'
        self._records = records
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        'Return local sample count.'
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        'Tokenize one record only when the data loader requests it.'
        return self._tokenizer(
            self._records[index].text,
            truncation=True,
            max_length=self._max_length,
            return_attention_mask=True,
        )


def _training_dependencies() -> tuple[Any, Any, Any]:
    'Load optional training packages with an actionable installation message.'
    try:
        import torch
        import transformers
        from safetensors import torch as safetensors_torch
    except ImportError as error:
        raise RuntimeError(
            "local GPT training requires requirements-training.txt; install a CUDA-compatible "
            "PyTorch build first /  GPT  requirements-training.txt CUDA  PyTorch"
        ) from error
    return torch, transformers, safetensors_torch


def _checkpoint_tensor(torch: Any, tensor: Any, checkpoint_precision: str) -> Any:
    'Return one CPU checkpoint tensor in the configured wire precision.'
    result = tensor.detach().cpu()
    if checkpoint_precision == "fp16" and result.is_floating_point():
        result = result.to(dtype=torch.float16)
    return result.contiguous()


def _select_device(torch: Any) -> Any:
    'Select a CUDA device when visible, otherwise use CPU for portability.'
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_gpu_memory_fraction(config: DistilledGptTrainingConfig) -> float | None:
    'Resolve the explicit or scheduler-provided memory share once.'
    if config.gpu_memory_fraction is not None:
        return config.gpu_memory_fraction
    raw_fraction = os.environ.get("DBTFL_GPU_MEMORY_FRACTION")
    return float(raw_fraction) if raw_fraction is not None else None


def _configure_runtime(
    torch: Any,
    config: DistilledGptTrainingConfig,
    device: Any,
    gpu_memory_fraction: float | None,
) -> None:
    'Enable 3090-safe throughput settings only on CUDA runs.'
    if device.type == "cuda" and config.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision") and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    if gpu_memory_fraction is not None:
        if not 0.0 < gpu_memory_fraction <= 1.0:
            raise ValueError("GPU memory fraction must be in (0, 1] / GPU  (0, 1]")
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=device)


def _cuda_metrics(
    torch: Any,
    device: Any,
    gpu_memory_fraction: float | None,
) -> dict[str, object]:
    'Return auditable CUDA identity and peak-memory observations.\n    These values are captured inside each isolated client process.  They therefore\n    identify the physical device exposed through ``CUDA_VISIBLE_DEVICES`` rather\n    than guessing GPU use from the evaluator parent process.'
    if device.type != "cuda":
        return {"available": False, "memory_fraction": gpu_memory_fraction}
    properties = torch.cuda.get_device_properties(device)
    return {
        "available": True,
        "name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "torch_cuda_version": str(getattr(torch.version, "cuda", "unknown")),
        "memory_fraction": gpu_memory_fraction,
    }


def _load_teacher(
    torch: Any,
    transformers: Any,
    config: DistilledGptTrainingConfig,
    device: Any,
) -> Any | None:
    'Load an optional frozen teacher for online knowledge distillation.'
    if config.teacher_model_name is None:
        return None
    teacher = transformers.AutoModelForCausalLM.from_pretrained(config.teacher_model_name)
    teacher.config.use_cache = False
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _create_optimizer(
    torch: Any,
    model: Any,
    config: DistilledGptTrainingConfig,
    device: Any,
) -> Any:
    'Create fused AdamW when supported, with a portable fallback.'
    arguments = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **arguments)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(model.parameters(), **arguments)


def _create_scheduler(
    torch: Any,
    optimizer: Any,
    batches_per_epoch: int,
    config: DistilledGptTrainingConfig,
) -> Any:
    'Create a deterministic linear warmup/decay schedule.'
    total_steps = max(
        1,
        (batches_per_epoch * config.epochs + config.gradient_accumulation_steps - 1)
        // config.gradient_accumulation_steps,
    )
    warmup_steps = min(config.warmup_steps, total_steps)

    def multiplier(step: int) -> float:
        'Return the learning-rate multiplier at one optimizer step.'
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / remaining)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _create_scaler(torch: Any, config: DistilledGptTrainingConfig, device: Any) -> Any | None:
    'Create a scaler only for CUDA FP16 runs.'
    needs_scaler = config.precision == "fp16" or (
        config.precision == "bf16" and not torch.cuda.is_bf16_supported()
    )
    if device.type == "cuda" and needs_scaler:
        return torch.amp.GradScaler("cuda")
    return None


def _autocast_context(torch: Any, config: DistilledGptTrainingConfig, device: Any) -> Any:
    'Return an autocast factory matching the selected portable precision.'
    if device.type != "cuda" or config.precision == "fp32":
        from contextlib import nullcontext

        return nullcontext
    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    return lambda: torch.autocast(device_type="cuda", dtype=dtype)


def _distillation_loss(
    torch: Any,
    student_logits: Any,
    teacher: Any,
    batch: dict[str, Any],
    cross_entropy_loss: Any,
    config: DistilledGptTrainingConfig,
) -> Any:
    'Combine causal cross entropy with padding-aware teacher KL divergence.'
    with torch.no_grad():
        teacher_logits = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
    temperature = config.distillation_temperature
    student_log_probabilities = torch.nn.functional.log_softmax(
        student_logits[:, :-1, :] / temperature,
        dim=-1,
    )
    teacher_probabilities = torch.nn.functional.softmax(
        teacher_logits[:, :-1, :] / temperature,
        dim=-1,
    )
    per_token_kl = torch.nn.functional.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1)
    valid_tokens = batch["attention_mask"][:, 1:].to(per_token_kl.dtype)
    kl_loss = (per_token_kl * valid_tokens).sum() / valid_tokens.sum().clamp_min(1)
    kl_loss = kl_loss * temperature**2
    return (
        config.distillation_alpha * cross_entropy_loss
        + (1.0 - config.distillation_alpha) * kl_loss
    )


def _backward_and_step(
    torch: Any,
    scaled_loss: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any | None,
    config: DistilledGptTrainingConfig,
    *,
    should_step: bool,
) -> None:
    'Backpropagate every batch and update only at accumulation boundaries.'
    if scaler is None:
        scaled_loss.backward()
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        return
    scaler.scale(scaled_loss).backward()
    if should_step:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)


def _json_safe_config(config: DistilledGptTrainingConfig) -> dict[str, object]:
    'Convert paths to strings before persisting an experiment configuration.'
    payload = asdict(config)
    payload["output_directory"] = str(config.output_directory)
    if config.initial_checkpoint_path is not None:
        payload["initial_checkpoint_path"] = str(config.initial_checkpoint_path)
    return payload
