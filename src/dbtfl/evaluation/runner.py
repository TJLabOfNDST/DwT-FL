"""One-command, paper-aligned DwT-FL system evaluation.

一次执行、与论文指标对齐的 DwT-FL 系统评估。

The runner intentionally excludes comparison variants. It runs
the implemented DwT-FL HTTP roles, records raw measurements, and writes JSON,
CSV, and bilingual Markdown reports. 默认使用真实预处理数据与真实 GPT 训练；
模拟训练仅供显式指定的协议快速测试，二者绝不混报。
"""

from __future__ import annotations

import csv
import io
import json
import math
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from secrets import token_urlsafe
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath
from tempfile import TemporaryDirectory
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

from dbtfl.communication import (
    AggregationServerPath,
    CommunicationError,
    JsonHttpClient,
    TrafficRecorder,
    WireMessage,
)
from dbtfl.communication.endpoints import KeyServerPath
from dbtfl.entities import (
    ClientConfig,
    ClientEntity,
    KeyServerConfig,
    KeyServerEntity,
    LocalTrainingQueues,
    ModelUpdateOwnershipLostError,
    TaskClaimDecision,
)
from dbtfl.entities.aggregation_server import (
    AS_EVALUATION_LEASE_REQUEST,
    AS_EVALUATION_LEASE_RESPONSE,
    AS_EVALUATION_RESET_REQUEST,
    AS_EVALUATION_RESET_RESPONSE,
    AS_METRICS_REQUEST,
    AS_METRICS_RESPONSE,
)
from dbtfl.entities.key_server import KS_METRICS_REQUEST, KS_METRICS_RESPONSE
from dbtfl.oprf import OPRF_SUITE_IDENTIFIER
from .arrival_schedule import arrival_schedule_contract, paired_arrival_delays
from .failure_schedule import staggered_disconnect_schedule
from dbtfl.training import (
    ClientTrainingJob,
    PreparedRecord,
    iter_prepared_records,
    materialize_hot_training_split,
    mps_partitioning_status,
    run_client_training_jobs,
)
from dbtfl.evaluation.oprf_precompute import (
    PrecomputedOprfDataset,
    allocate_joining_records as _allocate_precomputed_joiners,
    load_precomputed_dataset,
    precomputation_is_enabled,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
"""DwT-FL project root. / DwT-FL 项目根目录。"""


EVALUATION_SUITE_NAMES: Final[frozenset[str]] = frozenset({
    "parallel_client_scale",
    "dedup_load",
    "as_backend_parallelism",
    "data_scale",
    "fault_dedup",
    "fault_training",
    "dynamic_join_base",
    "dynamic_join",
    "ablation",
})
"""Stable suite selectors accepted by focused evaluation runs.

可供定向评估运行接受的稳定套件选择器。
"""


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Paper-scale-lite sweep covering every non-comparative paper metric.

    覆盖论文所有非对比指标的论文规模轻量扫描计划。

    The default axes preserve the reference paper's 10/30/50/70/90-percent
    duplicate settings while adding a zero-overlap control.  The client-scale
    suite gives every logical client a fixed 20-percent PyTorch memory cap and
    logical GPU slot, so 4-, 8-, and 10-client observations are not confounded
    by GPU queuing. WSL CUDA compute time remains shared. This is intentionally
    far below the reference 2^19-scale cryptographic benchmark and is recorded
    as such; two RTX 3090 GPUs cannot honestly claim that setting. 默认轴保留参考
    论文的 10/30/50/70/90% 重复率，并加入 0% 对照。客户端规模套件为每个逻辑客户端
    固定分配 20% PyTorch 显存上限及逻辑 GPU 槽位，因此 4、8、10 客户端观测不会被 GPU
    排队混杂；WSL CUDA 计算时间仍共享。这刻意低于参考论文的 2^19 密码学基准，并会被
    记录；两张 RTX 3090 不能诚实地声称达到该规模。
    """

    output_directory: Path
    repetitions: int = 4
    seed: int = 17
    base_clients: int = 10
    base_duplicate_ratio: float = 0.30
    base_backend_workers: int = 4
    base_records_per_client: int = 1024
    client_counts: tuple[int, ...] = (2, 4, 6, 8, 10)
    duplicate_ratios: tuple[float, ...] = (0.0, 0.10, 0.30, 0.50, 0.70, 0.90)
    backend_worker_counts: tuple[int, ...] = (1, 2, 4, 8)
    records_per_client_values: tuple[int, ...] = (256, 512, 1024, 1536)
    # At the fixed 10-client baseline these values deliberately exercise
    # 1, 2, 4, 6, and 8 offline clients. 10-client baseline 下这些比例恰好
    # 对应 1、2、4、6、8 个离线客户端。
    failure_rates: tuple[float, ...] = (0.10, 0.20, 0.40, 0.60, 0.80)
    # Ablations use four of ten offline clients rather than the most extreme
    # fault point. This keeps every paired branch recoverable while measuring
    # exactly the same live failure workload. 消融固定令 10 个客户端中的 4 个
    # 离线，而非采用最极端的故障点；这样每个成对分支都可恢复，并测量相同的实时故障负载。
    ablation_failure_rate: float = 0.40
    join_client_count: int = 2
    join_client_counts: tuple[int, ...] = (1, 2, 4, 6)
    join_duplicate_ratio: float = 0.30
    client_arrival_min_delay_seconds: float = 0.0
    client_arrival_max_delay_seconds: float = 20.0
    client_arrival_anchor_seconds: float = 19.0
    heartbeat_interval_seconds: float = 5.0
    heartbeat_timeout_seconds: float = 300.0
    failure_heartbeat_timeout_seconds: float = 15.0
    # Fault clients first establish an AS session and then leave one by one.
    # The same event offsets are imported by NDSS25 for paired comparisons.
    # 故障客户端先建立 AS 会话，再逐个离线；NDSS25 会导入相同事件偏移用于配对比较。
    failure_disconnect_initial_delay_seconds: float = 0.10
    failure_disconnect_interval_seconds: float = 0.20
    # Fault-recovery experiments stop their paper-facing clock when all required
    # recovery training finishes; model upload, FedAvg, and downloading are not
    # part of this requested recovery metric. 故障恢复实验在所有必要恢复训练结束时
    # 停止论文计时；模型上传、FedAvg 与下载不属于本次要求的恢复指标。
    # Fault-recovery experiments end after the specified recovery training
    # boundary by default. Model upload, FedAvg, and distribution are outside
    # this metric and must not cause an intentionally disconnected SID to make
    # the case fail after recovery has already completed.
    # 故障恢复实验默认在指定的恢复训练边界结束。模型上传、FedAvg 与分发不属于
    # 该指标，也不得在恢复已完成后因刻意断开的 SID 使该用例失败。
    fault_recovery_end_at_training_completion: bool = True
    training_mode: str = "gpt"
    prepared_data_path: Path | None = None
    simulated_training_seconds_per_record: float = 0.002
    training_job_timeout_seconds: float = 1800.0
    rpc_timeout_seconds: float = 120.0
    # Heartbeats are small idempotent control requests. They need a short
    # transport deadline independent of large OPRF/model RPCs so one saturated
    # loopback connection cannot stall a case for several minutes. 心跳是小型
    # 幂等控制请求，需要独立于大型 OPRF/模型 RPC 的短传输截止时间，避免一次饱和的
    # 回环连接将整个用例阻塞数分钟。
    heartbeat_rpc_timeout_seconds: float = 2.0
    oprf_batch_size: int = 1024
    # Larger integrity-checked chunks preserve the same upload protocol while
    # substantially reducing localhost HTTP round trips for compact models.
    # 更大的完整性校验分块保持相同上传协议，同时显著减少紧凑模型的 localhost HTTP 往返。
    model_chunk_bytes: int = 2 * 1024 * 1024
    global_model_download_workers: int = 10
    gpu_ids: tuple[int, ...] = (0, 1)
    clients_per_gpu: int = 5
    gpu_memory_fraction_per_client: float = 0.20
    require_mps_partitioning: bool = False
    gpt_batch_size: int = 8
    gpt_gradient_accumulation: int = 4
    gpt_local_epochs: int = 1
    gpt_max_length: int = 128
    gpt_precision: str = "bf16"
    gpt_checkpoint_precision: str = "fp16"
    require_cuda: bool = True
    federated_rounds: int = 1
    include_ablations: bool = True
    included_suites: tuple[str, ...] = ()
    start_case: int = 1
    service_mode: str = "isolated"
    as_base_url: str | None = None
    ks_base_url: str | None = None
    evaluation_reset_token: str | None = None
    precomputed_oprf_directory: Path | None = None
    # Formal paper results must execute the complete online OPRF workflow.
    # Cached labels remain available only after an explicit diagnostic opt-in.
    # 正式论文结果必须执行完整在线 OPRF；缓存标签仅在明确选择诊断模式后允许使用。
    require_live_oprf: bool = True

    def __post_init__(self) -> None:
        """Validate sweep bounds before any services or files are created.

        在创建服务或文件前验证扫描边界。
        """
        positive = (
            self.repetitions,
            self.base_clients,
            self.base_backend_workers,
            self.base_records_per_client,
            self.join_client_count,
            self.clients_per_gpu,
            self.oprf_batch_size,
            self.global_model_download_workers,
            self.gpt_batch_size,
            self.gpt_gradient_accumulation,
            self.gpt_local_epochs,
            self.gpt_max_length,
            self.federated_rounds,
        )
        if any(value < 1 for value in positive):
            raise ValueError("count values must be positive / 计数参数必须为正数")
        if self.start_case < 1:
            raise ValueError("start_case must be at least one / 起始用例编号必须至少为一")
        for values in (
            self.client_counts,
            self.backend_worker_counts,
            self.records_per_client_values,
            self.join_client_counts,
        ):
            if not values or any(value < 1 for value in values):
                raise ValueError("sweep count values must be non-empty and positive / 扫描计数值必须非空且为正数")
        if self.training_mode not in {"simulated", "gpt"}:
            raise ValueError("training_mode must be simulated or gpt / 训练模式必须为 simulated 或 gpt")
        if self.gpt_checkpoint_precision not in {"fp32", "fp16"}:
            raise ValueError(
                "gpt_checkpoint_precision must be fp32 or fp16 / "
                "GPT 检查点精度必须为 fp32 或 fp16"
            )
        if self.service_mode not in {"isolated", "remote"}:
            raise ValueError("service_mode must be isolated or remote / 服务模式必须为 isolated 或 remote")
        if self.service_mode == "remote" and (
            not self.as_base_url or not self.ks_base_url or not self.evaluation_reset_token
        ):
            raise ValueError(
                "remote mode requires AS URL, KS URL, and reset token / "
                "远程模式需要 AS URL、KS URL 与重置令牌"
            )
        for ratio in (
            *self.duplicate_ratios,
            self.base_duplicate_ratio,
            *self.failure_rates,
            self.ablation_failure_rate,
            self.join_duplicate_ratio,
        ):
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("ratios must be in [0, 1] / 比例必须位于 [0, 1]")
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive / 心跳发送周期必须为正数")
        if self.failure_disconnect_initial_delay_seconds < 0.0:
            raise ValueError("failure disconnect initial delay must be non-negative / 故障首次断连延迟必须非负")
        if self.failure_disconnect_interval_seconds <= 0.0:
            raise ValueError("failure disconnect interval must be positive / 故障断连间隔必须为正数")
        if self.client_arrival_min_delay_seconds < 0:
            raise ValueError(
                "client arrival minimum delay must be non-negative / "
                "客户端上线最小延迟必须为非负数"
            )
        if self.client_arrival_max_delay_seconds < self.client_arrival_min_delay_seconds:
            raise ValueError(
                "client arrival maximum delay must be at least the minimum / "
                "客户端上线最大延迟必须不小于最小延迟"
            )
        if self.client_arrival_anchor_seconds < 0.0:
            raise ValueError(
                "client arrival anchor must be non-negative / "
                "客户端上线锚点必须非负"
            )
        for timeout_seconds in (
            self.heartbeat_timeout_seconds,
            self.failure_heartbeat_timeout_seconds,
        ):
            if timeout_seconds <= self.heartbeat_interval_seconds:
                raise ValueError(
                    "each heartbeat timeout must exceed its interval / "
                    "每个心跳超时必须大于发送周期"
                )
        if self.rpc_timeout_seconds <= 0:
            raise ValueError("rpc_timeout_seconds must be positive / RPC 超时必须为正数")
        if self.heartbeat_rpc_timeout_seconds <= 0:
            raise ValueError(
                "heartbeat_rpc_timeout_seconds must be positive / "
                "心跳 RPC 超时必须为正数"
            )
        if self.training_job_timeout_seconds <= 0:
            raise ValueError(
                "training_job_timeout_seconds must be positive / "
                "训练子进程超时必须为正数"
            )
        if not 1 <= self.model_chunk_bytes <= 2 * 1024 * 1024:
            raise ValueError(
                "model_chunk_bytes must be in 1..2097152 / 模型分块字节数必须位于 1..2097152"
            )
        if self.gpt_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("gpt_precision must be fp32, fp16, or bf16 / GPT 精度必须为 fp32、fp16 或 bf16")
        if not 0.0 < self.gpu_memory_fraction_per_client <= 1.0:
            raise ValueError(
                "gpu_memory_fraction_per_client must be in (0, 1] / "
                "每客户端 GPU 显存比例必须位于 (0, 1]"
            )
        if self.clients_per_gpu * self.gpu_memory_fraction_per_client > 1.0 + 1e-9:
            raise ValueError(
                "clients_per_gpu times gpu_memory_fraction_per_client exceeds one / "
                "每 GPU 客户端数与每客户端显存比例之积不能超过一"
            )
        unknown_suites = set(self.included_suites).difference(EVALUATION_SUITE_NAMES)
        if unknown_suites:
            raise ValueError(
                "included_suites contains unknown suite names: "
                f"{sorted(unknown_suites)} / included_suites 包含未知套件名称："
                f"{sorted(unknown_suites)}"
            )
        if self.require_live_oprf and self.precomputed_oprf_directory is not None:
            raise ValueError(
                "formal evaluation requires live OPRF; remove precomputed_oprf_directory or explicitly allow cached diagnostics / "
                "正式评估必须执行实时 OPRF；请移除 precomputed_oprf_directory，或明确允许缓存诊断"
            )


@dataclass(frozen=True, slots=True)
class _Case:
    """One independently repeatable system measurement. / 一次独立可重复的系统测量。"""

    suite: str
    variable: str
    value: int | float
    clients: int
    request_workers: int
    duplicate_ratio: float
    backend_workers: int
    records_per_client: int
    failure_phase: str | None = None
    joining_clients: int = 0
    ablation: str | None = None


@dataclass(slots=True)
class _TrainingFailureContext:
    """Own one asynchronous training-dropout injection and its evidence.

    管理一次异步训练掉线注入及其证据。

    The context is prepared after CAS, started only when a selected victim's
    local trainer is about to launch, and joined before the evaluator decides
    the FedAvg roster. This preserves the paper's distinction between a
    training-stage dropout and a pre-training ownership test. 该上下文在 CAS 后
    准备，仅在选中的受害者本地训练器即将启动时开始，并在评估器确定 FedAvg 名册前
    等待完成；从而保持论文中训练阶段掉线与训练前所有权测试之间的区分。
    """

    victims: tuple[ClientEntity, ...]
    survivors: tuple[ClientEntity, ...]
    victim_task_ids: dict[str, set[int]]
    victim_sids_before_disconnect: dict[str, int]
    disconnect_started_at_by_client: dict[str, float]
    recovery_target_ids: set[str]
    started: threading.Event
    completed: threading.Event
    lock: threading.Lock
    thread: threading.Thread | None = None
    recovery_latency_seconds: float | None = None
    recoveries: list[dict[str, object]] | None = None
    disconnect_events: list[dict[str, object]] | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class _DynamicJoinContext:
    """Track one late-join protocol flow running beside base training.

    记录与基础训练并行运行的一次动态加入协议流。
    """

    started: threading.Event
    completed: threading.Event
    lock: threading.Lock
    thread: threading.Thread | None = None
    result: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    error: BaseException | None = None


class _AsResourceSampler:
    """Sample only the separate AS process, never client-process resources.

    仅采样独立 AS 进程，绝不混入客户端进程资源。
    """

    def __init__(self, process_id: int) -> None:
        """Create a best-effort psutil sampler. / 创建尽力而为的 psutil 采样器。"""
        self._stop = threading.Event()
        self._maximum_rss: int | None = None
        self._process: Any | None = None
        try:
            import psutil

            self._process = psutil.Process(process_id)
            self._process.cpu_percent(None)
        except ImportError:
            pass
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        """Start peak-RSS sampling. / 开始峰值 RSS 采样。"""
        self._thread.start()

    def finish(self) -> dict[str, object]:
        """Stop sampling and return CPU plus memory observations.

        停止采样并返回 CPU 与内存观测值。
        """
        self._stop.set()
        self._thread.join(timeout=2)
        if self._process is None:
            return {"status": "unavailable", "reason": "psutil_not_installed"}
        try:
            return {
                "status": "available",
                "cpu_percent": self._process.cpu_percent(None),
                "peak_rss_bytes": self._maximum_rss,
            }
        except Exception as error:
            return {"status": "unavailable", "reason": type(error).__name__}

    def _sample(self) -> None:
        """Record peak resident memory at a short fixed interval.

        以短固定间隔记录峰值驻留内存。
        """
        while not self._stop.wait(0.05):
            if self._process is None:
                return
            try:
                rss = int(self._process.memory_info().rss)
            except Exception:
                return
            self._maximum_rss = max(rss, self._maximum_rss or 0)


class _LocalAsProcess:
    """Launch a separate local AS process for honest AS resource metrics.

    为诚实的 AS 资源指标启动独立本地 AS 进程。
    """

    def __init__(
        self,
        root: Path,
        case: _Case,
        plan: EvaluationPlan,
        heartbeat_timeout_seconds: float,
    ) -> None:
        """Build a compatible native library and prepare one local AS command.

        构建兼容原生库并准备一个本地 AS 命令。
        """
        self._root = root
        self._port = _unused_port()
        # The child AS receives an unpredictable control token so the evaluator
        # can switch only its fault-injection lease without clearing protocol
        # state. 子进程 AS 接收不可预测控制令牌，使评估器可以只切换故障注入
        # 租约而不清除协议状态。
        self.evaluation_control_token = token_urlsafe(32)
        self._log_path = root / "as.log"
        self._library_path = root / f"atomic_word_evaluation{_library_suffix()}"
        subprocess.run(
            [sys.executable, "scripts/build_native.py", "--output", str(self._library_path)],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        command = [
            sys.executable, "scripts/run_as.py", "--host", "127.0.0.1", "--port", str(self._port),
            "--capacity", str(case.clients * case.records_per_client * 2 + 16),
            "--max-clients", str(case.clients + case.joining_clients + 8),
            "--max-edges", str(case.clients * case.records_per_client * 4 + 32),
            "--backend-workers", str(case.backend_workers),
            "--heartbeat-interval", str(plan.heartbeat_interval_seconds),
            "--heartbeat-timeout", str(heartbeat_timeout_seconds),
            "--model-update-directory", str(root / "model-updates"),
            "--native-library", str(self._library_path),
            # A URL-safe control token may legitimately start with ``-``.
            # Passing it as a separate argv item makes argparse mistake it for
            # another option on every supported platform. Keep the option and
            # value in one unambiguous argument. URL 安全控制令牌可以合法地以
            # ``-`` 开头；若将其作为独立 argv 项传递，argparse 会在所有受支持
            # 平台上将其误认为另一个选项。因此将选项和值合并为一个无歧义参数。
            f"--evaluation-reset-token={self.evaluation_control_token}",
        ]
        if case.ablation == "without_cas":
            command.extend(["--claim-mode", "mutex"])
        elif case.ablation == "without_inverse_index":
            command.extend(["--recovery-index-mode", "scan"])
        elif case.ablation == "without_history_scheduling":
            command.append("--disable-history-scheduling")
        self._log_stream = self._log_path.open("w", encoding="utf-8", newline="")
        self.process: Any | None = None
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
            )
            _wait_for_as_ready("127.0.0.1", self._port, self.process)
        except Exception:
            # Do not let a failed child startup leak an inherited log handle.
            # Windows cannot remove a temporary directory while that handle is
            # open, which used to conceal the original AS startup error. 子进程
            # 启动失败时不得泄漏继承的日志句柄。Windows 在句柄仍打开时无法移除
            # 临时目录，过去因此掩盖了原始 AS 启动错误。
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
            self._log_stream.close()
            raise

    @property
    def base_url(self) -> str:
        """Return the loopback AS URL. / 返回回环 AS URL。"""
        return f"http://127.0.0.1:{self._port}"

    def assert_ready(self) -> None:
        """Confirm that the child still serves an AS control-plane response.

        确认子进程仍可提供 AS 控制面响应。
        """
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("AS process exited before client protocol startup / AS 进程在客户端协议启动前退出")
        _request_as_metrics(self.base_url, timeout_seconds=0.5)

    def close(self) -> None:
        """Terminate the AS and preserve its diagnostic log. / 终止 AS 并保留诊断日志。"""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self._log_stream.close()

    def copy_log_to(self, destination: Path) -> Path:
        """Flush and retain the AS diagnostic log before temporary cleanup.

        在临时目录清理前刷新并保留 AS 诊断日志。
        """
        self._log_stream.flush()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._log_path, destination)
        return destination


class EvaluationRunner:
    """Run all required DwT-FL metric suites and materialize their evidence.

    运行全部必需 DwT-FL 指标套件并固化其证据。
    """

    def __init__(self, plan: EvaluationPlan, *, progress: Callable[[str], None] = print) -> None:
        """Store a plan and a caller-owned progress sink. / 保存计划与调用方持有的进度输出器。"""
        self.plan = plan
        self.progress = progress
        self._prepared_data_path: Path | None = None
        self._prepared_records: tuple[PreparedRecord, ...] = ()
        self._precomputed_oprf: PrecomputedOprfDataset | None = None

    def run(self) -> Path:
        """Run every non-comparative indicator and write formatted result files.

        运行每项非对比指标并写入格式化结果文件。
        """
        cases = _selected_cases(self.plan)
        if self.plan.precomputed_oprf_directory is not None:
            _validate_precomputed_oprf_cases(self.plan, cases)
        root = self.plan.output_directory.resolve()
        root.mkdir(parents=True, exist_ok=True)
        total = len(cases)
        if self.plan.start_case > total:
            raise ValueError(
                f"start_case={self.plan.start_case} exceeds {total} planned cases / "
                f"起始用例编号 {self.plan.start_case} 超出计划用例总数 {total}"
            )
        # A resumed run retains only the completed prefix. Results at and after
        # start_case are intentionally replaced because they may have been
        # produced by an earlier implementation. This prevents stale and new
        # measurements from being silently mixed in one paper table. 续跑仅保留
        # 起始用例前已完成的前缀；起始用例及之后的旧结果会被刻意替换，因为它们可能
        # 来自旧实现，从而避免在同一论文表格中静默混入新旧测量。
        results = _load_completed_trimmed_prefix(
            root, self.plan, cases, self.plan.start_case
        )
        completed = len(results)
        if completed:
            self.progress(
                f"[resume] retained {completed} completed cases; restarting at "
                f"{self.plan.start_case}/{total} / [续跑] 已保留 {completed} 个已完成用例；"
                f"将从 {self.plan.start_case}/{total} 重新开始"
            )
            _write_reports(self.plan, results)
        _write_run_status(root, "running", completed, total, results)
        try:
            self._ensure_prepared_data()
            if self.plan.training_mode == "gpt" and self.plan.require_cuda:
                self._verify_configured_cuda_devices(root)
            _write_run_metadata(root, self.plan, self._prepared_data_path)
            if self.plan.training_mode == "gpt" and not self.plan.require_mps_partitioning:
                mps_status = mps_partitioning_status()
                if not mps_status["available"]:
                    self.progress(
                        "[resource-policy] shared CUDA slots with a 20% memory cap; "
                        "hard MPS compute partitioning is unavailable / "
                        "[资源策略] 共享 CUDA 槽位与 20% 显存上限；硬性 MPS 计算分区不可用"
                    )
            for case in cases[completed:]:
                completed += 1
                self.progress(
                    f"[{completed}/{total}] {case.suite}: {case.variable}={case.value}"
                )
                repetitions: list[dict[str, object]] = []
                failures: list[dict[str, object]] = []
                for repetition in range(self.plan.repetitions):
                    self.progress(
                        f"  [repeat {repetition + 1}/{self.plan.repetitions}]"
                    )
                    try:
                        raw = self._run_case(case, repetition)
                        raw.setdefault("status", "completed")
                        repetitions.append(raw)
                    except Exception as error:
                        # Each repeat owns isolated services and temporary
                        # files. Preserve the exact failure but continue the
                        # other repeats so one transient failure never hides
                        # its diagnostic evidence. 每次重复拥有隔离服务和临时
                        # 文件；保留准确失败，但继续其余重复，避免瞬态失败掩盖诊断证据。
                        failure = self._failed_case_result(case, repetition, error)
                        failures.append(failure)
                        self.progress(
                            f"[repeat failed] {case.suite}: {case.variable}={case.value}; "
                            f"repeat={repetition + 1}; {type(error).__name__}: {error} / "
                            f"[重复失败] {case.suite}: {case.variable}={case.value}；"
                            f"第 {repetition + 1} 次；{type(error).__name__}: {error}"
                        )
                        self.progress(
                            "[failure-traceback] / [失败回溯]\n"
                            f"{failure['failure']['traceback']}"
                        )
                result = (
                    _trimmed_mean_case_result(repetitions)
                    if not failures
                    else _failed_trimmed_case_result(case, repetitions, failures)
                )
                results.append(result)
                # Persist the final case-level aggregate immediately. A later
                # failure never discards completed trimmed means. 每个用例的最终
                # 聚合值立刻持久化；后续失败绝不丢弃已完成截尾均值。
                _write_reports(self.plan, results)
                _write_run_status(root, "running", completed, total, results)
        except BaseException as error:
            # Preserve both completed measurements and the actionable failure when
            # a run stops early, including an operator interruption. 当运行提前停止
            # （包括操作员中断）时，同时保留已完成测量和可定位的失败信息。
            _write_reports(self.plan, results)
            _write_run_status(root, "failed", len(results), total, results, error)
            raise
        _write_reports(self.plan, results)
        _write_run_status(root, "completed", completed, total, results)
        return root

    def _verify_configured_cuda_devices(self, root: Path) -> None:
        """Require one completed CUDA kernel on every configured physical GPU.

        要求每一张已配置物理 GPU 都至少完成一个 CUDA 核函数。

        Scheduler assignment alone is not sufficient evidence: each isolated
        child imports PyTorch, allocates a tensor, executes matrix
        multiplication, and synchronizes on the physical device exposed by
        ``CUDA_VISIBLE_DEVICES``. The resulting JSON files are retained with
        the experiment rather than relying on a transient ``nvidia-smi`` view.
        仅有调度器分配并不足以构成证据：每个隔离子进程都会导入 PyTorch、分配张量、
        执行矩阵乘法，并在 ``CUDA_VISIBLE_DEVICES`` 暴露的物理设备上同步。所得 JSON
        文件会随实验保留，而不是依赖瞬时的 ``nvidia-smi`` 视图。
        """
        evidence_root = root / "gpu-preflight"
        jobs = [
            ClientTrainingJob(
                client_id=f"gpu-{gpu_id}",
                command=(
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "probe_cuda_device.py"),
                    "--output",
                    str(evidence_root / f"gpu-{gpu_id}.json"),
                ),
                log_path=evidence_root / f"gpu-{gpu_id}.log",
            )
            for gpu_id in self.plan.gpu_ids
        ]
        results = run_client_training_jobs(
            jobs,
            self.plan.gpu_ids,
            clients_per_gpu=1,
            gpu_memory_fraction=1.0,
            timeout_seconds=min(self.plan.training_job_timeout_seconds, 120.0),
        )
        failed = [result for result in results if result.return_code]
        assigned = {result.gpu_id for result in results}
        expected = set(self.plan.gpu_ids)
        if failed or assigned != expected:
            failure_logs = ", ".join(str(result.log_path) for result in failed)
            raise RuntimeError(
                "CUDA preflight did not execute on every configured GPU: "
                f"assigned={sorted(assigned)}, expected={sorted(expected)}, logs={failure_logs} / "
                "CUDA 预检未在每张已配置 GPU 上执行"
            )
        evidence: dict[str, object] = {}
        for gpu_id in self.plan.gpu_ids:
            path = evidence_root / f"gpu-{gpu_id}.json"
            if not path.is_file():
                raise RuntimeError(
                    f"CUDA preflight did not write {path} / CUDA 预检未写入 {path}"
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("cuda_available") is not True
                or payload.get("visible_cuda_device_count") != 1
                or payload.get("visible_devices") != str(gpu_id)
            ):
                raise RuntimeError(
                    f"CUDA preflight evidence is invalid for GPU {gpu_id} / "
                    f"GPU {gpu_id} 的 CUDA 预检证据无效"
                )
            evidence[str(gpu_id)] = payload
        (evidence_root / "summary.json").write_text(
            json.dumps(
                {
                    "configured_physical_gpu_ids": list(self.plan.gpu_ids),
                    "evidence": evidence,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    def _failed_case_result(
        self,
        case: _Case,
        repetition: int,
        error: Exception,
    ) -> dict[str, object]:
        """Encode one isolated case failure without fabricating metric values.

        对一个隔离用例失败进行编码，且绝不伪造指标数值。
        """
        return {
            "schema_version": "1.0",
            "status": "failed",
            "suite": case.suite,
            "variable": case.variable,
            "value": case.value,
            "repetition": repetition,
            "configuration": asdict(case),
            "training_mode": self.plan.training_mode,
            "service_mode": self.plan.service_mode,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "diagnostic_paths": list(
                    getattr(error, "dbtfl_diagnostic_paths", ())
                ),
            },
        }

    def _run_case(self, case: _Case, repetition: int) -> dict[str, object]:
        """Execute one full protocol-to-submission round and capture raw values.

        执行从协议到提交的一次完整轮次并捕获原始数值。
        """
        self._ensure_prepared_data()
        # All protocol setup runs under the normal long lease. A short lease is
        # activated only after registration and CAS, immediately before an
        # intentional training-dropout injection. 所有协议准备阶段均在正常长
        # 租约下运行；短租约仅在登记和 CAS 后、显式注入训练掉线前启用。
        heartbeat_timeout_seconds = self._initial_heartbeat_timeout_seconds(case)
        if self.plan.service_mode == "remote":
            if case.ablation is not None:
                raise ValueError(
                    "ablations require isolated local AS because they alter server internals / "
                "消融需要隔离本地 AS，因为它们会修改服务端内部实现"
            )
            if self.plan.precomputed_oprf_directory is not None:
                raise ValueError(
                    "precomputed OPRF requires the matching local persistent KS key and cannot use remote mode / "
                    "预计算 OPRF 需要匹配的本地持久 KS 密钥，不能使用 remote 模式"
                )
            self._reset_remote_as(case, heartbeat_timeout_seconds)
        with TemporaryDirectory(prefix="dbtfl-evaluation-") as temporary_name:
            root = Path(temporary_name)
            as_process: _LocalAsProcess | None = None
            sampler: _AsResourceSampler | None = None
            ks: KeyServerEntity | None = None
            if self.plan.service_mode == "remote":
                assert self.plan.as_base_url is not None and self.plan.ks_base_url is not None
                as_url = self.plan.as_base_url
                ks_url = self.plan.ks_base_url
                key_path: Path | None = None
            else:
                as_process = _LocalAsProcess(root, case, self.plan, heartbeat_timeout_seconds)
                sampler = _AsResourceSampler(as_process.process.pid)
                sampler.start()
                key_path = (
                    self._precomputed_oprf.key_path
                    if self._precomputed_oprf is not None
                    else root / "ks-ristretto255-key.json"
                )
                ks = KeyServerEntity(KeyServerConfig(key_path=key_path, host="127.0.0.1", port=0))
                as_url = as_process.base_url
                ks_url = ks.base_url
            clients: list[ClientEntity] = []
            try:
                if ks is not None:
                    ks.start()
                clients = self._create_clients(root, as_url, ks_url, case)
                records_by_client = self._records_for_case(case, repetition)
                if as_process is not None:
                    # Recheck the concrete AS control-plane response after
                    # local data setup but before the measured arrival
                    # schedule. This is not a protocol barrier and lies
                    # outside the paper's end-to-end interval; it prevents a
                    # dead helper process from being misreported as a client
                    # registration failure. 在本地数据准备后、测量上线调度前，
                    # 再次检查真实 AS 控制面响应。它不是协议屏障，且不计入论文
                    # 端到端区间；这样不会把已退出的辅助进程误报为客户端注册失败。
                    as_process.assert_ready()
                base_records = {
                    client.config.client_id: records_by_client[client.config.client_id]
                    for client in clients
                }
                dedup_recovery_latency: float | None = None
                dedup_fault_recoveries: list[dict[str, object]] = []
                dedup_disconnect_events: list[dict[str, object]] = []
                # The end-to-end clock begins when the evaluator releases the
                # client population to its random online schedule, not while
                # it starts local helper services or materializes data. This
                # matches the paper's client-request-to-model-distribution
                # definition. 端到端时钟从评估器将客户端群体交给随机上线调度时
                # 开始，而不是在启动本地辅助服务或物化数据时开始；这与论文中从客户
                # 端请求到全局模型分发结束的定义一致。
                measurement_started = time.perf_counter()
                # A normal DwT round has no global registration/CAS barrier:
                # every client appears after its own reproducible random delay,
                # connects, completes FP_i submission and CAS, then becomes
                # eligible for local training.  The evaluator waits only before
                # FedAvg, when the paper requires every participant to finish.
                # 正常 DwT 轮次不存在全局登记/CAS 屏障：每个客户端在其可复现的
                # 随机延迟后上线、连接、完成 FP_i 提交和 CAS，随后即可进入本地训练；
                # 评估器仅在论文要求所有参与方完成的 FedAvg 前等待。
                if case.failure_phase == "dedup" and case.value != 0.0:
                    (
                        dedup,
                        claims,
                        queues,
                        arrival_metrics,
                        dropped_dedup_clients,
                    ) = self._run_asynchronous_dedup_dropout_protocols(
                        clients, records_by_client, case, repetition
                    )
                    failed_dedup_seconds = 0.0
                    dedup_disconnect_events = [
                        {
                            "client_id": client_id,
                            "ordinal": int(timing["disconnect_ordinal"]),
                            "scheduled_disconnect_after_seconds": float(
                                timing["scheduled_disconnect_after_connection_seconds"]
                            ),
                            "actual_disconnect_after_seconds": float(
                                timing["actual_disconnect_after_connection_seconds"]
                            ),
                            "connection_state_before_disconnect": "connected",
                        }
                        for client_id, timing in arrival_metrics.items()
                        if "disconnect_ordinal" in timing
                    ]
                else:
                    failed_dedup_seconds = 0.0
                    dropped_dedup_clients = ()
                    dedup, claims, queues, arrival_metrics = (
                        self._run_asynchronous_initial_protocols(
                            clients, records_by_client, case, repetition
                        )
                    )
                if dropped_dedup_clients:
                    rejoin_dedup, rejoin_claims, rejoin_queues = self._rejoin_dedup_dropouts(
                        dropped_dedup_clients,
                        records_by_client,
                    )
                    # A recovered client restarts only its unfinished
                    # deduplication request. Its existing OPRF mapping is
                    # reused; all online clients retain their completed CAS
                    # result. 掉线客户端仅重新发起未完成的去重请求，复用既有 OPRF
                    # 映射；所有在线客户端保留已完成的 CAS 结果。
                    for phase, recovery_phase in (
                        (dedup, rejoin_dedup),
                        (claims, rejoin_claims),
                    ):
                        phase["wall_seconds"] += recovery_phase["wall_seconds"]
                        phase["accumulated_seconds"] += recovery_phase["accumulated_seconds"]
                        phase["values"].update(recovery_phase["values"])
                    queues.update(rejoin_queues)
                    # The requested recovery boundary is zero for a DwT-FL
                    # dedup-stage failure: no AS task/owner state exists yet,
                    # so AS performs no failure-location or reassignment work.
                    # The resumed client RPC cost is retained separately and
                    # remains part of end-to-end time. 请求的 DwT-FL 去重阶段
                    # 恢复边界为零：此时尚未存在 AS 任务或所有者状态，因此 AS 不执行
                    # 故障定位或重新分配；恢复客户端 RPC 耗时单独保留，仍计入端到端时间。
                    dedup_recovery_latency = 0.0
                    dedup_fault_recoveries = [
                        {
                            "victim_client_id": victim.config.client_id,
                            "successor_client_id": None,
                            "recovery_latency_seconds": dedup_recovery_latency,
                            "recovery_kind": "no_as_state_recovery",
                            "resumed_client_protocol_seconds": (
                                rejoin_dedup["wall_seconds"] + rejoin_claims["wall_seconds"]
                            ),
                        }
                        for victim in dropped_dedup_clients
                    ]
                # Training faults are prepared after normal registration/CAS,
                # but are started only once an affected local trainer launches.
                # This is essential: a CAS-stage disconnect is a different
                # paper scenario and must not be counted as training recovery.
                # 训练故障在正常登记/CAS 后准备，但仅在受影响的本地训练器启动时注入。
                # 这很关键：CAS 阶段断开属于另一种论文场景，不能计入训练恢复。
                recovery_latency = dedup_recovery_latency
                fault_recoveries = list(dedup_fault_recoveries)
                training_failure_context = self._prepare_training_failure(
                    case,
                    clients,
                    claims,
                    evaluation_control_token=(
                        self.plan.evaluation_reset_token
                        if self.plan.service_mode == "remote"
                        else as_process.evaluation_control_token
                    ),
                )
                dynamic_join_context = self._prepare_dynamic_join(case)
                training = {
                    "wall_seconds": 0.0,
                    "accumulated_seconds": 0.0,
                    "checkpoints": {},
                    "client_metrics": {},
                }
                submissions = {
                    "submitted_client_count": 0,
                    "ownership_retrain_count": 0,
                    "model_upload_wall_seconds": 0.0,
                    "client_metrics": {},
                }
                dynamic_join_protocol = {
                    "dedup_wall_seconds": 0.0,
                    "dedup_accumulated_seconds": 0.0,
                    "joining_client_count": 0,
                    "joining_oprf_wall_seconds": 0.0,
                    "joining_oprf_accumulated_seconds": 0.0,
                }
                round_metrics: list[dict[str, object]] = []
                training_completion_elapsed: float | None = None
                initial_checkpoints: dict[str, Path] | None = None
                case_round_count = _round_count_for_case(self.plan, case)
                for round_id in range(1, case_round_count + 1):
                    if round_id > 1:
                        # AS has reset task states after the prior FedAvg. The
                        # next heartbeat is the paper-defined source of history
                        # scheduling instructions. 前一轮 FedAvg 后 AS 已重置任务
                        # 状态；下一次心跳是论文定义的历史调度指令来源。
                        # Preserve the first-round CAS measurements and add the
                        # actual second-round heartbeat scheduling measurement.
                        # Replacing ``claims`` with a values-only dictionary
                        # loses the accumulated timing keys required by final
                        # metric serialization. 保留首轮 CAS 测量，并加入真实的
                        # 第二轮心跳调度测量；若仅用 values 字典覆盖 ``claims``，
                        # 会丢失最终指标序列化所需的累计时间字段。
                        prior_claim_wall_seconds = float(claims["wall_seconds"])
                        prior_claim_accumulated_seconds = float(
                            claims["accumulated_seconds"]
                        )
                        scheduling_phase = _parallel_phase(
                            clients,
                            case.request_workers,
                            lambda client: client.send_as_heartbeat(),
                        )
                        claims = {
                            "values": {
                                client.config.client_id: (
                                    list(_round_instruction_decisions(client)), 0.0
                                )
                                for client in clients
                            },
                            "wall_seconds": (
                                prior_claim_wall_seconds
                                + scheduling_phase["wall_seconds"]
                            ),
                            "accumulated_seconds": (
                                prior_claim_accumulated_seconds
                                + scheduling_phase["accumulated_seconds"]
                            ),
                        }
                        queues = {
                            client.config.client_id: client.route_round_instructions()
                            for client in clients
                        }
                    participants = _current_training_participants(clients, claims, queues)
                    if not participants:
                        raise RuntimeError("round has no TRAIN participants / 轮次没有 TRAIN 参与者")
                    participant_sids = tuple(
                        client.as_session.sid for client in participants
                        if client.as_session is not None
                    )
                    if len(participant_sids) != len(participants):
                        raise RuntimeError("round participant lost its AS session / 轮次参与者丢失 AS 会话")
                    prior_train_labels = {
                        client.config.client_id: {
                            decision.protected_label
                            for decision in claims["values"][client.config.client_id][0]
                            if decision.operation == "TRAIN"
                        }
                        for client in participants
                    }
                    round_training = self._run_training(
                        root,
                        participants,
                        queues,
                        repetition,
                        initial_checkpoints=initial_checkpoints,
                        on_job_start=(
                            lambda job: self._on_initial_training_job_start(
                                job,
                                training_failure_context,
                                dynamic_join_context,
                                root,
                                as_url,
                                ks_url,
                                case,
                                clients,
                                records_by_client,
                                repetition,
                            )
                            if round_id == 1 else None
                        ),
                        cancel_requested=(
                            lambda job: training_failure_context is not None
                            and job.client_id in training_failure_context.disconnect_started_at_by_client
                        ),
                        allowed_cancelled_client_ids=(
                            {
                                client.config.client_id
                                for client in training_failure_context.victims
                            }
                            if training_failure_context is not None else set()
                        ),
                    )
                    for key in ("wall_seconds", "accumulated_seconds"):
                        training[key] += round_training[key]
                    training["checkpoints"] = round_training["checkpoints"]
                    training["client_metrics"].update(round_training["client_metrics"])
                    if dynamic_join_context is not None and round_id == 1:
                        join_dedup, join_claims, join_queues = self._await_dynamic_join(
                            dynamic_join_context
                        )
                        claims["values"].update(join_claims["values"])
                        claims["accumulated_seconds"] += join_claims["accumulated_seconds"]
                        queues.update(join_queues)
                        joiners = list(clients[-case.joining_clients:])
                        dynamic_join_protocol = {
                            "dedup_wall_seconds": (
                                join_dedup["wall_seconds"] + join_claims["wall_seconds"]
                            ),
                            "dedup_accumulated_seconds": (
                                join_dedup["accumulated_seconds"]
                                + join_claims["accumulated_seconds"]
                            ),
                            "joining_client_count": len(joiners),
                            "joining_oprf_wall_seconds": join_dedup.get(
                                "oprf_wall_seconds", 0.0
                            ),
                            "joining_oprf_accumulated_seconds": join_dedup.get(
                                "oprf_accumulated_seconds", 0.0
                            ),
                        }
                        join_participants = _current_training_participants(
                            joiners, claims, queues
                        )
                        if join_participants:
                            join_training = self._run_training(
                                root,
                                join_participants,
                                queues,
                                repetition,
                            )
                            for key in ("wall_seconds", "accumulated_seconds"):
                                training[key] += join_training[key]
                                round_training[key] += join_training[key]
                            round_training["checkpoints"].update(
                                join_training["checkpoints"]
                            )
                            round_training["client_metrics"].update(
                                join_training["client_metrics"]
                            )
                            participants.extend(join_participants)
                    if training_failure_context is not None and round_id == 1:
                        recovery_latency, fault_recoveries = self._await_training_failure(
                            training_failure_context
                        )
                        # Faulted clients remain offline in this round.  Only
                        # live holders may receive newly released duplicate
                        # tasks; a task with no live holder stays EMPTY and is
                        # deliberately deferred to the next round. 掉线客户端
                        # 在本轮保持离线；只有在线持有者可接收新释放的重复任务；没有
                        # 在线持有者的任务保持 EMPTY，并明确延后到下一轮。
                        changed_clients = self._synchronize_recovered_training_ownership(
                            training_failure_context.survivors,
                            claims,
                            queues,
                        )
                        self._defer_offline_training_clients(
                            training_failure_context.victims,
                            claims,
                            queues,
                        )
                        victim_identifiers = {
                            client.config.client_id
                            for client in training_failure_context.victims
                        }
                        recovery_clients: list[ClientEntity] = []
                        recovery_queues: dict[str, LocalTrainingQueues] = {}
                        recovery_initial_checkpoints: dict[str, Path | None] = {}
                        for client in clients:
                            identifier = client.config.client_id
                            if identifier not in changed_clients:
                                continue
                            current_decisions = claims["values"][identifier][0]
                            current_train_labels = {
                                decision.protected_label
                                for decision in current_decisions
                                if decision.operation == "TRAIN"
                            }
                            if not current_train_labels:
                                continue
                            prior_labels = prior_train_labels.get(identifier, set())
                            if identifier in victim_identifiers:
                                # A deliberately disconnected client cannot
                                # contribute an update in this round. Its work
                                # is either taken over by a live peer or left
                                # EMPTY for the next round. 被刻意断开的客户端
                                # 本轮不得提交更新；其任务要么由在线同伴接管，要么保留
                                # 为 EMPTY 并留到下一轮。
                                continue
                            lost = prior_labels.difference(current_train_labels)
                            if lost:
                                # The local training set changed by removal or
                                # replacement. An existing checkpoint contains
                                # parameters fitted to data no longer owned by
                                # this client, so it cannot be safely reused.
                                # Restart from the global checkpoint on the
                                # current complete TRAIN set, exactly as the
                                # recovery rule requires. 本地训练集因删除或
                                # 替换而改变。既有检查点包含该客户端已不再拥有的
                                # 数据，不能安全复用；因此按恢复规则从全局检查点
                                # 在当前完整 TRAIN 集合上重新训练。
                                recovery_queues[identifier] = client.route_claim_decisions(
                                    [
                                        decision
                                        for decision in current_decisions
                                        if decision.operation == "TRAIN"
                                    ]
                                )
                                recovery_initial_checkpoints[identifier] = None
                                recovery_clients.append(client)
                                continue
                            added = [
                                decision for decision in current_decisions
                                if decision.operation == "TRAIN"
                                and decision.protected_label not in prior_labels
                            ]
                            if not added:
                                continue
                            # A live successor keeps the checkpoint trained on
                            # its original hot queue and incrementally trains
                            # only the newly assigned records. A client that had
                            # no original TRAIN task starts from the global
                            # checkpoint on its newly assigned records. 在线接管
                            # 者保留原热队列训练后的检查点，只对新分配记录增量训练；
                            # 原本没有 TRAIN 任务的客户端则从全局检查点开始训练新记录。
                            recovery_queues[identifier] = client.route_claim_decisions(added)
                            recovery_initial_checkpoints[identifier] = (
                                round_training["checkpoints"].get(identifier)
                            )
                            recovery_clients.append(client)
                        if recovery_clients:
                            recovered_training = self._run_training(
                                root,
                                recovery_clients,
                                recovery_queues,
                                repetition,
                                initial_checkpoints=recovery_initial_checkpoints,
                            )
                            for key in ("wall_seconds", "accumulated_seconds"):
                                training[key] += recovered_training[key]
                                round_training[key] += recovered_training[key]
                            round_training["checkpoints"].update(
                                recovered_training["checkpoints"]
                            )
                            round_training["client_metrics"].update(
                                recovered_training["client_metrics"]
                            )
                        # The recovery benchmark ends when the AS has issued
                        # current TRAIN instructions and every required local
                        # recovery job has finished. Model transfer is outside
                        # the metric boundary requested for this experiment.
                        # 恢复基准在 AS 下达当前 TRAIN 指令且所有必要本地恢复训练完成
                        # 时结束；模型传输不属于本次要求的指标边界。
                        if self.plan.fault_recovery_end_at_training_completion:
                            training_completion_elapsed = time.perf_counter() - measurement_started
                            round_metrics.append({
                                "round_id": round_id,
                                "participant_count": len(_current_training_participants(clients, claims, queues)),
                                "training_wall_seconds": round_training["wall_seconds"],
                                "measurement_boundary": "training_completed",
                            })
                            break
                        # Recovery may have transferred a task, but the roster
                        # is frozen only now, after every affected client has a
                        # valid current checkpoint. 恢复可能转移任务，但名册仅在
                        # 此处冻结：所有受影响客户端均已获得有效的当前检查点。
                        participants = _current_training_participants(clients, claims, queues)
                        participant_sids = tuple(
                            client.as_session.sid for client in participants
                            if client.as_session is not None
                        )
                        if len(participant_sids) != len(participants):
                            raise RuntimeError(
                                "recovered participant lost its AS session / "
                                "恢复后的参与客户端丢失 AS 会话"
                            )
                    else:
                        participant_sids = tuple(
                            client.as_session.sid for client in participants
                            if client.as_session is not None
                        )
                        if len(participant_sids) != len(participants):
                            raise RuntimeError(
                                "round participant lost its AS session / "
                                "轮次参与者丢失 AS 会话"
                            )
                    if (
                        self.plan.fault_recovery_end_at_training_completion
                        and case.failure_phase is not None
                        and round_id == 1
                        and training_completion_elapsed is None
                    ):
                        # Dedup faults have no AS-side recovery transition and
                        # training faults have already completed the verified
                        # instruction/retraining sequence above. Both requested
                        # cases therefore share this training-complete clock.
                        # 去重故障没有 AS 侧恢复迁移；训练故障已在上方完成经验证的
                        # 指令/重训序列。两类请求用例都在此共享训练完成计时边界。
                        training_completion_elapsed = time.perf_counter() - measurement_started
                        round_metrics.append({
                            "round_id": round_id,
                            "participant_count": len(participants),
                            "training_wall_seconds": round_training["wall_seconds"],
                            "measurement_boundary": "training_completed",
                        })
                        break
                    if self.plan.training_mode == "gpt":
                        participants[0].configure_federated_round_at_as(round_id, participant_sids)
                    round_submissions = self._submit_updates(
                        root,
                        participants,
                        claims,
                        queues,
                        round_training,
                        repetition,
                        round_id=round_id,
                    )
                    incremental = self._apply_pre_aggregate_incremental_training(
                        root, participants, claims, queues, round_training,
                        round_submissions, repetition, round_id,
                    )
                    round_submissions["ownership_retrain_count"] += incremental["replacement_count"]
                    round_submissions["client_metrics"].update(incremental["client_metrics"])
                    submissions["submitted_client_count"] += round_submissions["submitted_client_count"]
                    submissions["ownership_retrain_count"] += round_submissions["ownership_retrain_count"]
                    submissions["model_upload_wall_seconds"] += round_submissions["wall_seconds"]
                    submissions["client_metrics"].update(round_submissions["client_metrics"])
                    # Ownership recovery can replace a trainer metric after the
                    # first snapshot above. Retain the final fresh checkpoint
                    # evidence. 所有权恢复可能在上述首个快照后替换训练器指标；这里
                    # 保留最终新检查点的证据。
                    training["client_metrics"].update(round_training["client_metrics"])
                    aggregation_seconds: float | None = None
                    distribution = {"wall_seconds": 0.0, "accumulated_seconds": 0.0}
                    initial_checkpoints = None
                    if self.plan.training_mode == "gpt":
                        submitted_sids = tuple(round_submissions["submitted_sids"])
                        if set(submitted_sids) != set(participant_sids):
                            raise RuntimeError(
                                "FedAvg roster and submitted updates differ: "
                                f"expected={sorted(participant_sids)}, "
                                f"submitted={sorted(submitted_sids)} / "
                                "FedAvg 名册与已提交更新不一致"
                            )
                        aggregate_started = time.perf_counter()
                        aggregate = participants[0].aggregate_federated_round_at_as(
                            round_id, participant_sids
                        )
                        aggregation_seconds = time.perf_counter() - aggregate_started
                        destinations = {
                            client.config.client_id: root / client.config.client_id /
                            f"global-round-{round_id}.safetensors"
                            for client in participants
                        }
                        # Bounded model reads remain a real distribution to every
                        # participant, but must not share the unbounded generic
                        # RPC fan-out. This prevents transient local TCP backlog
                        # saturation while retaining measured wall and aggregate
                        # transfer time. 有界模型读取仍会分发给每位参与者，但不能与
                        # 无上限通用 RPC 扇出共用并发度。这样可避免瞬时本地 TCP 队列
                        # 饱和，同时保留可测的墙钟和累计传输时间。
                        distribution_workers = _global_model_distribution_workers(
                            self.plan, case, len(participants)
                        )
                        distribution_phase = _parallel_phase(
                            participants,
                            distribution_workers,
                            lambda client: client.download_global_model_from_as(
                                round_id, destinations[client.config.client_id]
                            ),
                        )
                        distribution = {
                            "wall_seconds": distribution_phase["wall_seconds"],
                            "accumulated_seconds": distribution_phase["accumulated_seconds"],
                            "workers": distribution_workers,
                        }
                        initial_checkpoints = destinations
                    else:
                        aggregate = {"status": "not_run_in_simulated_mode"}
                    round_metrics.append({
                        "round_id": round_id,
                        "participant_sids": list(participant_sids),
                        "submitted_sids": list(round_submissions["submitted_sids"]),
                        "participant_count": len(participants),
                        "training_wall_seconds": round_training["wall_seconds"],
                        "training_accumulated_seconds": round_training["accumulated_seconds"],
                        "model_upload_accumulated_seconds": sum(
                            float(item["upload_elapsed_seconds"])
                            for item in round_submissions["client_metrics"].values()
                        ),
                        "model_upload_wall_seconds": round_submissions["wall_seconds"],
                        "fedavg_seconds": aggregation_seconds,
                        "global_distribution_wall_seconds": distribution["wall_seconds"],
                        "global_distribution_accumulated_seconds": distribution["accumulated_seconds"],
                        "global_distribution_workers": distribution.get("workers", 0),
                        "fedavg": aggregate,
                    })
                # Stop the paper-facing time measurement before read-only
                # metric collection. Collecting diagnostic JSON is not a
                # client protocol step. 在只读指标采集前停止论文使用的计时；
                # 诊断 JSON 的采集不属于客户端协议步骤。
                elapsed = (
                    training_completion_elapsed
                    if training_completion_elapsed is not None
                    else time.perf_counter() - measurement_started
                )
                as_metrics = _fetch_as_metrics(as_url)
                ks_metrics = _fetch_ks_metrics(ks_url)
                metadata = _metadata_sizes(root, clients, key_path, as_metrics, ks_metrics)
                communication = _communication_metrics(clients, ks_metrics)
                return {
                    "schema_version": "1.0",
                    "suite": case.suite,
                    "variable": case.variable,
                    "value": case.value,
                    "repetition": repetition,
                    "configuration": asdict(case),
                    "failure_victim_count": _failure_victim_count(case, len(clients))
                    if case.failure_phase is not None else 0,
                    "training_mode": self.plan.training_mode,
                    "oprf_suite": OPRF_SUITE_IDENTIFIER,
                    "service_mode": self.plan.service_mode,
                    "client_arrivals": arrival_metrics,
                    "arrival_schedule": arrival_schedule_contract({
                        client_id: float(timing["scheduled_delay_seconds"])
                        for client_id, timing in arrival_metrics.items()
                    }, late_arrival_seconds=self.plan.client_arrival_anchor_seconds),
                    "dataset": self._dataset_provenance(),
                    "total_completion_seconds": elapsed,
                    "measurement_boundary": (
                        "training_completed"
                        if training_completion_elapsed is not None
                        else "global_model_distributed"
                    ),
                    "dedup_wall_seconds": (
                        failed_dedup_seconds + dedup["wall_seconds"] + claims["wall_seconds"]
                    ),
                    # This companion metric includes the sampled time before
                    # the first client appears. ``dedup_wall_seconds`` is the
                    # first-request-to-last-completion campaign makespan and
                    # therefore includes gaps between arrivals; the separate
                    # active-interval metric below isolates occupied protocol
                    # time. 该配套指标包括首个客户端上线前的采样等待时间。
                    # ``dedup_wall_seconds`` 是从首个请求到最后完成的批次完成时间，
                    # 因此包含客户端上线之间的空档；下方独立的活跃区间指标用于隔离
                    # 实际被协议占用的时间。
                    "dedup_arrival_inclusive_wall_seconds": (
                        failed_dedup_seconds
                        + float(dedup.get("arrival_inclusive_wall_seconds", dedup["wall_seconds"]))
                        + claims["wall_seconds"]
                    ),
                    "dedup_active_interval_wall_seconds": (
                        failed_dedup_seconds
                        + float(dedup.get("active_interval_wall_seconds", dedup["wall_seconds"]))
                        + claims["wall_seconds"]
                    ),
                    "dedup_accumulated_seconds": (
                        failed_dedup_seconds
                        + dedup["accumulated_seconds"]
                        + claims["accumulated_seconds"]
                    ),
                    "training_wall_seconds": training["wall_seconds"],
                    "training_accumulated_seconds": training["accumulated_seconds"],
                    # ``without_history_scheduling`` needs a completed first
                    # round and a second scheduling decision; one round cannot
                    # exercise a prior-trainer policy. 历史调度消融必须先完成
                    # 首轮，再观察第二轮调度；单轮无法验证前训练者策略。
                    "federated_rounds": case_round_count,
                    "round_metrics": round_metrics,
                    "model_upload_accumulated_seconds": sum(
                        float(item["upload_elapsed_seconds"])
                        for item in submissions["client_metrics"].values()
                    ),
                    "model_upload_wall_seconds": submissions["model_upload_wall_seconds"],
                    "recovery_latency_seconds": recovery_latency,
                    "recovery_measurement_boundary": (
                        "no_as_state_transition_for_dedup_failure"
                        if case.failure_phase == "dedup"
                        else "disconnect_to_as_train_instruction"
                        if case.failure_phase == "training"
                        else None
                    ),
                    "fault_recoveries": fault_recoveries,
                    "fault_disconnect_events": (
                        dedup_disconnect_events
                        if dedup_disconnect_events
                        else (
                            list(training_failure_context.disconnect_events or [])
                            if training_failure_context is not None
                            else []
                        )
                    ),
                    "dynamic_join_protocol": dynamic_join_protocol,
                    "submitted_client_count": submissions["submitted_client_count"],
                    "ownership_retrain_count": submissions["ownership_retrain_count"],
                    "input_record_count": (
                        case.clients + case.joining_clients
                    ) * case.records_per_client,
                    "as_resources": (
                        sampler.finish()
                        if sampler is not None
                        else dict(as_metrics.get("server_resources", {
                            "status": "unavailable", "reason": "remote_as_did_not_report"
                        }))
                    ),
                    "metadata_bytes": metadata,
                    "communication_bytes": communication["bytes"],
                    "communication_timing_seconds": communication["timing_seconds"],
                    "metadata_counts": {
                        "as_task_count": int(as_metrics["task_count"]),
                        "as_owner_edge_count": int(as_metrics["owner_edge_count"]),
                        "as_total_sessions": int(as_metrics["total_sessions"]),
                        "as_online_sessions": int(as_metrics["online_sessions"]),
                        "logical_client_count": len(clients),
                        **communication["counts"],
                    },
                    "client_metrics": _client_metric_rows(
                        clients,
                        records_by_client,
                        queues,
                        training["client_metrics"],
                        submissions["client_metrics"],
                    ),
                }
            except Exception as error:
                diagnostic_paths: list[str] = []
                if as_process is not None:
                    try:
                        diagnostic_path = self._failed_case_log_path(case, repetition)
                        as_process.copy_log_to(diagnostic_path)
                        diagnostic_paths.append(str(diagnostic_path))
                        # Keep diagnostics attached to the original exception so
                        # the outer per-case policy can write it beside the
                        # traceback without altering the primary failure.
                        # 将诊断信息附加到原始异常，使外层逐用例策略能将其与 traceback
                        # 一同写出，且不改变主要失败。
                    except OSError:
                        # A full or unavailable result filesystem must not hide
                        # the actual protocol or training failure. 结果文件系统
                        # 写满或不可用时，不得掩盖真实的协议或训练失败。
                        pass
                try:
                    diagnostic_root = self._failed_case_log_path(case, repetition).parent
                    for log_path in root.rglob("*.log"):
                        copied_log_path = diagnostic_root / log_path.relative_to(root)
                        copied_log_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(log_path, copied_log_path)
                        diagnostic_paths.append(str(copied_log_path))
                except OSError:
                    # Retention is useful for post-mortem analysis, but must
                    # remain secondary to the original case exception. 诊断保留
                    # 有助于事后分析，但必须服从原始用例异常。
                    pass
                if diagnostic_paths:
                    error.dbtfl_diagnostic_paths = tuple(diagnostic_paths)
                raise
            finally:
                for client in clients:
                    try:
                        client.close()
                    except Exception:
                        # Cleanup is best effort: a secondary heartbeat or
                        # socket-close error must never replace the foreground
                        # case diagnostic. 清理尽力而为：次要心跳或套接字关闭错误
                        # 绝不能替换前台用例诊断。
                        pass
                if ks is not None:
                    try:
                        ks.close()
                    except Exception:
                        pass
                if as_process is not None:
                    try:
                        as_process.close()
                    except Exception:
                        pass

    def _failed_case_log_path(self, case: _Case, repetition: int) -> Path:
        """Return a deterministic, portable retained-log path for one case.

        返回一个用例确定且可移植的保留日志路径。
        """
        value = str(case.value).replace(".", "_")
        return (
            self.plan.output_directory.resolve()
            / "failed-case-diagnostics"
            / f"{case.suite}-{case.variable}-{value}-repeat-{repetition}-as.log"
        )

    def _reset_remote_as(self, case: _Case, heartbeat_timeout_seconds: float) -> None:
        """Clear a dedicated remote experiment AS before one independent case.

        在每个独立用例前清空专用远程实验 AS 并配置本用例租约。
        """
        assert self.plan.as_base_url is not None and self.plan.evaluation_reset_token is not None
        response = JsonHttpClient(
            self.plan.as_base_url,
            timeout_seconds=self.plan.rpc_timeout_seconds,
        ).send(
            AggregationServerPath.EVALUATION_RESET.value,
            WireMessage.create(
                AS_EVALUATION_RESET_REQUEST,
                {
                    "token": self.plan.evaluation_reset_token,
                    "backend_worker_count": case.backend_workers,
                    "heartbeat_interval_seconds": self.plan.heartbeat_interval_seconds,
                    "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
                },
            ),
        )
        if response.message_type != AS_EVALUATION_RESET_RESPONSE or response.payload != {
            "reset": True,
            "backend_worker_count": case.backend_workers,
            "heartbeat_interval_seconds": self.plan.heartbeat_interval_seconds,
            "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        }:
            raise RuntimeError("remote AS rejected evaluation reset / 远程 AS 拒绝实验重置")

    def _initial_heartbeat_timeout_seconds(self, case: _Case) -> float:
        """Return the long setup lease for every case, including fault cases.

        为所有用例（包括故障用例）返回长准备阶段租约。

        ``case`` remains an explicit argument so this boundary is directly
        regression-testable: a future failure-suite change must not silently
        move the short lease back into AS creation. 保留显式 ``case`` 参数使该
        边界可直接回归测试：后续故障套件改动不得悄然将短租约移回 AS 创建阶段。
        """
        del case
        return self.plan.heartbeat_timeout_seconds

    def _configure_training_failure_lease(
        self,
        clients: Sequence[ClientEntity],
        evaluation_control_token: str | None,
    ) -> None:
        """Synchronize live SIDs, then activate the short dropout-only lease.

        先同步在线 SID，再启用仅用于掉线的短租约。
        """
        if not clients or evaluation_control_token is None:
            raise RuntimeError(
                "training fault injection requires an evaluation control token / "
                "训练故障注入需要实验控制令牌"
            )
        # AS refreshes every online SID atomically with the lease change. A
        # separate ten-client heartbeat burst here would duplicate the normal
        # heartbeat workers and can saturate a four-worker local AS. AS 会在
        # 租约切换时原子刷新全部在线 SID；若在此额外发送十客户端心跳突发，会与常规
        # 心跳线程重复，并可能压满仅四个工作线程的本地 AS。
        response = JsonHttpClient(
            clients[0].config.as_base_url,
            timeout_seconds=self.plan.rpc_timeout_seconds,
        ).send(
            AggregationServerPath.EVALUATION_LEASE.value,
            WireMessage.create(
                AS_EVALUATION_LEASE_REQUEST,
                {
                    "token": evaluation_control_token,
                    "heartbeat_timeout_seconds": self.plan.failure_heartbeat_timeout_seconds,
                },
            ),
        )
        expected = {
            "heartbeat_interval_seconds": self.plan.heartbeat_interval_seconds,
            "heartbeat_timeout_seconds": self.plan.failure_heartbeat_timeout_seconds,
        }
        if response.message_type != AS_EVALUATION_LEASE_RESPONSE or response.payload != expected:
            raise RuntimeError(
                "AS rejected training-failure lease configuration / "
                "AS 拒绝训练故障租约配置"
            )

    def _ensure_prepared_data(self) -> None:
        """Load a unique real-text pool when real GPT evaluation is requested.

        在请求真实 GPT 评估时加载去重后的真实文本池。
        """
        if self.plan.training_mode == "simulated" or self._prepared_data_path is not None:
            return
        configured_path = self.plan.prepared_data_path
        source_path = (
            Path(configured_path)
            if configured_path is not None
            else PROJECT_ROOT / "results" / "prepared-haiku" / "train.jsonl"
        ).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                "prepared training data was not found / 未找到预处理训练数据："
                f"{source_path}\n"
                "Run / 请先执行：python scripts/prepare_text_dataset.py"
            )
        unique_records: list[PreparedRecord] = []
        seen_texts: set[bytes] = set()
        for record in iter_prepared_records(source_path):
            encoded_text = record.text.encode("utf-8")
            if encoded_text not in seen_texts:
                unique_records.append(record)
                seen_texts.add(encoded_text)
        if not unique_records:
            raise ValueError("prepared data has no usable text / 预处理数据没有可用文本")
        self._prepared_data_path = source_path
        self._prepared_records = tuple(unique_records)
        if self.plan.precomputed_oprf_directory is not None:
            if not precomputation_is_enabled(self.plan.precomputed_oprf_directory):
                raise ValueError(
                    "DwT-FL OPRF precomputation is disabled; enable it or omit --precomputed-oprf-directory / "
                    "DwT-FL OPRF 预计算已关闭；请启用它或省略 --precomputed-oprf-directory"
                )
            self._precomputed_oprf = load_precomputed_dataset(
                source_path, self.plan.precomputed_oprf_directory,
            )

    def _records_for_case(self, case: _Case, repetition: int) -> dict[str, list[str]]:
        """Assign either explicit test text or actual prepared records per client.

        为每个客户端分配显式测试文本或真实预处理记录。
        """
        if self.plan.training_mode == "simulated":
            return _synthetic_records_for_case(case, repetition)
        self._ensure_prepared_data()
        if self._precomputed_oprf is not None:
            if (
                case.clients != 10
                or case.records_per_client != 1024
                or case.duplicate_ratio != 0.30
                or repetition != 0
            ):
                raise ValueError(
                    "precomputed OPRF is fixed to the 10-client, 1024-record, r=0.3 first repetition configuration / "
                    "预计算 OPRF 固定为 10 客户端、1024 条记录、r=0.3 的首次重复配置"
                )
            records = {
                client_id: list(values)
                for client_id, values in self._precomputed_oprf.records_by_client.items()
            }
            if case.joining_clients:
                records.update(_allocate_precomputed_joiners(
                    self._prepared_records,
                    self._precomputed_oprf.records_by_client,
                    joining_clients=case.joining_clients,
                    duplicate_ratio=self.plan.join_duplicate_ratio,
                    seed=self.plan.seed,
                ))
            return records
        return _prepared_records_for_case(case, repetition, self._prepared_records,
                                          self.plan.join_duplicate_ratio)

    def _dataset_provenance(self) -> dict[str, object]:
        """Return transparent dataset provenance without disclosing plaintext.

        返回透明的数据集溯源信息，但不披露明文内容。
        """
        if self.plan.training_mode == "simulated":
            return {"kind": "explicit_protocol_simulation"}
        return {
            "kind": "prepared_local_text",
            "path": str(self._prepared_data_path),
            "unique_text_records": len(self._prepared_records),
        }

    def _create_clients(
        self, root: Path, as_url: str, ks_url: str, case: _Case
    ) -> list[ClientEntity]:
        """Create base clients without imposing a serial online order.

        创建基础客户端，但不强加串行上线顺序。
        """
        clients: list[ClientEntity] = []
        for position in range(case.clients):
            client = ClientEntity(ClientConfig(
                client_id=f"client-{position}", ks_base_url=ks_url, as_base_url=as_url,
                label_store_path=(
                    self._precomputed_oprf.label_store_path(f"client-{position}")
                    if self._precomputed_oprf is not None
                    else root / f"client-{position}-labels.json"
                ),
                timeout_seconds=self.plan.rpc_timeout_seconds,
                heartbeat_rpc_timeout_seconds=self.plan.heartbeat_rpc_timeout_seconds,
                oprf_batch_size=self.plan.oprf_batch_size,
                model_chunk_bytes=self.plan.model_chunk_bytes,
                traffic_recorder=TrafficRecorder(),
            ))
            clients.append(client)
        return clients

    def _arrival_delays(
        self,
        clients: Sequence[ClientEntity],
        case: _Case,
        repetition: int,
    ) -> dict[str, float]:
        """Produce the paired anchored schedule shared with the NDSS baseline.

        生成与 NDSS 基线共享的配对锚定调度。
        """
        return paired_arrival_delays(
            seed=self.plan.seed,
            case=case,
            repetition=repetition,
            client_ids=(client.config.client_id for client in clients),
            minimum_delay_seconds=self.plan.client_arrival_min_delay_seconds,
            maximum_delay_seconds=self.plan.client_arrival_max_delay_seconds,
            late_arrival_seconds=self.plan.client_arrival_anchor_seconds,
        )

    def _connect_clients_asynchronously(
        self,
        clients: list[ClientEntity],
        case: _Case,
        repetition: int,
    ) -> dict[str, dict[str, float]]:
        """Connect every client after its independently sampled arrival delay.

        在每个客户端各自采样的上线延迟后并发连接所有客户端。
        """
        delays = self._arrival_delays(clients, case, repetition)
        origin = time.perf_counter()

        def connect(client: ClientEntity) -> dict[str, float]:
            """Execute one independently scheduled client connection.

            执行一个独立调度的客户端连接。
            """
            delay = delays[client.config.client_id]
            if delay:
                time.sleep(delay)
            connect_started = time.perf_counter()
            client.connect_to_as()
            connected = time.perf_counter()
            return {
                "scheduled_delay_seconds": delay,
                "connected_after_seconds": connected - origin,
                "connect_elapsed_seconds": connected - connect_started,
            }

        phase = _parallel_phase(clients, len(clients), connect)
        return {
            client.config.client_id: dict(phase["values"][client.config.client_id][0])
            for client in clients
        }

    def _run_asynchronous_initial_protocols(
        self,
        clients: list[ClientEntity],
        records_by_client: Mapping[str, Sequence[str]],
        case: _Case,
        repetition: int,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, LocalTrainingQueues],
        dict[str, dict[str, float]],
    ]:
        """Run connect, label registration, and CAS as independent client flows.

        将连接、标签登记与 CAS 作为相互独立的客户端工作流执行。

        A client does not wait for the other clients to register before it
        submits its complete FP_i and immediately claims its own tasks. The
        method still waits for all futures before returning because the caller
        must construct the first fixed FedAvg roster from a complete observed
        population. 客户端不会等待其他客户端完成登记；其提交完整 FP_i 后立即认领
        自己的任务。该方法返回前仍等待所有 future，因为调用方必须根据完整的已观测
        参与者集合构造首个固定 FedAvg 名册。
        """
        delays = self._arrival_delays(clients, case, repetition)
        origin = time.perf_counter()

        def workflow(client: ClientEntity) -> tuple[
            list[str], list[TaskClaimDecision], LocalTrainingQueues, dict[str, float]
        ]:
            """Execute one complete first-round client protocol flow.

            执行一个完整的首轮客户端协议流程。
            """
            identifier = client.config.client_id
            delay = delays[identifier]
            if delay:
                time.sleep(delay)
            connect_started = time.perf_counter()
            client.connect_to_as()
            connected = time.perf_counter()
            registration_started = connected
            labels = client.register_records_with_as(
                records_by_client[identifier],
                created_round=1,
                # The connected client already owns a periodic heartbeat
                # worker and the normal setup lease is long. Registration and
                # CAS remain independent parallel protocol requests; inserting
                # a full client-wide foreground heartbeat wave before each one
                # is neither required by the paper nor part of the metric.
                # 已连接客户端已有周期心跳工作线程，且正常准备租约足够长。登记和
                # CAS 仍是独立的并行协议请求；在每一步前插入全客户端前台心跳既非
                # 论文要求，也不属于被测指标。
                refresh_lease=False,
            )
            registration_finished = time.perf_counter()
            claim_started = registration_finished
            decisions = client.claim_registered_labels_at_as(
                labels,
                refresh_lease=False,
            )
            claim_finished = time.perf_counter()
            return labels, decisions, client.route_claim_decisions(decisions), {
                "scheduled_delay_seconds": delay,
                "connected_after_seconds": connected - origin,
                "connect_elapsed_seconds": connected - connect_started,
                "first_protocol_started_after_seconds": registration_started - origin,
                "registration_elapsed_seconds": registration_finished - registration_started,
                "claim_elapsed_seconds": claim_finished - claim_started,
                "protocol_completed_after_seconds": claim_finished - origin,
            }

        phase = _parallel_phase(clients, len(clients), workflow)
        registration_values: dict[str, tuple[list[str], float]] = {}
        claim_values: dict[str, tuple[list[TaskClaimDecision], float]] = {}
        queues: dict[str, LocalTrainingQueues] = {}
        arrivals: dict[str, dict[str, float]] = {}
        for client in clients:
            labels, decisions, queue, timing = phase["values"][client.config.client_id][0]
            identifier = client.config.client_id
            registration_values[identifier] = (labels, timing["registration_elapsed_seconds"])
            claim_values[identifier] = (decisions, timing["claim_elapsed_seconds"])
            queues[identifier] = queue
            arrivals[identifier] = timing
        first_start = min(
            timing["first_protocol_started_after_seconds"] for timing in arrivals.values()
        )
        final_completion = max(
            timing["protocol_completed_after_seconds"] for timing in arrivals.values()
        )
        active_interval_wall_seconds = _merged_interval_seconds(
            (
                float(timing["first_protocol_started_after_seconds"]),
                float(timing["protocol_completed_after_seconds"]),
            )
            for timing in arrivals.values()
        )
        return (
            {
                "values": registration_values,
                # This is the asynchronous campaign makespan: it includes
                # delayed clients that have not yet started after the first
                # request. It is the correct wall time for the real arrival
                # scenario, but not a pure service-capacity denominator.
                # 这是异步批次完成时间：首个请求后尚未上线的延迟客户端也包含在内。
                # 它是实际到达场景正确的墙钟时间，但不是纯服务能力的分母。
                "wall_seconds": final_completion - first_start,
                # Merge only intervals in which a client is executing label
                # registration plus CAS. The union excludes idle gaps caused
                # solely by the chosen arrival schedule. 仅合并客户端执行标签
                # 登记与 CAS 的区间；并集排除仅由指定上线调度产生的空闲间隔。
                "active_interval_wall_seconds": active_interval_wall_seconds,
                "arrival_inclusive_wall_seconds": final_completion,
                "accumulated_seconds": sum(
                    timing["registration_elapsed_seconds"] + timing["claim_elapsed_seconds"]
                    for timing in arrivals.values()
                ),
            },
            {"values": claim_values, "wall_seconds": 0.0, "accumulated_seconds": 0.0},
            queues,
            arrivals,
        )

    def _run_asynchronous_dedup_dropout_protocols(
        self,
        clients: list[ClientEntity],
        records_by_client: Mapping[str, Sequence[str]],
        case: _Case,
        repetition: int,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, LocalTrainingQueues],
        dict[str, dict[str, float]],
        tuple[ClientEntity, ...],
    ]:
        """Run per-client asynchronous arrival with an OPRF-stage dropout.

        运行逐客户端异步上线流程，并在 OPRF 阶段注入掉线。

        A selected victim independently arrives, completes only its private
        OPRF work, and then loses connectivity before it mutates AS state.
        Every survivor retains its own independent arrival schedule and
        completes registration/CAS immediately; no survivor waits for the
        victim or for a global registration barrier. 被选中的受害者独立上线、仅完成
        私有 OPRF 工作，随后在改变 AS 状态前失联；每个存活客户端保持各自独立上线
        调度并立即完成登记/CAS，不会等待受害者或全局登记屏障。
        """
        victim_count = _failure_victim_count(case, len(clients))
        victims = tuple(clients[:victim_count])
        victim_ids = {client.config.client_id for client in victims}
        disconnect_schedule = {
            str(item["client_id"]): item
            for item in staggered_disconnect_schedule(
                victim_ids,
                initial_delay_seconds=self.plan.failure_disconnect_initial_delay_seconds,
                interval_seconds=self.plan.failure_disconnect_interval_seconds,
            )
        }
        delays = self._arrival_delays(clients, case, repetition)
        origin = time.perf_counter()

        def workflow(client: ClientEntity) -> tuple[
            list[str] | None,
            list[TaskClaimDecision] | None,
            LocalTrainingQueues | None,
            dict[str, float],
        ]:
            """Perform either one survivor flow or the victim's pre-AS flow.

            执行一个存活客户端流程，或受害者的 AS 前流程。
            """
            identifier = client.config.client_id
            delay = delays[identifier]
            if delay:
                time.sleep(delay)
            connect_started = time.perf_counter()
            client.connect_to_as()
            connected = time.perf_counter()
            registration_started = connected
            if identifier in victim_ids:
                # Deliberately persist the protected labels before disconnect.
                # The later rejoin proves OPRF caching and does not synthesize
                # an alternative dataset. 在断联前刻意持久化保护标签；随后重连
                # 证明 OPRF 缓存被复用，而非构造替代数据集。
                client.generate_protected_labels(records_by_client[identifier])
                planned = disconnect_schedule[identifier]
                # This wait is deliberately after a successful AS connection.
                # It models a connected client that later fails, rather than a
                # participant that never joined. 该等待刻意置于成功 AS 连接之后，
                # 用于模拟已连接后才发生故障的客户端，而非从未加入的参与者。
                time.sleep(float(planned["scheduled_disconnect_after_seconds"]))
                client.close()
                finished = time.perf_counter()
                return None, None, None, {
                    "scheduled_delay_seconds": delay,
                    "connected_after_seconds": connected - origin,
                    "connect_elapsed_seconds": connected - connect_started,
                    "first_protocol_started_after_seconds": registration_started - origin,
                    "registration_elapsed_seconds": finished - registration_started,
                    "claim_elapsed_seconds": 0.0,
                    "protocol_completed_after_seconds": finished - origin,
                    "disconnect_ordinal": float(planned["ordinal"]),
                    "scheduled_disconnect_after_connection_seconds": float(
                        planned["scheduled_disconnect_after_seconds"]
                    ),
                    "actual_disconnect_after_connection_seconds": finished - connected,
                }
            labels = client.register_records_with_as(
                records_by_client[identifier],
                created_round=1,
                # See the initial protocol flow: the periodic worker owns
                # liveness while this client-parallel protocol flow owns
                # registration and CAS.
                # 见初始协议流程：周期工作线程负责保活，客户端并行协议流程负责
                # 登记和 CAS。
                refresh_lease=False,
            )
            registration_finished = time.perf_counter()
            decisions = client.claim_registered_labels_at_as(
                labels,
                refresh_lease=False,
            )
            claim_finished = time.perf_counter()
            return labels, decisions, client.route_claim_decisions(decisions), {
                "scheduled_delay_seconds": delay,
                "connected_after_seconds": connected - origin,
                "connect_elapsed_seconds": connected - connect_started,
                "first_protocol_started_after_seconds": registration_started - origin,
                "registration_elapsed_seconds": registration_finished - registration_started,
                "claim_elapsed_seconds": claim_finished - registration_finished,
                "protocol_completed_after_seconds": claim_finished - origin,
            }

        phase = _parallel_phase(clients, len(clients), workflow)
        registration_values: dict[str, tuple[list[str], float]] = {}
        claim_values: dict[str, tuple[list[TaskClaimDecision], float]] = {}
        queues: dict[str, LocalTrainingQueues] = {}
        arrivals: dict[str, dict[str, float]] = {}
        for client in clients:
            labels, decisions, queue, timing = phase["values"][client.config.client_id][0]
            identifier = client.config.client_id
            arrivals[identifier] = timing
            if identifier in victim_ids:
                continue
            if labels is None or decisions is None or queue is None:
                raise RuntimeError(
                    "surviving dedup client returned no protocol result / "
                    "存活去重客户端未返回协议结果"
                )
            registration_values[identifier] = (labels, timing["registration_elapsed_seconds"])
            claim_values[identifier] = (decisions, timing["claim_elapsed_seconds"])
            queues[identifier] = queue
        survivor_timings = [
            timing for identifier, timing in arrivals.items() if identifier not in victim_ids
        ]
        if not survivor_timings:
            raise RuntimeError(
                "dedup-fault case retained no online protocol client / "
                "去重故障用例未保留在线协议客户端"
            )
        first_start = min(
            timing["first_protocol_started_after_seconds"] for timing in survivor_timings
        )
        final_completion = max(
            timing["protocol_completed_after_seconds"] for timing in survivor_timings
        )
        return (
            {
                "values": registration_values,
                "wall_seconds": final_completion - first_start,
                "active_interval_wall_seconds": _merged_interval_seconds(
                    (
                        timing["first_protocol_started_after_seconds"],
                        timing["protocol_completed_after_seconds"],
                    )
                    for timing in survivor_timings
                ),
                "arrival_inclusive_wall_seconds": final_completion,
                "accumulated_seconds": sum(
                    timing["registration_elapsed_seconds"] + timing["claim_elapsed_seconds"]
                    for timing in survivor_timings
                ),
            },
            {"values": claim_values, "wall_seconds": 0.0, "accumulated_seconds": 0.0},
            queues,
            arrivals,
            victims,
        )

    def _add_dynamic_clients(
        self,
        root: Path,
        as_url: str,
        ks_url: str,
        case: _Case,
        clients: list[ClientEntity],
        records: dict[str, list[str]],
        repetition: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
        """Join clients after base CAS work without restarting existing deduplication.

        在基础 CAS 工作之后加入客户端，且不重启既有去重流程。
        """
        if case.joining_clients == 0:
            return None
        joiners: list[ClientEntity] = []
        for offset in range(case.joining_clients):
            position = case.clients + offset
            identifier = f"client-{position}"
            client = ClientEntity(ClientConfig(
                client_id=identifier, ks_base_url=ks_url, as_base_url=as_url,
                label_store_path=root / f"{identifier}-labels.json",
                timeout_seconds=self.plan.rpc_timeout_seconds,
                heartbeat_rpc_timeout_seconds=self.plan.heartbeat_rpc_timeout_seconds,
                oprf_batch_size=self.plan.oprf_batch_size,
                model_chunk_bytes=self.plan.model_chunk_bytes,
                traffic_recorder=TrafficRecorder(),
            ))
            clients.append(client)
            joiners.append(client)
        join_delays = self._arrival_delays(joiners, case, repetition)

        def connect_joiner(client: ClientEntity) -> Any:
            """Delay and connect one late client without serial ordering.

            延迟并连接一名后加入客户端，不引入串行顺序。
            """
            delay = join_delays[client.config.client_id]
            if delay:
                time.sleep(delay)
            return client.connect_to_as()

        _parallel_phase(joiners, len(joiners), connect_joiner)
        # Keep the joiner's OPRF phase independently observable. Calling
        # ``register_records_with_as`` would blend label evaluation with AS
        # registration, which cannot answer the paper's separate new-client
        # OPRF-time metric. 显式拆出加入客户端的 OPRF 阶段；若直接调用
        # ``register_records_with_as``，标签求值会与 AS 登记混合，无法回答论文
        # 所要求的“新加入客户端 OPRF 时间”指标。
        oprf = _parallel_phase(
            joiners,
            len(joiners),
            lambda client: client.generate_protected_labels(
                records[client.config.client_id]
            ),
        )
        def register_generated_labels(client: ClientEntity) -> list[Any]:
            """Register the just-computed labels without a second OPRF call.

            登记刚计算的标签，不再执行第二次 OPRF。
            """
            labels = list(oprf["values"][client.config.client_id][0])
            registrations = client.register_protected_labels_with_as(
                tuple(dict.fromkeys(labels)), created_round=1,
            )
            by_label = {item.protected_label: item for item in registrations}
            return [by_label[label] for label in labels]

        dedup = _parallel_phase(joiners, len(joiners), register_generated_labels)
        dedup["oprf_wall_seconds"] = oprf["wall_seconds"]
        dedup["oprf_accumulated_seconds"] = oprf["accumulated_seconds"]
        claims = _parallel_phase(
            joiners,
            len(joiners),
            lambda client: client.claim_registered_labels_at_as(
                dedup["values"][client.config.client_id][0],
                refresh_lease=False,
            ),
        )
        queues = {
            client.config.client_id: client.route_claim_decisions(
                claims["values"][client.config.client_id][0]
            )
            for client in joiners
        }
        return dedup, claims, queues

    def _prepare_dynamic_join(self, case: _Case) -> _DynamicJoinContext | None:
        """Allocate synchronization state for a non-blocking late join.

        为不阻塞既有训练的动态加入分配同步状态。
        """
        if case.joining_clients == 0:
            return None
        return _DynamicJoinContext(
            started=threading.Event(),
            completed=threading.Event(),
            lock=threading.Lock(),
        )

    def _start_dynamic_join_when_training_starts(
        self,
        context: _DynamicJoinContext,
        job: ClientTrainingJob,
        root: Path,
        as_url: str,
        ks_url: str,
        case: _Case,
        clients: list[ClientEntity],
        records: dict[str, list[str]],
        repetition: int,
    ) -> None:
        """Launch one join flow after an existing client begins local training.

        在既有客户端开始本地训练后启动一次加入流程。

        The callback fires from the scheduler just before an already-admitted
        client starts training. It therefore proves that joining neither waits
        for global re-deduplication nor prevents established hot queues from
        making progress. 调度器会在已加入客户端开始训练前调用该回调，因此该流程证明
        新加入不等待全局重新去重，也不会阻止既有热队列继续执行。
        """
        del job
        with context.lock:
            if context.thread is not None:
                return
            context.started.set()

            def join_flow() -> None:
                """Execute the late-join registration/CAS independently.

                独立执行后加入客户端的登记/CAS。
                """
                try:
                    context.result = self._add_dynamic_clients(
                        root, as_url, ks_url, case, clients, records, repetition
                    )
                except BaseException as error:  # Surface a background failure to the case.
                    # A join failure is evidence, not a reason to silently omit
                    # the joiner from a paper result. 加入失败是实验事实，不能静默
                    # 省略新客户端后仍输出论文结果。
                    context.error = error
                finally:
                    context.completed.set()

            context.thread = threading.Thread(
                target=join_flow,
                name="dbtfl-dynamic-join",
                daemon=True,
            )
            context.thread.start()

    @staticmethod
    def _await_dynamic_join(
        context: _DynamicJoinContext,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Return the completed late-join flow or raise its actual failure.

        返回完成的后加入流程，或抛出其真实失败。
        """
        if not context.started.is_set() or context.thread is None:
            raise RuntimeError(
                "dynamic join never began after base training started / "
                "基础训练开始后动态加入从未启动"
            )
        context.thread.join()
        if context.error is not None:
            raise RuntimeError("dynamic join failed / 动态加入失败") from context.error
        if context.result is None:
            raise RuntimeError(
                "dynamic join returned no protocol result / 动态加入未返回协议结果"
            )
        return context.result

    def _on_initial_training_job_start(
        self,
        job: ClientTrainingJob,
        training_failure: _TrainingFailureContext | None,
        dynamic_join: _DynamicJoinContext | None,
        root: Path,
        as_url: str,
        ks_url: str,
        case: _Case,
        clients: list[ClientEntity],
        records: dict[str, list[str]],
        repetition: int,
    ) -> None:
        """Start independent fault and join flows at the training boundary.

        在训练边界启动独立的故障与加入流程。
        """
        if training_failure is not None:
            self._start_training_failure_when_victim_starts(training_failure, job)
        if dynamic_join is not None:
            self._start_dynamic_join_when_training_starts(
                dynamic_join, job, root, as_url, ks_url, case, clients, records,
                repetition,
            )

    def _inject_dedup_dropout(
        self,
        case: _Case,
        clients: list[ClientEntity],
        records: dict[str, list[str]],
    ) -> tuple[float, tuple[ClientEntity, ...]]:
        """Stop victims after OPRF and defer AS work until their later rejoin.

        在 OPRF 后停止受害客户端，并将其 AS 请求推迟到随后重连时。

        A deduplication-phase dropout never creates a task entry or a PENDING
        owner for the victim. The remaining clients therefore continue their
        full registration and CAS workflows independently. This is the paper's
        ``EMPTY``-state recovery, rather than a client-side transport retry.
        去重阶段掉线不会为受害客户端创建任务表项或 PENDING 所有者；其余客户端会
        独立完成完整的登记和 CAS 工作流。这是论文规定的 ``EMPTY`` 状态恢复，而
        不是客户端侧传输重试。
        """
        if case.failure_phase != "dedup" or case.value == 0.0 or not clients:
            return 0.0, ()
        started = time.perf_counter()
        victims = tuple(clients[:_failure_victim_count(case, len(clients))])
        for victim in victims:
            # Generate OPRF labels before failure so the replacement proves that
            # it resumes the private local correspondence rather than generating
            # synthetic data. 在故障前生成 OPRF 标签，使随后重连证明其恢复的是私有
            # 本地对应关系，而非生成合成数据。
            victim.generate_protected_labels(records[victim.config.client_id])
            # ``close`` stops only the local heartbeat worker. It deliberately
            # sends no final heartbeat or AS request, so the client performs no
            # index update while it is offline. ``close`` 只停止本地心跳工作线程，
            # 刻意不发送最终心跳或 AS 请求，因此该客户端离线时不会更新索引。
            victim.close()
        return time.perf_counter() - started, victims

    def _rejoin_dedup_dropouts(
        self,
        victims: Sequence[ClientEntity],
        records_by_client: Mapping[str, Sequence[str]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, LocalTrainingQueues]]:
        """Resume only dropped clients' unfinished deduplication requests.

        仅恢复掉线客户端未完成的去重请求。

        Reconnection uses the stable client ID and the pre-failure protected
        label store. AS then returns DEDUP for labels already PENDING or
        COMMITTED elsewhere, while an EMPTY label is still eligible for CAS as
        required by the manuscript. 重连使用稳定客户端标识和故障前受保护标签
        存储。AS 对已被其他客户端 PENDING 或 COMMITTED 的标签返回 DEDUP；仍为
        EMPTY 的标签依旧可按论文规定参加 CAS。
        """
        # Rejoining clients reconnect independently, just like first arrivals;
        # serial reconnection would manufacture a client-order effect. 重连客户端
        # 与首次上线一样独立连接；串行重连会人为制造客户端顺序效应。
        _parallel_phase(victims, len(victims), lambda client: client.connect_to_as())
        registrations = _parallel_phase(
            victims,
            len(victims),
            lambda client: client.register_records_with_as(
                records_by_client[client.config.client_id],
                created_round=1,
                # ``connect_to_as`` has just established this SID and started
                # its periodic heartbeat. Do not create a redundant foreground
                # control burst before the rejoin registration; the registration
                # request itself validates the live SID.
                # ``connect_to_as`` 刚建立该 SID 并启动其周期心跳。重连登记前
                # 不再制造冗余前台控制突发；登记请求本身会验证在线 SID。
                refresh_lease=False,
            ),
        )
        claims = _parallel_phase(
            victims,
            len(victims),
            lambda client: client.claim_registered_labels_at_as(
                registrations["values"][client.config.client_id][0],
                # The immediately preceding registration is the authoritative
                # liveness boundary. Preserve parallel CAS contention without
                # adding a second all-rejoiner heartbeat wave.
                # 紧邻的登记是权威保活边界。保留并行 CAS 竞争，但不再增加第二轮
                # 全体重连客户端心跳。
                refresh_lease=False,
            ),
        )
        queues = {
            client.config.client_id: client.route_claim_decisions(
                claims["values"][client.config.client_id][0]
            )
            for client in victims
        }
        return registrations, claims, queues

    def _inject_failure_if_requested(
        self,
        case: _Case,
        clients: list[ClientEntity],
        claims: dict[str, Any],
        queues: dict[str, Any],
        *,
        evaluation_control_token: str | None,
    ) -> tuple[float | None, list[dict[str, object]]]:
        """Trigger a configured dropout fraction and measure every takeover.

        触发配置比例的训练掉线，并测量每次接管延迟。
        """
        if case.failure_phase != "training" or case.value == 0.0 or len(clients) < 2:
            return None, []
        self._configure_training_failure_lease(clients, evaluation_control_token)
        victim_count = _failure_victim_count(case, len(clients))
        victims = _select_training_failure_victims(clients, claims, victim_count)
        victim_identifiers = {client.config.client_id for client in victims}
        survivors = [
            client for client in clients
            if client.config.client_id not in victim_identifiers
        ]
        if not survivors:
            raise RuntimeError(
                "training-fault case must retain at least one live client / "
                "训练故障用例必须至少保留一个在线客户端"
            )
        victim_task_ids = {
            victim.config.client_id: {
                decision.task_id
                for decision in claims["values"][victim.config.client_id][0]
                if decision.operation == "TRAIN"
            }
            for victim in victims
        }
        victim_sids_before_disconnect = {
            victim.config.client_id: (
                victim.as_session.sid if victim.as_session is not None else -1
            )
            for victim in victims
        }
        survivor_known_task_ids = {
            decision.task_id
            for survivor in survivors
            for decision in claims["values"][survivor.config.client_id][0]
        }
        # Only a duplicated task has an online alternate owner. An exclusive
        # task cannot have a takeover latency; it is reclaimed after the
        # original client reconnects and must not consume a three-lease wait.
        # 只有重复任务才存在在线备用所有者。独有任务没有接管延迟；原客户端重连后
        # 会重新认领它，因此绝不能消耗三个租约的等待时间。
        recovery_target_ids = {
            victim_id
            for victim_id, task_ids in victim_task_ids.items()
            if task_ids.intersection(survivor_known_task_ids)
        }
        # Model a simultaneous population dropout. Stopping victims one at a
        # time permits a later victim to take an earlier victim's task before
        # it is itself disconnected, which is not the requested fault rate.
        # 模拟同一批客户端同时掉线。逐个停止会使后续受害者在其自身掉线前抢到先前
        # 受害者的任务，这不符合指定的故障比例。
        for victim in victims:
            victim._stop_heartbeat_worker()
        before = time.perf_counter()
        deadline = before + self.plan.failure_heartbeat_timeout_seconds * 3
        recovered_by_victim: dict[str, ClientEntity] = {}
        latest_metrics: dict[str, Any] | None = None
        while (
            time.perf_counter() < deadline
            and len(recovered_by_victim) < len(recovery_target_ids)
        ):
            # Every surviving owner receives an independent heartbeat. The AS,
            # not the evaluator, performs the CAS and selects the valid owner.
            # 每个存活所有者独立发送心跳；由 AS 而非评估器执行 CAS 并选择合法所有者。
            _parallel_phase(survivors, len(survivors), lambda client: client.send_as_heartbeat())
            latest_metrics = _fetch_as_metrics(survivors[0].config.as_base_url)
            for victim in victims:
                victim_id = victim.config.client_id
                if victim_id not in recovery_target_ids or victim_id in recovered_by_victim:
                    continue
                for successor in survivors:
                    successor_task_ids = {
                        instruction.task_id
                        for instruction in successor.last_round_instructions
                        if instruction.operation == "TRAIN"
                    }
                    if successor_task_ids.intersection(victim_task_ids[victim_id]):
                        recovered_by_victim[victim_id] = successor
                        break
            if len(recovered_by_victim) < len(victims):
                time.sleep(self.plan.heartbeat_interval_seconds)

        # Reconnection happens after all live owners had the same opportunity
        # to compete. The following full synchronization reconstructs claims
        # and queues from AS instructions, so this method never combines stale
        # local decisions with recovered ones. 所有存活所有者公平竞争后才重连。
        # 随后的完整同步会根据 AS 指令重建认领和队列，因此此方法绝不混合陈旧本地
        # 决策与恢复后的决策。
        for victim in victims:
            victim.connect_to_as()
            victim.send_as_heartbeat()

        recoveries: list[dict[str, object]] = []
        detection_times = (
            latest_metrics.get("offline_detection_by_sid_seconds", {})
            if latest_metrics is not None else {}
        )
        takeover_times = (
            latest_metrics.get("recovery_takeover_by_sid_seconds", {})
            if latest_metrics is not None else {}
        )
        for victim in victims:
            successor = recovered_by_victim.get(victim.config.client_id)
            victim_sid = victim.as_session.sid if victim.as_session is not None else -1
            successor_sid = successor.as_session.sid if successor is not None and successor.as_session else -1
            detected = detection_times.get(str(victim_sid))
            taken = takeover_times.get(str(successor_sid))
            recoveries.append({
                "victim_client_id": victim.config.client_id,
                "successor_client_id": None if successor is None else successor.config.client_id,
                "recovery_latency_seconds": (
                    max(0.0, float(taken) - float(detected))
                    if detected is not None and taken is not None else None
                ),
            })
        latencies = [
            float(item["recovery_latency_seconds"])
            for item in recoveries if item["recovery_latency_seconds"] is not None
        ]
        return (max(latencies) if latencies else None), recoveries

    def _prepare_training_failure(
        self,
        case: _Case,
        clients: Sequence[ClientEntity],
        claims: Mapping[str, Any],
        *,
        evaluation_control_token: str | None,
    ) -> _TrainingFailureContext | None:
        """Prepare a real training-stage failure without stopping it yet.

        准备真实训练阶段故障，但暂不使客户端掉线。

        CAS and normal registration have already completed when this method is
        called. The short lease is therefore scoped to the deliberately faulty
        training period. Crucially, heartbeats continue until a selected
        victim's trainer begins, rather than making a CAS-successful client
        appear to have failed before it ever trained. 调用此方法时 CAS 与正常
        登记已经完成，因此短租约只作用于刻意注入故障的训练期间。关键是，在选中
        受害者的训练器真正开始前心跳仍会继续，不能把刚成功 CAS 的客户端误判为尚未
        训练即掉线。
        """
        if case.failure_phase != "training" or case.value == 0.0 or len(clients) < 2:
            return None
        self._configure_training_failure_lease(clients, evaluation_control_token)
        victims = tuple(_select_training_failure_victims(
            clients,
            claims,
            _failure_victim_count(case, len(clients)),
        ))
        victim_identifiers = {client.config.client_id for client in victims}
        survivors = tuple(
            client for client in clients if client.config.client_id not in victim_identifiers
        )
        if not survivors:
            raise RuntimeError(
                "training-fault case must retain at least one live client / "
                "训练故障用例必须至少保留一个在线客户端"
            )
        victim_task_ids = {
            victim.config.client_id: {
                decision.task_id
                for decision in claims["values"][victim.config.client_id][0]
                if decision.operation == "TRAIN"
            }
            for victim in victims
        }
        # Persist the pre-failure SID so each actual disconnect and subsequent
        # AS takeover can be audited as one protocol event. 保存断连前的 SID，
        # 使每次真实断连及其后的 AS 接管可审计为同一协议事件。
        victim_sids_before_disconnect = {
            victim.config.client_id: (
                victim.as_session.sid if victim.as_session is not None else -1
            )
            for victim in victims
        }
        survivor_known_task_ids = {
            decision.task_id
            for survivor in survivors
            for decision in claims["values"][survivor.config.client_id][0]
        }
        recovery_target_ids = {
            victim_id
            for victim_id, task_ids in victim_task_ids.items()
            if task_ids.intersection(survivor_known_task_ids)
        }
        return _TrainingFailureContext(
            victims=victims,
            survivors=survivors,
            victim_task_ids=victim_task_ids,
            victim_sids_before_disconnect=victim_sids_before_disconnect,
            disconnect_started_at_by_client={},
            recovery_target_ids=recovery_target_ids,
            started=threading.Event(),
            completed=threading.Event(),
            lock=threading.Lock(),
        )

    def _start_training_failure_when_victim_starts(
        self,
        context: _TrainingFailureContext,
        job: ClientTrainingJob,
    ) -> None:
        """Start the failure monitor only after an affected trainer starts.

        仅在受影响训练器启动后开始故障监控。
        """
        victim_identifiers = {client.config.client_id for client in context.victims}
        if job.client_id not in victim_identifiers:
            return
        with context.lock:
            if context.thread is not None:
                return
            context.started.set()
            context.thread = threading.Thread(
                target=self._run_training_failure_monitor,
                args=(context,),
                name="dbtfl-training-failure-monitor",
                daemon=True,
            )
            context.thread.start()

    def _run_training_failure_monitor(self, context: _TrainingFailureContext) -> None:
        """Detect each dropout and dispatch only to live duplicate holders.

        检测每个掉线事件，并仅向在线重复数据持有者下发接管指令。

        A deliberately faulted client does not reconnect in the measured
        round.  If all holders of a released task are offline, AS leaves the
        task ``EMPTY`` for the next round. This is not a failed recovery: it
        is the explicitly selected next-round deferral policy. 被刻意注入故障
        的客户端不会在计量轮次内重新连接。若已释放任务的所有持有者均离线，AS 将
        任务保留为 ``EMPTY`` 并留到下一轮；这不是恢复失败，而是明确选择的延后策略。
        """
        try:
            # Every fault client is already connected.  Arm one common clock
            # and stop them at fixed, recorded offsets instead of collapsing
            # the population into one simultaneous failure. 每个故障客户端均已
            # 连接；以一个共同计时起点和固定、可记录偏移依次停止，而不是把群体压缩成
            # 一次同时故障。
            schedule = staggered_disconnect_schedule(
                (victim.config.client_id for victim in context.victims),
                initial_delay_seconds=self.plan.failure_disconnect_initial_delay_seconds,
                interval_seconds=self.plan.failure_disconnect_interval_seconds,
            )
            victims_by_id = {victim.config.client_id: victim for victim in context.victims}
            events: list[dict[str, object]] = []

            # ``configure_evaluation_lease`` has already refreshed every
            # online session atomically. Each surviving client's existing
            # periodic heartbeat now maintains that lease. Do not add a second
            # all-client foreground refresher here: with a 0.1 s interval it
            # doubles control traffic and can itself starve the AS listener.
            # ``configure_evaluation_lease`` 已原子刷新所有在线会话；存活客户端
            # 既有的周期心跳会维持该租约。这里不能再添加一次全客户端前台刷新：在
            # 0.1 秒周期下它会使控制流量翻倍，并可能反过来挤占 AS 监听器。
            before = time.perf_counter()

            # Execute the complete disconnect schedule before synchronously
            # polling survivors. A slow heartbeat RPC must never consume the
            # scheduled interval and silently turn a k-client fault into a
            # smaller failure. 在同步轮询存活客户端之前执行完整断连计划；缓慢的
            # 心跳 RPC 绝不能吞掉计划间隔，并把 k 客户端故障悄然变成较小故障。
            for planned in schedule:
                due = before + float(planned["scheduled_disconnect_after_seconds"])
                remaining = due - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                victim = victims_by_id[str(planned["client_id"])]
                victim._stop_heartbeat_worker()
                disconnected_at = time.perf_counter()
                context.disconnect_started_at_by_client[victim.config.client_id] = disconnected_at
                events.append({
                    **planned,
                    "actual_disconnect_after_seconds": disconnected_at - before,
                    "sid_before_disconnect": context.victim_sids_before_disconnect[
                        victim.config.client_id
                    ],
                })
            if len(events) != len(schedule):
                raise RuntimeError(
                    "training-failure disconnect schedule was incomplete / "
                    "训练故障断连计划未完整执行"
                )

            # The recovery deadline starts only after the final planned fault,
            # so it bounds state recovery without truncating a requested fault
            # population. 恢复截止时间仅在最后一个计划故障后开始，因此它限制状态
            # 恢复耗时，但不会截断请求的故障客户端集合。
            deadline = time.perf_counter() + self.plan.failure_heartbeat_timeout_seconds * 3
            recovered_by_task: dict[int, tuple[ClientEntity, float]] = {}
            latest_metrics: dict[str, Any] | None = None
            # Continue through the complete short-lease window even when no
            # task is recoverable. This gives AS one authoritative opportunity
            # to release every offline PENDING task to EMPTY. 即使没有可接管
            # 任务，也必须完整经过短租约窗口，以便 AS 有一次权威机会将所有离线
            # PENDING 任务释放为 EMPTY。
            while time.perf_counter() < deadline:
                # Every surviving SID concurrently reads its own response-local
                # snapshot. This is the paper's parallel recovery dispatch:
                # AS independently decides each live holder's newly released
                # duplicate tasks. The control route is isolated from data
                # requests and each SID has a single-flight guard, so parallel
                # dispatch does not duplicate a client's heartbeat.
                # 每个存活 SID 并行读取各自响应本地的快照。这是论文中的并行恢复
                # 下发：AS 独立决定每个在线持有者接管的新释放重复任务。控制路由已与
                # 数据请求隔离，且每个 SID 具有单飞保护，因此并行下发不会重复同一
                # 客户端的心跳。
                recovery_heartbeat = _heartbeat_instruction_snapshots(context.survivors)
                observed_at = time.perf_counter()
                for successor in context.survivors:
                    instructions = recovery_heartbeat[successor.config.client_id]
                    for instruction in instructions:
                        if instruction.operation == "TRAIN":
                            recovered_by_task.setdefault(
                                instruction.task_id, (successor, observed_at)
                            )
                time.sleep(self.plan.heartbeat_interval_seconds)
            # Recovery correctness and latency are determined by AS-issued
            # heartbeat instructions. The metrics route is observability only;
            # sample it once after the protocol window and never let a
            # transient measurement connection decide a recovery outcome.
            # 恢复正确性与延迟由 AS 下发的心跳指令决定。metrics 路由仅用于观测；
            # 在协议窗口结束后最多采样一次，绝不能让瞬态测量连接决定恢复成败。
            latest_metrics = _try_fetch_as_metrics(
                context.survivors[0].config.as_base_url
            )

            detection_times = (
                latest_metrics.get("offline_detection_by_sid_seconds", {})
                if latest_metrics is not None else {}
            )
            recoveries: list[dict[str, object]] = []
            latencies: list[float] = []
            for victim in context.victims:
                recovered_tasks = {
                    task_id: recovered_by_task[task_id]
                    for task_id in context.victim_task_ids[victim.config.client_id]
                    if task_id in recovered_by_task
                }
                deferred_task_ids = sorted(
                    context.victim_task_ids[victim.config.client_id]
                    .difference(recovered_tasks)
                )
                successor_ids = sorted({
                    successor.config.client_id
                    for successor, _observed_at in recovered_tasks.values()
                })
                victim_sid = context.victim_sids_before_disconnect[victim.config.client_id]
                detected = detection_times.get(str(victim_sid))
                instruction_times = [
                    observed_at for _successor, observed_at in recovered_tasks.values()
                ]
                latency = None
                if instruction_times:
                    latency = max(
                        0.0,
                        max(instruction_times) - context.disconnect_started_at_by_client[
                            victim.config.client_id
                        ],
                    )
                if latency is not None:
                    latencies.append(latency)
                recoveries.append({
                    "victim_client_id": victim.config.client_id,
                    "successor_client_id": successor_ids[0] if len(successor_ids) == 1 else None,
                    "successor_client_ids": successor_ids,
                    "reassigned_task_count": len(recovered_tasks),
                    "deferred_task_count": len(deferred_task_ids),
                    "deferred_task_ids": deferred_task_ids,
                    "recovery_status": (
                        "takeover_dispatched"
                        if recovered_tasks and not deferred_task_ids
                        else "partially_deferred"
                        if recovered_tasks else "deferred_to_next_round"
                    ),
                    "recovery_latency_seconds": latency,
                    "offline_detected_after_seconds": (
                        None
                        if detected is None or victim.config.client_id not in context.disconnect_started_at_by_client
                        else max(
                            0.0,
                            float(detected) - context.disconnect_started_at_by_client[
                                victim.config.client_id
                            ],
                        )
                    ),
                })
            context.recovery_latency_seconds = max(latencies) if latencies else None
            context.recoveries = recoveries
            context.disconnect_events = events
        except BaseException as error:  # Preserve the monitor failure for the caller.
            # The worker must not silently disappear: a missing recovery is a
            # failed experimental case, never a zero-latency success. 后台线程
            # 不得静默退出；缺失恢复应使实验用例失败，绝不能伪装成零延迟成功。
            context.error = error
        finally:
            context.completed.set()

    def _await_training_failure(
        self,
        context: _TrainingFailureContext,
    ) -> tuple[float | None, list[dict[str, object]]]:
        """Join the monitor and return only verified recovery observations.

        等待监控线程并只返回经验证的恢复观测。
        """
        if not context.started.is_set() or context.thread is None:
            raise RuntimeError(
                "training-fault victim never began local training / "
                "训练故障受害客户端从未开始本地训练"
            )
        # The monitor's own protocol deadline is three short leases. The join
        # only adds a small scheduling allowance and cannot turn a lost worker
        # into an unbounded test hang. 监控器自身的协议截止时间是三个短租约；此
        # 等待只额外留出少量调度余量，不会将丢失的工作线程变成无限卡死的测试。
        context.thread.join(self.plan.failure_heartbeat_timeout_seconds * 3 + 10.0)
        if context.thread.is_alive():
            raise RuntimeError(
                "training-failure monitor exceeded its protocol deadline / "
                "训练故障监控超过协议截止时间"
            )
        if context.error is not None:
            raise RuntimeError(
                "training-failure monitor failed / 训练故障监控失败"
            ) from context.error
        return context.recovery_latency_seconds, list(context.recoveries or [])

    @staticmethod
    def _synchronize_recovered_training_ownership(
        clients: Sequence[ClientEntity],
        claims: dict[str, Any],
        queues: dict[str, LocalTrainingQueues],
    ) -> set[str]:
        """Synchronize post-recovery ownership from AS protocol responses.

        从 AS 协议响应同步恢复后的训练权。

        A recovery release normally enables AS heartbeat dispatch.  A heartbeat
        without instructions is nevertheless valid for an SID that receives no
        newly allocated task. In that case the client must execute the paper's
        independent CAS phase over its own already-registered labels and route
        that authoritative response. This preserves the distinction between an
        empty dispatch and a failed recovery, and never retains a stale local
        TRAIN decision. 恢复释放通常会启用 AS 心跳调度；但对未收到新分配任务的
        SID，空心跳指令仍是有效状态。此时客户端必须针对自己已登记的保护标签执行
        论文规定的独立 CAS 阶段，并依据该权威响应路由队列。这样能区分“未下发新任务”
        与“恢复失败”，且绝不保留过期的本地 TRAIN 决策。
        """
        # A recovery synchronization is a client-parallel protocol boundary:
        # first collect AS's heartbeat responses for all live SIDs together.
        # 恢复所有权同步是客户端并行的协议边界：先同时收集所有在线 SID 的 AS
        # 心跳响应。
        heartbeat_snapshots = _heartbeat_instruction_snapshots(clients)
        empty_instruction_clients = [
            client
            for client in clients
            if not heartbeat_snapshots[client.config.client_id]
        ]

        # A valid empty dispatch requires the paper's independent CAS phase.
        # Run only those confirmation requests concurrently as well; no client
        # waits for an unrelated SID before learning its authoritative state.
        # 有效的空下发需要执行论文规定的独立 CAS 阶段。仅对这些客户端并行执行
        # 状态确认；任何客户端均不会因无关 SID 而延迟获知其权威状态。
        fallback_decisions: dict[str, list[TaskClaimDecision]] = {}
        if empty_instruction_clients:
            def confirm_empty_dispatch(client: ClientEntity) -> tuple[str, list[TaskClaimDecision]]:
                """Read the exact current CAS decision for one empty dispatch.

                读取一个空下发对应的精确当前 CAS 决策。
                """
                prior_decisions, _prior_elapsed = claims["values"][client.config.client_id]
                return (
                    client.config.client_id,
                    client.claim_protected_labels_at_as(
                        [decision.protected_label for decision in prior_decisions]
                    ),
                )

            with ThreadPoolExecutor(max_workers=len(empty_instruction_clients)) as executor:
                fallback_decisions = dict(
                    executor.map(confirm_empty_dispatch, empty_instruction_clients)
                )

        changed_clients: set[str] = set()
        for client in clients:
            identifier = client.config.client_id
            prior_decisions, prior_elapsed = claims["values"][identifier]
            prior_train_labels = {
                decision.protected_label
                for decision in prior_decisions
                if decision.operation == "TRAIN"
            }
            instructions = heartbeat_snapshots[identifier]
            # Use the response-local snapshot obtained by the concurrent
            # protocol phase. Reading the mutable cached instructions here
            # permits a concurrently completing background heartbeat to
            # replace this recovery response with an older view.
            # 使用并行协议阶段获取的响应本地快照。此处读取可变缓存指令会使并发
            # 完成的后台心跳用较旧视图覆盖本次恢复响应。
            decisions = list(_instruction_decisions(instructions))
            if not decisions:
                # The fallback is a real CAS request on the exact protected
                # labels this client previously registered. It cannot create a
                # new label/index edge; it only obtains the current ownership
                # decision from AS. 回退路径对该客户端先前已登记的精确保护标签
                # 发起真实 CAS；不会创建新标签或索引边，仅从 AS 获取当前所有权决策。
                decisions = fallback_decisions[identifier]
            if not decisions or len({decision.protected_label for decision in decisions}) != len(decisions):
                raise RuntimeError(
                    "recovery ownership synchronization returned an invalid full instruction set / "
                    "恢复所有权同步返回了无效的完整指令集"
                )
            claims["values"][identifier] = (decisions, float(prior_elapsed))
            queues[identifier] = client.route_claim_decisions(decisions)
            current_train_labels = {
                decision.protected_label
                for decision in decisions
                if decision.operation == "TRAIN"
            }
            if current_train_labels != prior_train_labels:
                changed_clients.add(identifier)
        return changed_clients


    @staticmethod
    def _defer_offline_training_clients(
        clients: Sequence[ClientEntity],
        claims: dict[str, Any],
        queues: dict[str, LocalTrainingQueues],
    ) -> None:
        """Remove faulted clients from this round without deleting their data.

        将故障客户端移出本轮，但不删除其数据。

        Their original local records remain in the cold queue for observability.
        ``TRAIN`` decisions are converted to local DEDUP placeholders so the
        participant selector cannot include a disconnected SID in this round's
        FedAvg roster. AS is still authoritative for the actual task state:
        tasks claimed by live peers are PENDING there, while unrecoverable
        tasks are EMPTY and deferred to the next round. 原本的本地记录保留在冷队列
        中以便审计；``TRAIN`` 决策转换为本地 DEDUP 占位，确保参与者选择器不会将
        断连 SID 纳入本轮 FedAvg 名册。AS 仍是任务状态的权威：在线同伴接管的任务
        在 AS 中为 PENDING，无法接管的任务为 EMPTY 并留到下一轮。
        """
        for client in clients:
            identifier = client.config.client_id
            decisions, elapsed = claims["values"][identifier]
            claims["values"][identifier] = (
                [
                    TaskClaimDecision(
                        protected_label=decision.protected_label,
                        task_id=decision.task_id,
                        operation="DEDUP",
                        state="EMPTY",
                    )
                    for decision in decisions
                ],
                float(elapsed),
            )
            previous = queues[identifier]
            queues[identifier] = LocalTrainingQueues(
                hot_records=(),
                cold_records=tuple(
                    dict.fromkeys(previous.cold_records + previous.hot_records)
                ),
            )

    def _run_training(
        self,
        root: Path,
        clients: list[ClientEntity],
        queues: dict[str, Any],
        repetition: int,
        *,
        initial_checkpoints: dict[str, Path | None] | None = None,
        on_job_start: Callable[[ClientTrainingJob], None] | None = None,
        cancel_requested: Callable[[ClientTrainingJob], bool] | None = None,
        allowed_cancelled_client_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Run either a clearly marked simulation or real GPT client jobs.

        运行明确标注的模拟训练或真实 GPT 客户端任务。
        """
        jobs = [(client, queues[client.config.client_id].hot_records) for client in clients]
        started = time.perf_counter()
        if self.plan.training_mode == "simulated":
            durations = []
            client_metrics: dict[str, dict[str, object]] = {}
            for _client, records in jobs:
                if on_job_start is not None:
                    on_job_start(ClientTrainingJob(
                        client_id=_client.config.client_id,
                        command=("simulated",),
                        log_path=root / _client.config.client_id / "simulated-training.log",
                    ))
                duration = len(records) * self.plan.simulated_training_seconds_per_record
                time.sleep(duration)
                durations.append(duration)
                client_metrics[_client.config.client_id] = {
                    "sample_count": len(records),
                    "elapsed_seconds": duration,
                    "device": "simulated",
                    "cuda": {"available": False},
                }
            return {
                "wall_seconds": time.perf_counter() - started,
                "accumulated_seconds": sum(durations),
                "checkpoints": {},
                "client_metrics": client_metrics,
            }
        process_jobs = []
        checkpoints: dict[str, Path] = {}
        for client, records in jobs:
            if not records:
                continue
            data_path = root / f"{client.config.client_id}-hot.jsonl"
            if self._prepared_data_path is None:
                raise RuntimeError("prepared data was not loaded / 未加载预处理数据")
            materialize_hot_training_split(self._prepared_data_path, records, data_path)
            output = root / client.config.client_id / "training"
            checkpoints[client.config.client_id] = output / "local_model.safetensors"
            process_jobs.append(ClientTrainingJob(
                client_id=client.config.client_id,
                command=self._training_command(
                    data_path,
                    output,
                    self.plan.seed + repetition,
                    None if initial_checkpoints is None else initial_checkpoints.get(
                        client.config.client_id
                    ),
                ),
                log_path=root / client.config.client_id / "training.log",
            ))
        results = run_client_training_jobs(
            process_jobs,
            self.plan.gpu_ids,
            clients_per_gpu=self.plan.clients_per_gpu,
            gpu_memory_fraction=self.plan.gpu_memory_fraction_per_client,
            require_mps_partitioning=self.plan.require_mps_partitioning,
            timeout_seconds=self.plan.training_job_timeout_seconds,
            on_job_start=on_job_start,
            cancel_requested=cancel_requested,
        )
        if len(results) != len(process_jobs):
            raise RuntimeError(
                "scheduler did not return every GPT job / 调度器未返回每个 GPT 任务的结果"
            )
        allowed_cancelled = allowed_cancelled_client_ids or set()
        unexpected_cancellations = [
            result for result in results
            if result.cancelled and result.client_id not in allowed_cancelled
        ]
        if unexpected_cancellations:
            raise RuntimeError(
                "an unselected client training job was cancelled / "
                "未选中的客户端训练任务被取消"
            )
        failed = [
            result for result in results
            if result.return_code and not (
                result.cancelled and result.client_id in allowed_cancelled
            )
        ]
        if failed:
            timed_out = [result.client_id for result in failed if result.timed_out]
            detail = (
                "a GPT client training job timed out / GPT 客户端训练任务超时："
                + ", ".join(timed_out)
                if timed_out else
                "a GPT client training job failed / 一个 GPT 客户端训练任务失败"
            )
            raise RuntimeError(detail)
        # A paper run with at least one trainable client per configured physical
        # GPU must use every configured device. The scheduler assigns physical
        # IDs before CUDA_VISIBLE_DEVICES hides them from subprocesses; rejecting
        # a one-GPU result prevents a misleading two-GPU report. 当论文运行的可
        # 训练客户端数不少于配置物理 GPU 数时，必须使用每张配置 GPU。调度器在子进程
        # 通过 CUDA_VISIBLE_DEVICES 隐藏物理编号前完成分配；拒绝单 GPU 结果可避免
        # 生成误导性的双 GPU 报告。
        completed_results = [result for result in results if not result.cancelled]
        if len(completed_results) >= len(self.plan.gpu_ids):
            assigned_gpu_ids = {result.gpu_id for result in completed_results}
            required_gpu_ids = set(self.plan.gpu_ids)
            if assigned_gpu_ids != required_gpu_ids:
                raise RuntimeError(
                    "two-GPU training assignment is incomplete: "
                    f"assigned={sorted(assigned_gpu_ids)}, required={sorted(required_gpu_ids)} / "
                    "双 GPU 训练分配不完整"
                )
        client_metrics = {
            result.client_id: self._read_training_metrics(
                checkpoints[result.client_id].with_name("training_metrics.json"), result
            )
            for result in completed_results
        }
        if self.plan.require_cuda and len(completed_results) >= len(self.plan.gpu_ids):
            non_cuda = [
                identifier for identifier, metrics in client_metrics.items()
                if metrics.get("cuda", {}).get("available") is not True
            ]
            if non_cuda:
                raise RuntimeError(
                    "configured GPU training fell back from CUDA: "
                    f"{sorted(non_cuda)} / 配置的 GPU 训练回退了 CUDA"
                )
        completed_checkpoints = {
            result.client_id: checkpoints[result.client_id]
            for result in completed_results
        }
        return {
            "wall_seconds": time.perf_counter() - started,
            "accumulated_seconds": sum(
                result.elapsed_seconds for result in completed_results
            ),
            "checkpoints": completed_checkpoints,
            "client_metrics": client_metrics,
            "cancelled_client_ids": [
                result.client_id for result in results if result.cancelled
            ],
        }

    def _training_command(
        self,
        data_path: Path,
        output_directory: Path,
        seed: int,
        initial_checkpoint: Path | None,
    ) -> tuple[str, ...]:
        """Build one reproducible GPU trainer command without shell quoting.

        构建一个无需 shell 转义的可复现 GPU 训练器命令。
        """
        command = [
            sys.executable,
            "scripts/train_local_gpt.py",
            "--data", str(data_path),
            "--output-directory", str(output_directory),
            "--seed", str(seed),
            "--batch-size", str(self.plan.gpt_batch_size),
            "--gradient-accumulation", str(self.plan.gpt_gradient_accumulation),
            "--epochs", str(self.plan.gpt_local_epochs),
            "--max-length", str(self.plan.gpt_max_length),
            "--precision", self.plan.gpt_precision,
            "--checkpoint-precision", self.plan.gpt_checkpoint_precision,
        ]
        if initial_checkpoint is not None:
            command.extend(["--initial-checkpoint", str(initial_checkpoint)])
        if self.plan.require_cuda:
            command.append("--require-cuda")
        return tuple(command)

    @staticmethod
    def _read_training_metrics(path: Path, result: Any) -> dict[str, object]:
        """Merge trainer-emitted GPU evidence with scheduler assignment data.

        合并训练器写出的 GPU 证据和调度器分配数据。
        """
        if not path.is_file():
            raise RuntimeError("trainer did not write metrics / 训练器未写入指标")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("trainer metrics are invalid / 训练器指标无效")
        return {
            **payload,
            "scheduler": {
                "physical_gpu_id": result.gpu_id,
                "slot_index": result.slot_index,
                "gpu_memory_fraction": result.gpu_memory_fraction,
                "mps_partitioning_enabled": result.mps_partitioning_enabled,
                "mps_active_thread_percentage": result.mps_active_thread_percentage,
                "timed_out": result.timed_out,
                "subprocess_elapsed_seconds": result.elapsed_seconds,
            },
        }

    def _submit_updates(
        self,
        root: Path,
        clients: list[ClientEntity],
        claims: dict[str, Any],
        queues: dict[str, Any],
        training: dict[str, Any],
        repetition: int,
        *,
        round_id: int = 1,
        max_submit_workers: int | None = None,
    ) -> dict[str, Any]:
        """Submit independent model updates concurrently before FedAvg.

        在 FedAvg 前并发提交相互独立的模型更新。

        Each submission independently proves the SID still owns every TRAIN
        task and atomically changes those tasks from ``PENDING`` to
        ``COMMITTED`` at AS finalization.  The caller waits for every future,
        then AS verifies the complete roster and absence of any PENDING task
        before aggregation. 每次提交都会独立证明 SID 仍拥有全部 TRAIN 任务，并在
        AS 完成校验时原子地将其从 ``PENDING`` 改为 ``COMMITTED``。调用方等待每个
        future 完成，再由 AS 验证完整名册及不存在任何 PENDING 任务后执行聚合。
        """
        eligible_clients = []
        for client in clients:
            identifier = client.config.client_id
            initial_decisions = [
                item
                for item in claims["values"][identifier][0]
                if item.operation == "TRAIN"
            ]
            if initial_decisions:
                eligible_clients.append(client)
        recovery_lock = threading.Lock()

        def submit_one(client: ClientEntity) -> dict[str, object] | None:
            """Upload one client's checkpoint without serializing other SIDs.

            上传一个客户端的检查点，不串行化其他 SID。
            """
            identifier = client.config.client_id
            current_decisions = [
                item for item in claims["values"][identifier][0] if item.operation == "TRAIN"
            ]
            ownership_retrain_count = 0
            while current_decisions:
                hot_records = queues[identifier].hot_records
                if not hot_records:
                    return None
                client.send_as_heartbeat()
                checkpoint = training["checkpoints"].get(identifier)
                if checkpoint is None:
                    checkpoint = root / f"{identifier}-simulated-update.safetensors"
                    checkpoint.write_bytes(
                        f"simulated:{identifier}:{len(hot_records)}".encode("utf-8")
                    )
                try:
                    submit_started = time.perf_counter()
                    client.submit_model_update_at_as(
                        checkpoint,
                        current_decisions,
                        round_id=round_id,
                        sample_count=len(hot_records),
                    )
                except ModelUpdateOwnershipLostError as error:
                    # The AS 409 payload proves which original TRAIN labels are
                    # now DEDUP. A fresh CAS remains the network source of truth
                    # for every candidate label; it can also re-acquire a task
                    # that was released but not taken over. AS 409 载荷证明哪些
                    # 原 TRAIN 标签现在是 DEDUP；最新 CAS 仍是每个候选标签的网络
                    # 状态源，并可重新抢占已释放但尚未被接管的任务。
                    candidate_labels = tuple(
                        decision.protected_label for decision in current_decisions
                    )
                    fresh_claims = client.claim_protected_labels_at_as(candidate_labels)
                    forced_dedup_labels = tuple(
                        decision.protected_label for decision in error.dedup_instructions
                    )
                    refreshed_decisions = client.reconcile_recovery_claims(
                        fresh_claims,
                        forced_dedup_labels=forced_dedup_labels,
                    )
                    _replace_claim_decisions(
                        claims,
                        identifier,
                        candidate_labels,
                        refreshed_decisions,
                    )
                    refreshed_queues = client.route_claim_decisions(refreshed_decisions)
                    # Preserve previously cold records while placing every newly
                    # transferred label into the cold queue. 保留已有冷队列记录，
                    # 并将每个新转移标签加入冷队列。
                    queues[identifier] = LocalTrainingQueues(
                        refreshed_queues.hot_records,
                        tuple(dict.fromkeys(
                            queues[identifier].cold_records + refreshed_queues.cold_records
                        )),
                    )
                    current_decisions = [
                        item for item in refreshed_decisions if item.operation == "TRAIN"
                    ]
                    if not current_decisions:
                        return None
                    ownership_retrain_count += 1
                    # Recovery replaces shared training aggregates, so preserve
                    # their accounting consistency while normal submissions
                    # remain fully concurrent. 恢复会替换共享训练累计指标，因此
                    # 在保持普通提交全并发的同时，保护其记账一致性。
                    with recovery_lock:
                        self._run_recovery_training(
                            root,
                            client,
                            queues[identifier].hot_records,
                            training,
                            repetition,
                            ownership_retrain_count,
                        )
                    continue
                session = client.as_session
                if session is None:
                    raise RuntimeError(
                        "submitting client lost its AS session / 提交客户端丢失 AS 会话"
                    )
                return {
                    "sid": session.sid,
                    "decisions": tuple(current_decisions),
                    "sample_count": len(hot_records),
                    "ownership_retrain_count": ownership_retrain_count,
                    "submitted": True,
                    "upload_elapsed_seconds": time.perf_counter() - submit_started,
                    "trained_task_count": len(current_decisions),
                    "trained_sample_count": len(hot_records),
                }
            return None

        # A caller may bound the number of *independent* upload streams to the
        # AS request-worker capacity. This retains concurrent client uploads
        # while avoiding an evaluator-created burst that exceeds a deliberately
        # constrained AS. ``None`` preserves the normal full-client-parallel
        # protocol path. 调用方可将相互独立的上传流数量限制为 AS 请求工作线程
        # 容量：这会保留客户端并发上传，但避免评估器制造超过受限 AS 容量的突发。
        # ``None`` 保持常规的全部客户端并行协议路径。
        submit_workers = (
            len(eligible_clients)
            if max_submit_workers is None
            else min(len(eligible_clients), max(1, max_submit_workers))
        )
        submitted_phase = _parallel_phase(
            eligible_clients,
            submit_workers,
            submit_one,
        ) if eligible_clients else {
            "wall_seconds": 0.0,
            "accumulated_seconds": 0.0,
            "values": {},
        }
        client_metrics: dict[str, dict[str, object]] = {}
        submitted_sids: list[int] = []
        submitted_decisions: dict[str, tuple[TaskClaimDecision, ...]] = {}
        submitted_sample_counts: dict[str, int] = {}
        ownership_retrain_count = 0
        for client in eligible_clients:
            submitted = submitted_phase["values"][client.config.client_id][0]
            if submitted is None:
                continue
            identifier = client.config.client_id
            # Client-side response validation already guarantees a real SID in
            # production; keep this aggregation duck-typed for protocol tests.
            # 客户端响应校验已在生产路径保证 SID 合法；此处保持鸭子类型以支持协议测试。
            submitted_sids.append(submitted["sid"])
            submitted_decisions[identifier] = tuple(submitted["decisions"])
            submitted_sample_counts[identifier] = int(submitted["sample_count"])
            ownership_retrain_count += int(submitted["ownership_retrain_count"])
            client_metrics[identifier] = {
                key: value for key, value in submitted.items()
                if key not in {"sid", "decisions", "sample_count", "ownership_retrain_count"}
            }
        return {
            "submitted_client_count": len(submitted_sids),
            "submitted_sids": tuple(submitted_sids),
            "ownership_retrain_count": ownership_retrain_count,
            "client_metrics": client_metrics,
            "submitted_decisions": submitted_decisions,
            "submitted_sample_counts": submitted_sample_counts,
            "wall_seconds": submitted_phase["wall_seconds"],
            "accumulated_seconds": submitted_phase["accumulated_seconds"],
        }

    def _apply_pre_aggregate_incremental_training(
        self,
        root: Path,
        clients: Sequence[ClientEntity],
        claims: dict[str, Any],
        queues: dict[str, LocalTrainingQueues],
        training: dict[str, Any],
        submissions: dict[str, Any],
        repetition: int,
        round_id: int,
    ) -> dict[str, Any]:
        """Train heartbeat-assigned work and replace cumulative pending updates.

        训练心跳分配的新增工作，并替换累积的待聚合更新。

        A fixed FedAvg roster fixes participating SIDs, not the exact task set
        held by each SID. Before aggregation, every submitted participant polls
        its heartbeat. A newly issued ``TRAIN`` task is routed from cold to hot,
        trained from that client's current local checkpoint, then uploaded as a
        cumulative replacement that retains all earlier committed task IDs.
        Incremental jobs are scheduled together, so a recovery burst continues
        to use both configured GPUs. 固定 FedAvg 名册固定的是参与 SID，而不是每个
        SID 精确持有的任务集合。聚合前，每个已提交参与者都会轮询心跳；新下发的
        ``TRAIN`` 任务会从冷队列进入热队列，以该客户端当前本地检查点为起点训练，
        并作为保留所有先前已提交任务 ID 的累积替换更新上传。增量任务会一起调度，
        因此恢复突发仍可使用全部配置 GPU。
        """
        submitted_clients = [
            client
            for client in clients
            if client.config.client_id in submissions["submitted_decisions"]
        ]
        if not submitted_clients:
            return {"replacement_count": 0, "client_metrics": {}}

        # Heartbeats are a read/claim boundary, not a label-sharding mechanism:
        # each client receives one complete instruction list. 心跳是读取/认领
        # 边界而非标签切片机制：每个客户端接收一份完整指令列表。
        instruction_snapshots = _heartbeat_instruction_snapshots(submitted_clients)
        incremental_clients: list[ClientEntity] = []
        incremental_queues: dict[str, LocalTrainingQueues] = {}
        cumulative_decisions: dict[str, tuple[TaskClaimDecision, ...]] = {}
        for client in submitted_clients:
            identifier = client.config.client_id
            previous = tuple(submissions["submitted_decisions"][identifier])
            previous_ids = {item.task_id for item in previous}
            added = tuple(
                TaskClaimDecision(
                    instruction.protected_label,
                    instruction.task_id,
                    "TRAIN",
                    "PENDING",
                )
                for instruction in instruction_snapshots[identifier]
                if instruction.operation == "TRAIN"
                and instruction.task_id not in previous_ids
            )
            if not added:
                continue
            added_queues = client.route_claim_decisions(added)
            if not added_queues.hot_records:
                continue
            checkpoint = training["checkpoints"].get(identifier)
            if checkpoint is None:
                raise RuntimeError(
                    "incremental TRAIN work has no initial checkpoint / "
                    "增量 TRAIN 工作没有初始检查点"
                )
            existing_decisions, phase_seconds = claims["values"][identifier]
            claims["values"][identifier] = (
                list(existing_decisions) + list(added),
                phase_seconds,
            )
            previous_queues = queues[identifier]
            queues[identifier] = LocalTrainingQueues(
                hot_records=tuple(
                    dict.fromkeys(previous_queues.hot_records + added_queues.hot_records)
                ),
                cold_records=tuple(
                    dict.fromkeys(previous_queues.cold_records + added_queues.cold_records)
                ),
            )
            incremental_clients.append(client)
            incremental_queues[identifier] = added_queues
            cumulative_decisions[identifier] = previous + added

        if not incremental_clients:
            return {"replacement_count": 0, "client_metrics": {}}

        initial_checkpoints = {
            client.config.client_id: training["checkpoints"][client.config.client_id]
            for client in incremental_clients
        }
        incremental_training = self._run_training(
            root,
            incremental_clients,
            incremental_queues,
            repetition,
            initial_checkpoints=initial_checkpoints,
        )
        training["wall_seconds"] += incremental_training["wall_seconds"]
        training["accumulated_seconds"] += incremental_training["accumulated_seconds"]
        training["checkpoints"].update(incremental_training["checkpoints"])
        training.setdefault("client_metrics", {}).update(
            incremental_training["client_metrics"]
        )

        def submit_replacement(client: ClientEntity) -> tuple[int, float]:
            """Upload one post-training cumulative replacement update.

            上传一个训练后的累积替换更新。
            """
            identifier = client.config.client_id
            started = time.perf_counter()
            sample_count = len(queues[identifier].hot_records)
            client.submit_model_update_at_as(
                training["checkpoints"][identifier],
                cumulative_decisions[identifier],
                round_id=round_id,
                sample_count=sample_count,
            )
            return sample_count, time.perf_counter() - started

        uploaded = _parallel_phase(
            incremental_clients,
            len(incremental_clients),
            submit_replacement,
        )
        metrics: dict[str, dict[str, object]] = {}
        for client in incremental_clients:
            identifier = client.config.client_id
            sample_count, elapsed_seconds = uploaded["values"][identifier][0]
            decisions = cumulative_decisions[identifier]
            submissions["submitted_decisions"][identifier] = decisions
            submissions["submitted_sample_counts"][identifier] = sample_count
            metrics[identifier] = {
                "submitted": True,
                "replacement_update": True,
                "incremental_task_count": len(incremental_queues[identifier].hot_records),
                "trained_task_count": len(decisions),
                "trained_sample_count": sample_count,
                "upload_elapsed_seconds": elapsed_seconds,
            }
        return {"replacement_count": len(incremental_clients), "client_metrics": metrics}

    def _run_recovery_training(
        self,
        root: Path,
        client: ClientEntity,
        hot_records: tuple[bytes, ...],
        training: dict[str, Any],
        repetition: int,
        recovery_index: int,
        *,
        initial_checkpoint: Path | None = None,
    ) -> None:
        """Restart one client's training on only its current recovered hot set.

        仅使用一个客户端当前恢复后的热集合重新开始训练。
        """
        identifier = client.config.client_id
        started = time.perf_counter()
        if self.plan.training_mode == "simulated":
            checkpoint = root / f"{identifier}-recovery-{recovery_index}.safetensors"
            checkpoint.write_bytes(
                f"simulated-recovery:{identifier}:{len(hot_records)}".encode("utf-8")
            )
            training["checkpoints"][identifier] = checkpoint
            elapsed_seconds = time.perf_counter() - started
            training["wall_seconds"] += elapsed_seconds
            training["accumulated_seconds"] += elapsed_seconds
            return
        if self._prepared_data_path is None:
            raise RuntimeError("prepared data was not loaded / 未加载预处理数据")
        data_path = root / f"{identifier}-recovery-{recovery_index}-hot.jsonl"
        materialize_hot_training_split(self._prepared_data_path, hot_records, data_path)
        output = root / identifier / f"training-recovery-{recovery_index}"
        checkpoint = output / "local_model.safetensors"
        result = run_client_training_jobs(
            [
                ClientTrainingJob(
                    client_id=identifier,
                    command=self._training_command(
                        data_path,
                        output,
                        self.plan.seed + repetition,
                        initial_checkpoint,
                    ),
                    log_path=root / identifier / f"training-recovery-{recovery_index}.log",
                )
            ],
            self.plan.gpu_ids,
            clients_per_gpu=self.plan.clients_per_gpu,
            gpu_memory_fraction=self.plan.gpu_memory_fraction_per_client,
            require_mps_partitioning=self.plan.require_mps_partitioning,
            timeout_seconds=self.plan.training_job_timeout_seconds,
        )[0]
        if result.return_code:
            if result.timed_out:
                raise RuntimeError(
                    "recovered GPT client training job timed out / "
                    "恢复后的 GPT 客户端训练任务超时"
                )
            raise RuntimeError(
                "recovered GPT client training job failed / 恢复后的 GPT 客户端训练任务失败"
            )
        training["checkpoints"][identifier] = checkpoint
        training["wall_seconds"] += time.perf_counter() - started
        training["accumulated_seconds"] += result.elapsed_seconds
        training.setdefault("client_metrics", {})[identifier] = self._read_training_metrics(
            output / "training_metrics.json", result
        )


def run_evaluation_plan(plan: EvaluationPlan, *, progress: Callable[[str], None] = print) -> Path:
    """Run a plan through the public one-command API. / 通过公开一键 API 运行计划。"""
    return EvaluationRunner(plan, progress=progress).run()


def _trimmed_mean_case_result(repetitions: Sequence[dict[str, object]]) -> dict[str, object]:
    """Return the paper-facing mean after trimming end-to-end extremes.

    Formal runs contain four fully independent executions. They are ordered by
    end-to-end completion time, the fastest and slowest are discarded, and
    every numeric measurement in the remaining two is averaged. A one-run
    diagnostic remains supported for narrow unit tests. 正式运行包含四次完整独立
    执行；按端到端完成时间排序，剔除最快与最慢两次，对中间两次的每项数值测量
    求均值。窄范围单元测试仍允许一次诊断运行。
    """
    if not repetitions:
        raise ValueError("cannot aggregate zero completed repetitions / 无法聚合零次完成重复")
    ordered = sorted(
        enumerate(repetitions, start=1),
        key=lambda item: (float(item[1]["total_completion_seconds"]), item[0]),
    )
    is_formal_trimmed_run = len(ordered) == 4
    retained = ordered[1:-1] if is_formal_trimmed_run else ordered
    averaged = _mean_evaluation_value([result for _index, result in retained])
    if not isinstance(averaged, dict):
        raise RuntimeError("averaged evaluation result must be an object / 平均评估结果必须是对象")
    exemplar = repetitions[0]
    averaged.update({
        "schema_version": exemplar["schema_version"],
        "status": "completed",
        "suite": exemplar["suite"],
        "variable": exemplar["variable"],
        "value": exemplar["value"],
        "configuration": exemplar["configuration"],
        "training_mode": exemplar["training_mode"],
        "oprf_suite": exemplar["oprf_suite"],
        "service_mode": exemplar["service_mode"],
        "aggregation": {
            "method": (
                f"{len(ordered)}_run_trimmed_mean_by_total_completion_seconds"
                if is_formal_trimmed_run else "arithmetic_mean_diagnostic"
            ),
            "total_repetitions": len(ordered),
            "retained_repetitions": [index for index, _result in retained],
            "discarded_repetitions": (
                [ordered[0][0], ordered[-1][0]] if is_formal_trimmed_run else []
            ),
        },
    })
    # Repeat IDs and temporary service paths are raw-run provenance, not
    # reportable averages. 重复编号和临时服务路径属于原始运行溯源，不是可报告均值。
    averaged.pop("repetition", None)
    return averaged


def _failed_trimmed_case_result(
    case: _Case,
    completed: Sequence[dict[str, object]],
    failures: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Record an incomplete formal observation without fabricating a mean.

    记录未完成的正式观测，且绝不伪造均值。
    """
    return {
        "schema_version": "1.0",
        "status": "failed",
        "suite": case.suite,
        "variable": case.variable,
        "value": case.value,
        "configuration": asdict(case),
        "failure": {
            "completed_repetitions": len(completed),
            "failed_repetitions": [
                {
                    "repetition": item.get("repetition"),
                    "type": dict(item.get("failure", {})).get("type"),
                    "message": dict(item.get("failure", {})).get("message"),
                    "traceback": dict(item.get("failure", {})).get("traceback"),
                }
                for item in failures
            ],
        },
    }


def _mean_evaluation_value(values: Sequence[Any]) -> Any:
    """Average numeric JSON-like metrics while retaining stable descriptors.

    平均 JSON 风格指标中的数值，同时保留稳定描述字段。
    """
    if not values:
        raise ValueError("cannot average an empty value sequence / 无法平均空值序列")
    first = values[0]
    if isinstance(first, bool):
        return first
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        return sum(float(value) for value in numeric) / len(numeric) if len(numeric) == len(values) else first
    if first is None:
        return None
    if isinstance(first, dict):
        common_keys = set(first)
        for value in values[1:]:
            if not isinstance(value, dict):
                return first
            common_keys.intersection_update(value)
        return {
            key: _mean_evaluation_value([value[key] for value in values])
            for key in first
            if key in common_keys and key not in {"checkpoint_path", "diagnostic_paths"}
        }
    if isinstance(first, list):
        if not all(isinstance(value, list) and len(value) == len(first) for value in values):
            return []
        return [
            _mean_evaluation_value([value[index] for value in values])
            for index in range(len(first))
        ]
    # IDs and categorical values are not numeric measurements. They are
    # retained only as stable descriptors from the first retained execution.
    # 标识符和分类值不是数值测量，仅保留中间组首个运行的稳定描述。
    return first


def _load_completed_trimmed_prefix(
    root: Path,
    plan: EvaluationPlan,
    cases: Sequence[_Case],
    start_case: int,
) -> list[dict[str, object]]:
    """Load only complete aggregate rows before a logical case ordinal.

    仅加载逻辑用例序号之前的完整聚合行。
    """
    if start_case == 1:
        return []
    results_path = root / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"cannot resume without {results_path} / 缺少 {results_path}，无法续跑")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("existing results payload is invalid / 已有结果负载无效")
    _validate_resume_plan(payload, plan)
    stored = payload.get("results")
    if not isinstance(stored, list) or len(stored) < start_case - 1:
        raise ValueError("existing aggregate prefix is incomplete / 已有聚合前缀不完整")
    prefix: list[dict[str, object]] = []
    for ordinal, (case, result) in enumerate(
        zip(cases[:start_case - 1], stored[:start_case - 1], strict=True), start=1
    ):
        if not isinstance(result, dict) or result.get("status") != "completed":
            raise ValueError(f"resume case {ordinal} is not completed / 续跑第 {ordinal} 个用例未完成")
        if result.get("suite") != case.suite or result.get("variable") != case.variable or result.get("value") != case.value:
            raise ValueError(f"resume case {ordinal} identity differs / 续跑第 {ordinal} 个用例标识不一致")
        aggregation = result.get("aggregation")
        if not isinstance(aggregation, dict) or aggregation.get("total_repetitions") != plan.repetitions:
            raise ValueError(f"resume case {ordinal} aggregation differs / 续跑第 {ordinal} 个用例聚合口径不一致")
        prefix.append(result)
    return prefix


def _load_completed_resume_prefix(
    root: Path,
    plan: EvaluationPlan,
    case_runs: Sequence[tuple[_Case, int]],
    start_case: int,
) -> list[dict[str, object]]:
    """Load and validate the completed prefix required by a resumed run.

    ``start_case`` is a one-based ordinal in the deterministic plan order. A
    continuation may retain only a fully completed prefix; failed, missing, or
    differently configured rows must be rerun instead of being merged into a
    new measurement. ``results.json`` remains the source of truth because it
    contains the full per-case configuration and raw observations. ``start_case``
    是确定性计划顺序中的一基编号。续跑只能保留完整成功的前缀；失败、缺失或配置
    不同的行必须重新运行，不能与新测量混合。``results.json`` 是事实来源，因为它
    包含每个用例的完整配置与原始观测值。
    """
    if start_case == 1:
        return []

    results_path = root / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            "cannot resume without existing results.json / "
            "缺少已有 results.json，无法续跑："
            f"{results_path}"
        )
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "existing results.json is invalid JSON / 已有 results.json 不是有效 JSON"
        ) from error
    stored_results = payload.get("results")
    if not isinstance(stored_results, list):
        raise ValueError(
            "existing results.json has no results list / "
            "已有 results.json 不含 results 列表"
        )
    _validate_resume_plan(payload, plan)

    prefix_count = start_case - 1
    if len(stored_results) < prefix_count:
        raise ValueError(
            f"resume needs {prefix_count} prior results but found {len(stored_results)} / "
            f"续跑需要前 {prefix_count} 个结果，但只找到 {len(stored_results)} 个"
        )
    prefix: list[dict[str, object]] = []
    for ordinal, ((case, repetition), stored) in enumerate(
        zip(case_runs[:prefix_count], stored_results[:prefix_count], strict=True),
        start=1,
    ):
        if not isinstance(stored, dict):
            raise ValueError(
                f"stored case {ordinal} is not an object / 已存储的第 {ordinal} 个用例不是对象"
            )
        _validate_resume_case(stored, case, repetition, ordinal)
        prefix.append(stored)
    return prefix


def _validate_resume_plan(payload: dict[str, object], plan: EvaluationPlan) -> None:
    """Require every retained result to use the same global experiment setup.

    要求每个保留结果均使用相同的全局实验设置。
    """
    stored_plan = payload.get("plan")
    if not isinstance(stored_plan, dict):
        raise ValueError(
            "existing results.json has no plan provenance / "
            "已有 results.json 缺少计划溯源信息"
        )
    expected_plan = _json_plan(plan)
    # The directory is already fixed by the file being resumed, and start_case
    # deliberately changes between the old and new invocation. Every other
    # plan field, including data provenance and trainer settings, must agree.
    # 目录已经由被续跑的文件固定，且 start_case 必然在旧、新调用间变化；其余所有
    # 计划字段（包括数据溯源和训练设置）都必须一致。
    ignored = {"output_directory", "start_case"}
    mismatches = [
        field
        for field, expected in expected_plan.items()
        if field not in ignored and stored_plan.get(field) != expected
    ]
    if mismatches:
        rendered = ", ".join(mismatches)
        raise ValueError(
            "resume plan does not match existing results: "
            f"{rendered} / 续跑计划与已有结果不一致：{rendered}"
        )


def _validate_resume_case(
    stored: dict[str, object],
    expected_case: _Case,
    expected_repetition: int,
    ordinal: int,
) -> None:
    """Reject a prefix row that cannot be compared with the current plan.

    拒绝无法与当前计划比较的前缀结果行。
    """
    identity = {
        "suite": expected_case.suite,
        "variable": expected_case.variable,
        "value": expected_case.value,
        "repetition": expected_repetition,
    }
    for field, expected in identity.items():
        if stored.get(field) != expected:
            raise ValueError(
                f"resume case {ordinal} mismatches {field}: expected {expected!r}, "
                f"found {stored.get(field)!r} / 续跑第 {ordinal} 个用例的 {field} 不匹配："
                f"期望 {expected!r}，实际 {stored.get(field)!r}"
            )
    if stored.get("status", "completed") != "completed":
        raise ValueError(
            f"resume case {ordinal} is not completed / 续跑第 {ordinal} 个用例未成功完成"
        )
    configuration = stored.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(
            f"resume case {ordinal} has no configuration / "
            f"续跑第 {ordinal} 个用例缺少 configuration"
        )
    expected_configuration = {
        "clients": expected_case.clients,
        "request_workers": expected_case.request_workers,
        "duplicate_ratio": expected_case.duplicate_ratio,
        "backend_workers": expected_case.backend_workers,
        "records_per_client": expected_case.records_per_client,
        "failure_phase": expected_case.failure_phase,
        "joining_clients": expected_case.joining_clients,
        "ablation": expected_case.ablation,
    }
    for field, expected in expected_configuration.items():
        if configuration.get(field) != expected:
            raise ValueError(
                f"resume case {ordinal} configuration mismatches {field}: "
                f"expected {expected!r}, found {configuration.get(field)!r} / "
                f"续跑第 {ordinal} 个用例的配置 {field} 不匹配：期望 {expected!r}，"
                f"实际 {configuration.get(field)!r}"
            )


def _build_cases(plan: EvaluationPlan) -> Iterable[_Case]:
    """Yield one-factor suites plus recovery and dynamic-join measurements.

    产生单因素套件以及恢复和动态加入测量。
    """
    base = dict(
        clients=plan.base_clients,
        # Each client submits its complete protected-label set FP_i exactly
        # once. Therefore the protocol workflow concurrency is the number of
        # active clients, not an artificial tag-shard count. 每个客户端恰好一次
        # 提交完整受保护标签集合 FP_i；因此协议工作流并发度等于在线客户端数，不能
        # 伪造为标签分片数量。
        request_workers=plan.base_clients,
        duplicate_ratio=plan.base_duplicate_ratio,
        backend_workers=plan.base_backend_workers,
        records_per_client=plan.base_records_per_client,
    )
    for name, values, key in (
        ("parallel_client_scale", plan.client_counts, "clients"),
        ("dedup_load", plan.duplicate_ratios, "duplicate_ratio"),
        ("as_backend_parallelism", plan.backend_worker_counts, "backend_workers"),
        ("data_scale", plan.records_per_client_values, "records_per_client"),
    ):
        for value in values:
            settings = {**base, key: value}
            if key == "clients":
                # Keep one complete-FP_i protocol worker per active client in
                # the client-scale suite. 客户端规模套件中，每个在线客户端对应一个
                # 完整 FP_i 协议工作流。
                settings["request_workers"] = int(value)
            yield _Case(name, key, value, **settings)
    for failure_rate in plan.failure_rates:
        yield _Case(
            "fault_dedup", "failure_rate", failure_rate, **base, failure_phase="dedup"
        )
        yield _Case(
            "fault_training",
            "failure_rate",
            failure_rate,
            **base,
            failure_phase="training",
        )
    yield _Case("dynamic_join_base", "joining_clients", 0, **base)
    for joining_clients in plan.join_client_counts:
        yield _Case(
            "dynamic_join", "joining_clients", joining_clients, **base,
            joining_clients=joining_clients,
        )
    if plan.include_ablations:
        # Full DwT-FL controls execute the same live OPRF, registration, CAS,
        # training, upload, and FedAvg flow as their variants. 完整 DwT-FL
        # 对照与各变体执行相同的实时 OPRF、登记、CAS、训练、上传和 FedAvg 流程。
        yield _Case(
            "ablation", "full_dwtfl_single_round", plan.ablation_failure_rate,
            **base, failure_phase="training", ablation="full_dwtfl_single_round",
        )
        for variant in (
            "without_cas",
            "without_inverse_index",
        ):
            yield _Case(
                "ablation",
                variant,
                plan.ablation_failure_rate,
                **base,
                failure_phase="training",
                ablation=variant,
            )
        yield _Case(
            "ablation", "full_dwtfl_history", plan.ablation_failure_rate,
            **base, failure_phase="training", ablation="full_dwtfl_history",
        )
        yield _Case(
            "ablation", "without_history_scheduling", plan.ablation_failure_rate,
            **base, failure_phase="training", ablation="without_history_scheduling",
        )


def _round_count_for_case(plan: EvaluationPlan, case: _Case) -> int:
    """Return the minimum meaningful round count for one measurement.

    返回一次测量所需的最小有效轮数。

    CAS and inverse-index ablations affect a single round. History scheduling
    instead compares the next-round owner against the prior trainer, so the
    real GPT experiment must run two rounds even when the normal plan requests
    one. Explicit multi-round plans remain unchanged through ``max``. The
    protocol-only simulation deliberately remains one round because it does
    not execute FedAvg or reset AS state, and therefore cannot stand in for a
    history-scheduling result. CAS 和倒排索引消融影响单轮；历史调度则比较下一轮
    所有者与前一轮训练者，因此即使普通计划请求一轮，真实 GPT 实验也必须运行两轮。
    显式多轮计划通过 ``max`` 保持不变。纯协议模拟刻意保持一轮，因为它不执行
    FedAvg 或重置 AS 状态，不能冒充历史调度结果。
    """
    if case.ablation in {"full_dwtfl_history", "without_history_scheduling"} and plan.training_mode == "gpt":
        return max(plan.federated_rounds, 2)
    return plan.federated_rounds


def _selected_cases(plan: EvaluationPlan) -> tuple[_Case, ...]:
    """Return every case requested by the complete or focused plan.

    返回完整计划或定向计划请求的全部用例。

    An empty selector preserves the paper-metric default. A non-empty selector
    is intentionally an allow-list, so a recovery rerun cannot accidentally
    spend hours repeating unrelated real GPT cases. 空选择器保留论文指标的默认
    全量行为；非空选择器是显式允许列表，因此恢复重跑不会意外花费数小时重复无关的
    真实 GPT 用例。
    """
    cases = tuple(_build_cases(plan))
    if not plan.included_suites:
        return cases
    selected = tuple(case for case in cases if case.suite in plan.included_suites)
    if not selected:
        raise ValueError(
            "included_suites selected no cases; check include_ablations and selectors / "
            "included_suites 未选中任何用例；请检查 include_ablations 与选择器"
        )
    return selected


def _failure_victim_count(case: _Case, total_clients: int) -> int:
    """Convert a nonzero configured rate into distinct recoverable victims.

    将非零配置比例转换为不同且可恢复的掉线客户端数量。

    At least one online successor is always retained.  The concrete count is
    emitted in every case configuration and individual recovery rows, so a
    30-percent setting is never mislabeled as a single-client failure. 始终保留
    至少一个在线接管者；具体数量会写入每个用例配置和每客户端恢复行，因此 30% 设置
    不会被错误标记为单客户端故障。
    """
    if total_clients < 2:
        return 0
    rate = float(case.value)
    if rate <= 0.0:
        return 0
    return min(total_clients - 1, max(1, math.ceil(total_clients * rate)))


def _select_training_failure_victims(
    clients: Sequence[ClientEntity],
    claims: Mapping[str, Any],
    requested_count: int,
) -> list[ClientEntity]:
    """Select dropped trainers that expose an observable takeover path.

    选择具有可观测接管路径的掉线训练者。

    A training-fault experiment is meaningful only if an offline trainer owns
    at least one duplicated task that remains known to an online client.  The
    old positional selection (for example, always selecting ``client-0``)
    became invalid once clients started arriving asynchronously: a selected
    client could own only exclusive work, so AS correctly had no successor to
    report.  This helper derives ownership solely from the completed CAS
    responses, retains the requested dropout population when possible, and
    never selects every owner of an otherwise recoverable task.  训练故障实验
    只有在掉线训练者至少持有一项仍被在线客户端知晓的重复任务时才具有可观测意义。
    旧的按位置选择方式（例如总是选择 ``client-0``）在客户端异步上线后不再成立：
    被选择的客户端可能只持有独有任务，因此 AS 正确地不存在可报告的接管者。该辅助
    函数仅依据已完成的 CAS 响应推导所有权；在可行时保留请求的掉线规模，并且绝不
    选择某一可恢复任务的全部所有者。
    """
    if requested_count <= 0 or len(clients) < 2:
        return []

    client_by_identifier = {
        client.config.client_id: client for client in clients
    }
    known_owners_by_task: dict[int, set[str]] = {}
    train_tasks_by_client: dict[str, set[int]] = {
        identifier: set() for identifier in client_by_identifier
    }
    for identifier in client_by_identifier:
        decisions, _elapsed = claims["values"][identifier]
        for decision in decisions:
            known_owners_by_task.setdefault(decision.task_id, set()).add(identifier)
            if decision.operation == "TRAIN":
                train_tasks_by_client[identifier].add(decision.task_id)

    # Prefer owners of more shared tasks so every configured dropout rate
    # measures AS detection plus a genuine takeover, not an exclusive-task
    # reconnect.  The identifier tie-breaker keeps the generated report
    # reproducible for a fixed evaluation seed. 优先选择拥有更多共享任务的所有者，
    # 使每个配置的掉线率都测量 AS 检测与真实接管，而非独有任务的重连；客户端标识
    # 作为并列规则，确保固定评估种子下报告可复现。
    candidates = sorted(
        (
            client for client in clients
            if train_tasks_by_client[client.config.client_id]
        ),
        key=lambda client: (
            -sum(
                len(known_owners_by_task[task_id]) > 1
                for task_id in train_tasks_by_client[client.config.client_id]
            ),
            client.config.client_id,
        ),
    )
    selected: list[ClientEntity] = []
    selected_identifiers: set[str] = set()
    all_identifiers = set(client_by_identifier)

    def leaves_online_alternate(candidate_identifier: str) -> bool:
        """Check that each selected owner retains a live alternate.

        检查每个已选所有者仍保留在线备用者。
        """
        tentative = selected_identifiers | {candidate_identifier}
        online_identifiers = all_identifiers - tentative
        if not online_identifiers:
            return False
        for selected_identifier in tentative:
            shared_train_tasks = (
                task_id
                for task_id in train_tasks_by_client[selected_identifier]
                if len(known_owners_by_task[task_id]) > 1
            )
            if not any(
                known_owners_by_task[task_id].intersection(online_identifiers)
                for task_id in shared_train_tasks
            ):
                return False
        return True

    for candidate in candidates:
        if len(selected) >= requested_count:
            break
        identifier = candidate.config.client_id
        if leaves_online_alternate(identifier):
            selected.append(candidate)
            selected_identifiers.add(identifier)

    # The configured rate describes offline *clients*, not only current TRAIN
    # winners. A small all-duplicate topology can correctly elect fewer TRAIN
    # owners than the requested number of dropped clients. First retain any
    # remaining trainers, then fill the population with non-training clients;
    # the latter legitimately have no takeover row. 配置比例描述的是掉线
    # *客户端*，而不只是当前的 TRAIN 获胜者。小型全重复拓扑可能正确地产生少于
    # 请求掉线人数的 TRAIN 所有者。先保留其余训练者，再以未训练客户端补足掉线
    # 群体；后者没有接管记录是符合协议语义的。
    for candidate in candidates:
        if len(selected) >= requested_count:
            break
        identifier = candidate.config.client_id
        if identifier not in selected_identifiers and len(selected) < len(clients) - 1:
            selected.append(candidate)
            selected_identifiers.add(identifier)

    for candidate in clients:
        if len(selected) >= requested_count:
            break
        identifier = candidate.config.client_id
        if identifier not in selected_identifiers and len(selected) < len(clients) - 1:
            selected.append(candidate)
            selected_identifiers.add(identifier)

    if len(selected) != requested_count:
        raise RuntimeError(
            "training-fault case cannot retain one live client at the requested rate / "
            "训练故障用例无法在请求比例下保留一个在线客户端"
        )
    return selected


def _current_training_participants(
    clients: Sequence[ClientEntity],
    claims: dict[str, Any],
    queues: dict[str, LocalTrainingQueues],
) -> list[ClientEntity]:
    """Return only clients whose current decisions and hot queues agree.

    仅返回当前决策与热队列一致的客户端。

    The configured FedAvg roster must never include a client whose data was
    moved cold by recovery. Conversely, a hot queue without a current TRAIN
    decision is unsafe because its update would commit no corresponding task.
    固定的 FedAvg 名册绝不能包含恢复后数据已转入冷队列的客户端；反之，没有当前
    TRAIN 决策的热队列也不安全，因为其更新没有对应可提交任务。
    """
    participants: list[ClientEntity] = []
    for client in clients:
        identifier = client.config.client_id
        decisions = claims["values"][identifier][0]
        has_train_decision = any(decision.operation == "TRAIN" for decision in decisions)
        has_hot_records = bool(queues[identifier].hot_records)
        if has_train_decision != has_hot_records:
            raise RuntimeError(
                "current training claims and hot queue disagree for "
                f"{identifier} / 当前训练决策与热队列不一致：{identifier}"
            )
        if has_train_decision:
            participants.append(client)
    return participants


def _merged_interval_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the duration of the union of non-empty monotonic intervals.

    返回非空单调时间区间并集的持续时间。

    The evaluator uses this for client protocol activity only. It intentionally
    removes gaps in which no client has begun registration/CAS because of the
    experiment's own asynchronous-arrival schedule. 评估器仅将其用于客户端
    协议活跃时间；它刻意排除因实验自身异步上线调度而没有客户端执行登记/CAS 的
    空档。
    """
    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    )
    if not ordered:
        return 0.0
    total = 0.0
    active_start, active_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= active_end:
            active_end = max(active_end, end)
            continue
        total += active_end - active_start
        active_start, active_end = start, end
    return total + active_end - active_start


def _parallel_phase(
    clients: list[ClientEntity],
    workers: int,
    operation: Callable[[ClientEntity], Any],
) -> dict[str, Any]:
    """Run one client RPC phase concurrently and retain individual durations.

    并发运行一个客户端 RPC 阶段并保留各自耗时。
    """
    def invoke(client: ClientEntity) -> tuple[Any, float]:
        started = time.perf_counter()
        return operation(client), time.perf_counter() - started

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pairs = list(executor.map(invoke, clients))
    values = {client.config.client_id: pair for client, pair in zip(clients, pairs, strict=True)}
    return {"wall_seconds": time.perf_counter() - started,
            "accumulated_seconds": sum(duration for _value, duration in pairs), "values": values}


def _heartbeat_instruction_snapshots(
    clients: Sequence[ClientEntity],
) -> dict[str, tuple[RoundInstruction, ...]]:
    """Read one parallel, response-local heartbeat snapshot per live client.

    为每个在线客户端并行读取一次响应本地的心跳快照。

    Recovery instruction dispatch is a client-parallel protocol phase: every
    online holder must receive AS's state decision without waiting behind an
    unrelated holder. The worker count therefore equals the number of supplied
    clients. ``ClientEntity`` serializes only duplicate requests for the same
    SID, while AS exposes heartbeat on its isolated control route.
    恢复指令下发是客户端并行的协议阶段：每个在线持有者都必须接收 AS 的状态决策，
    而不能被无关持有者阻塞。因此工作线程数等于传入的客户端数。``ClientEntity``
    只串行化同一 SID 的重复请求，AS 则在独立控制路由上提供心跳服务。
    """
    if not clients:
        return {}

    def read_one(client: ClientEntity) -> tuple[str, tuple[RoundInstruction, ...]]:
        """Return the immutable heartbeat response belonging to one SID.

        返回一个 SID 所属的不可变心跳响应。
        """
        _session, instructions = client.send_as_heartbeat_with_instructions()
        return client.config.client_id, instructions

    with ThreadPoolExecutor(max_workers=len(clients)) as executor:
        pairs = list(executor.map(read_one, clients))
    return dict(pairs)


def _global_model_distribution_workers(
    plan: EvaluationPlan,
    case: _Case,
    participant_count: int,
) -> int:
    """Return the caller-side read concurrency for one immutable model phase.

    返回一个不可变模型阶段的客户端侧读取并发度。

    The executor may start every client read concurrently. The AS itself still
    enforces ``case.backend_workers`` at its listener, so server capacity is
    measured without silently turning a client-scale experiment into a client
    throttle. 执行器可以并发启动每个客户端的读取；AS 仍会在监听端按
    ``case.backend_workers`` 强制服务端容量，因此不会悄然把客户端规模实验变成
    客户端限流实验。
    """
    if participant_count < 1:
        raise ValueError("participant_count must be positive / 参与客户端数量必须为正数")
    return min(plan.global_model_download_workers, participant_count)


def _replace_claim_decisions(
    claims: dict[str, Any],
    client_id: str,
    replaced_labels: Sequence[str],
    refreshed_decisions: Sequence[TaskClaimDecision],
) -> None:
    """Replace one recovering client's decisions without disturbing other labels.

    在不影响其他标签的情况下替换一个恢复客户端的决策。
    """
    replaced = set(replaced_labels)
    refreshed_by_label = {
        decision.protected_label: decision for decision in refreshed_decisions
    }
    if set(refreshed_by_label) != replaced:
        raise RuntimeError(
            "recovery CAS must return every requested label / "
            "恢复 CAS 必须返回每个请求标签"
        )
    existing_decisions, phase_seconds = claims["values"][client_id]
    merged = [
        refreshed_by_label[decision.protected_label]
        if decision.protected_label in replaced else decision
        for decision in existing_decisions
    ]
    claims["values"][client_id] = (merged, phase_seconds)


def _round_instruction_decisions(client: ClientEntity) -> tuple[TaskClaimDecision, ...]:
    """Convert one post-FedAvg heartbeat into submission-safe decisions.

    将一次 FedAvg 后心跳转换为可安全提交的决策。
    """
    return _instruction_decisions(client.last_round_instructions)


def _instruction_decisions(
    instructions: Sequence[Any],
) -> tuple[TaskClaimDecision, ...]:
    """Convert one immutable AS instruction snapshot into task decisions.

    将一份不可变 AS 指令快照转换为任务决策。
    """
    return tuple(
        TaskClaimDecision(
            protected_label=instruction.protected_label,
            task_id=instruction.task_id,
            operation=instruction.operation,
            state="PENDING" if instruction.operation == "TRAIN" else "EMPTY",
        )
        for instruction in instructions
    )


def _client_metric_rows(
    clients: Sequence[ClientEntity],
    records_by_client: dict[str, list[str]],
    queues: dict[str, LocalTrainingQueues],
    training_metrics: dict[str, dict[str, object]],
    submission_metrics: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Write one observable row for every logical client, including no-op ones.

    为每个逻辑客户端写入一条可观测记录，包括未训练客户端。
    """
    rows: list[dict[str, object]] = []
    for client in clients:
        identifier = client.config.client_id
        queues_for_client = queues[identifier]
        assigned = records_by_client[identifier]
        rows.append({
            "client_id": identifier,
            "input_record_count": len(assigned),
            "unique_input_record_count": len(set(assigned)),
            "hot_record_count": len(queues_for_client.hot_records),
            "cold_record_count": len(queues_for_client.cold_records),
            "training": training_metrics.get(identifier, {}),
            "submission": submission_metrics.get(identifier, {"submitted": False}),
        })
    return rows


def _synthetic_records_for_case(case: _Case, repetition: int) -> dict[str, list[str]]:
    """Create deterministic plaintext ownership with a controlled shared fraction.

    创建确定性明文所有权，并控制共享比例。
    """
    shared_count = round(case.records_per_client * case.duplicate_ratio)
    shared = [f"shared-{repetition}-{number}" for number in range(shared_count)]
    # Simulation remains protocol-only, but its ownership map must include
    # late joiners exactly as the real prepared-data allocator does. Otherwise
    # dynamic-join control-flow coverage would fail before it reaches AS. 模拟
    # 仍仅用于协议测试，但其所有权映射必须与真实预处理数据分配器一样包含后加入
    # 客户端；否则动态加入控制流会在到达 AS 前失败。
    total_clients = case.clients + case.joining_clients
    return {
        f"client-{client}": shared + [f"unique-{repetition}-{client}-{number}"
            for number in range(case.records_per_client - shared_count)]
        for client in range(total_clients)
    }


def _prepared_records_for_case(
    case: _Case,
    repetition: int,
    prepared_records: tuple[PreparedRecord, ...],
    join_duplicate_ratio: float,
) -> dict[str, list[str]]:
    """Partition real text with paper-style pairwise client duplicates.

    使用论文式客户端两两重复来划分真实文本。

    The reference paper implants duplicate samples pairwise rather than placing
    one identical shared pool in every client.  This implementation preserves
    that topology: every generated duplicate belongs to exactly two client
    splits, and every other selected record belongs to one split.  The source
    text is rotated by repetition without replacement within a case. 参考论文
    以客户端两两方式植入重复样本，而非让所有客户端共享同一个文本池。本实现保留该
    拓扑：每个生成的重复项恰好属于两个客户端划分，其他选中记录只属于一个划分；
    每个用例内源文本按重复次数轮换且不重复。
    """
    total_clients = case.clients + case.joining_clients
    identifiers = [f"client-{index}" for index in range(total_clients)]
    requested_duplicates = [
        round(case.records_per_client * (
            case.duplicate_ratio if index < case.clients else join_duplicate_ratio
        ))
        for index in range(total_clients)
    ]
    # One pairwise duplicate consumes two requested duplicate slots. Rounding can
    # leave an odd total; reduce only one requested slot so the actual overlap is
    # attainable and later recorded from the concrete assignment. 一条两两重复会
    # 消耗两个重复配额。四舍五入可能留下奇数总量；此时仅减少一个配额，使实际重叠可
    # 实现，并从具体划分中记录实际比例。
    if sum(requested_duplicates) % 2:
        largest = max(range(total_clients), key=requested_duplicates.__getitem__)
        requested_duplicates[largest] -= 1
    pair_count = sum(requested_duplicates) // 2
    required = pair_count + sum(
        case.records_per_client - duplicate_count
        for duplicate_count in requested_duplicates
    )
    if len(prepared_records) < required:
        raise ValueError(
            "prepared data is too small for this paper-scale-lite case / "
            "预处理数据不足以运行该论文规模轻量用例："
            f"need {required}, have {len(prepared_records)}"
        )
    offset = (repetition * required) % len(prepared_records)
    selected = (prepared_records[offset:] + prepared_records[:offset])[:required]
    records = {identifier: [] for identifier in identifiers}
    remaining = list(requested_duplicates)
    cursor = 0
    while any(remaining):
        candidates = sorted(
            (index for index, count in enumerate(remaining) if count > 0),
            key=lambda index: (-remaining[index], index),
        )
        if len(candidates) < 2:
            raise RuntimeError(
                "pairwise duplicate allocation is inconsistent / 两两重复分配不一致"
            )
        first, second = candidates[:2]
        text = selected[cursor].text
        cursor += 1
        records[identifiers[first]].append(text)
        records[identifiers[second]].append(text)
        remaining[first] -= 1
        remaining[second] -= 1
    for index, identifier in enumerate(identifiers):
        unique_count = case.records_per_client - requested_duplicates[index]
        records[identifier].extend(
            record.text for record in selected[cursor:cursor + unique_count]
        )
        cursor += unique_count
    if cursor != required:
        raise RuntimeError("prepared-data cursor mismatch / 预处理数据游标不匹配")
    return records


def _validate_precomputed_oprf_cases(
    plan: EvaluationPlan,
    cases: Sequence[_Case],
) -> None:
    """Reject a sweep that would silently abandon the fixed cached allocation.

    拒绝会静默放弃固定缓存分配的扫描。

    The one-time cache intentionally represents only the agreed 10-client,
    1024-record, r=0.3 baseline. Variable-scale or variable-ratio suites must
    run with this feature closed; otherwise they would no longer measure their
    stated independent variable. 一次性缓存只表示约定的 10 客户端、1024 条、
    r=0.3 基线。可变规模或可变重复率套件必须关闭该功能，否则将不再测量其声称的
    独立变量。
    """
    if plan.training_mode != "gpt":
        raise ValueError("precomputed OPRF requires real prepared-data GPT mode / 预计算 OPRF 需要真实预处理数据 GPT 模式")
    incompatible = [
        case for case in cases
        if case.clients != 10 or case.records_per_client != 1024 or case.duplicate_ratio != 0.30
    ]
    if incompatible:
        suites = sorted({case.suite for case in incompatible})
        raise ValueError(
            "precomputed OPRF supports only fixed 10-client, 1024-record, r=0.3 cases; "
            f"disable it for suites: {suites} / 预计算 OPRF 仅支持固定 10 客户端、"
            f"1024 条记录、r=0.3 用例；请为以下套件关闭它：{suites}"
        )


def _fetch_as_metrics(as_url: str) -> dict[str, Any]:
    """Read the dedicated non-sensitive AS evaluation snapshot.

    读取专用且不含敏感数据的 AS 评估快照。
    """
    response = JsonHttpClient(as_url).send(AggregationServerPath.METRICS.value,
                                           WireMessage.create(AS_METRICS_REQUEST, {}))
    if response.message_type != AS_METRICS_RESPONSE:
        raise RuntimeError("AS returned an unexpected metrics response / AS 返回了意外指标响应")
    return dict(response.payload)


def _try_fetch_as_metrics(as_url: str) -> dict[str, Any] | None:
    """Return an AS metrics snapshot without changing protocol success.

    尽力读取 AS 指标快照，但不改变协议成功状态。

    Metrics are reporting evidence, whereas heartbeat instructions are the
    authoritative recovery protocol response. A transient control-plane
    transport failure must leave the corresponding optional report fields
    unavailable rather than converting a completed takeover into a failed
    experiment. 指标属于报告证据，心跳指令才是恢复协议的权威响应。瞬态控制面
    传输失败应仅使相应可选报告字段不可用，不能把已经完成的接管转成失败实验。
    """
    try:
        return _fetch_as_metrics(as_url)
    except CommunicationError:
        return None


def _fetch_ks_metrics(ks_url: str) -> dict[str, Any]:
    """Read only the KS private-key storage size needed for overhead reports.

    仅读取开销报告所需的 KS 私钥存储大小。
    """
    response = JsonHttpClient(ks_url).send(
        KeyServerPath.METRICS.value,
        WireMessage.create(KS_METRICS_REQUEST, {}),
    )
    if response.message_type != KS_METRICS_RESPONSE:
        raise RuntimeError("KS returned an unexpected metrics response / KS 返回了意外指标响应")
    return dict(response.payload)


def _metadata_sizes(root: Path, clients: list[ClientEntity], key_path: Path | None,
                    as_metrics: dict[str, Any], ks_metrics: dict[str, Any]) -> dict[str, int]:
    """Report protocol metadata separately from model artifacts and logs.

    将协议元数据与模型产物、日志分别报告，并给出透明总量。

    Checkpoints are deliberately not called metadata: they are model payloads.
    This split prevents a large Safetensors file from hiding the storage cost of
    protected labels, the native index, and KS key material. 检查点不被称作元数据，
    而是模型载荷；该划分避免大型 Safetensors 文件掩盖受保护标签、原生索引与 KS
    密钥材料的存储代价。
    """
    label_bytes = sum(client.config.label_store_path.stat().st_size for client in clients
                      if client.config.label_store_path.is_file())
    update_bytes = _directory_bytes(root / "model-updates")
    client_checkpoint_bytes = sum(
        item.stat().st_size
        for item in root.rglob("*.safetensors")
        if item.is_file()
    )
    training_metric_bytes = sum(
        item.stat().st_size for item in root.rglob("training_metrics.json") if item.is_file()
    )
    training_log_bytes = sum(
        item.stat().st_size for item in root.rglob("*.log") if item.is_file()
    )
    values = {
        "as_native_index_bytes": int(as_metrics["native_index_bytes"]),
        "as_accepted_update_bytes": int(as_metrics["accepted_model_update_bytes"]),
        "as_update_filesystem_bytes": update_bytes,
        "client_label_store_bytes": label_bytes,
        "client_checkpoint_bytes": client_checkpoint_bytes,
        "training_metric_bytes": training_metric_bytes,
        "training_log_bytes": training_log_bytes,
        "ks_private_key_bytes": (
            key_path.stat().st_size
            if key_path is not None and key_path.is_file()
            else int(ks_metrics["private_key_bytes"])
        ),
    }
    values["protocol_metadata_bytes"] = (
        values["as_native_index_bytes"]
        + values["client_label_store_bytes"]
        + values["ks_private_key_bytes"]
    )
    values["experiment_metadata_bytes"] = training_metric_bytes + training_log_bytes
    values["model_artifact_bytes"] = (
        values["as_update_filesystem_bytes"] + client_checkpoint_bytes
    )
    values["total_metadata_bytes"] = (
        values["protocol_metadata_bytes"] + values["experiment_metadata_bytes"]
    )
    values["total_reported_storage_bytes"] = (
        values["total_metadata_bytes"] + values["model_artifact_bytes"]
    )
    return values


def _communication_metrics(
    clients: Sequence[ClientEntity],
    ks_metrics: Mapping[str, Any],
) -> dict[str, dict[str, int | float]]:
    """Measure actual client protocol bodies without counting model bytes as metadata.

    测量实际客户端协议报文体，且绝不将模型字节计为元数据。

    Every byte value is the serialized UTF-8 JSON body observed at the client
    transport boundary. TCP/IP and HTTP headers are deliberately excluded so
    that isolated and remote deployments use the same accounting unit. 每个字节
    值均为客户端传输边界观测到的序列化 UTF-8 JSON 报文体；刻意不统计 TCP/IP
    及 HTTP 头，以便隔离部署与远程部署采用相同口径。
    """
    observations = tuple(
        observation
        for client in clients
        if client.config.traffic_recorder is not None
        for observation in client.config.traffic_recorder.snapshot()
    )
    model_paths = {
        AggregationServerPath.SUBMIT_MODEL_UPDATE.value,
        AggregationServerPath.AGGREGATE_MODEL_UPDATES.value,
        AggregationServerPath.DOWNLOAD_GLOBAL_MODEL.value,
    }
    oprf_path = KeyServerPath.EVALUATE_OPRF.value
    heartbeat_path = AggregationServerPath.HEARTBEAT.value
    oprf = tuple(item for item in observations if item.path == oprf_path)
    model = tuple(item for item in observations if item.path in model_paths)
    protocol = tuple(item for item in observations if item.path not in model_paths)
    as_control = tuple(item for item in protocol if item.path != oprf_path)
    heartbeats = tuple(item for item in as_control if item.path == heartbeat_path)

    def request_bytes(items: Sequence[Any]) -> int:
        """Return observed request JSON bytes. / 返回观测到的请求 JSON 字节数。"""
        return sum(int(item.request_body_bytes) for item in items)

    def response_bytes(items: Sequence[Any]) -> int:
        """Return observed response JSON bytes. / 返回观测到的响应 JSON 字节数。"""
        return sum(int(item.response_body_bytes) for item in items)

    def both_directions(items: Sequence[Any]) -> int:
        """Return request plus response JSON bytes. / 返回请求与响应 JSON 字节总和。"""
        return request_bytes(items) + response_bytes(items)

    bytes_result = {
        "oprf_request_body_bytes": request_bytes(oprf),
        "oprf_response_body_bytes": response_bytes(oprf),
        "oprf_communication_bytes": both_directions(oprf),
        "as_control_request_body_bytes": request_bytes(as_control),
        "as_control_response_body_bytes": response_bytes(as_control),
        "as_control_communication_bytes": both_directions(as_control),
        "heartbeat_communication_bytes": both_directions(heartbeats),
        "protocol_metadata_communication_bytes": both_directions(protocol),
        "model_transport_communication_bytes": both_directions(model),
        "total_client_service_communication_bytes": both_directions(observations),
    }
    timing_result = {
        "oprf_rpc_accumulated_seconds": sum(float(item.elapsed_seconds) for item in oprf),
        "ks_oprf_evaluation_compute_seconds": float(
            ks_metrics.get("oprf_evaluation_compute_seconds", 0.0)
        ),
    }
    counts_result = {
        "oprf_http_exchange_count": len(oprf),
        "as_control_http_exchange_count": len(as_control),
        "heartbeat_http_exchange_count": len(heartbeats),
        "model_http_exchange_count": len(model),
    }
    return {
        "bytes": bytes_result,
        "timing_seconds": timing_result,
        "counts": counts_result,
    }


def _write_reports(plan: EvaluationPlan, results: list[dict[str, object]]) -> Path:
    """Persist final case aggregates and concise bilingual summaries.

    持久化最终用例聚合结果与简洁的中英文摘要。
    """
    root = plan.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _apply_dynamic_join_cost(results)
    _write_text_atomically(
        root / "results.json",
        json.dumps(_json_safe({"plan": _json_plan(plan), "results": results}), ensure_ascii=False,
                   indent=2, sort_keys=True) + "\n",
    )
    failures = [result for result in results if result.get("status") == "failed"]
    _write_text_atomically(
        root / "failed_cases.json",
        json.dumps(_json_safe({"failed_cases": failures}), ensure_ascii=False, indent=2,
                   sort_keys=True) + "\n",
    )
    fields = [
        "status", "suite", "variable", "value", "repetition", "training_mode",
              "failure_type", "failure_message", "total_completion_seconds",
              "dedup_wall_seconds", "dedup_arrival_inclusive_wall_seconds",
              "dedup_active_interval_wall_seconds", "dedup_accumulated_seconds", "training_wall_seconds",
              "training_accumulated_seconds", "recovery_latency_seconds", "submitted_client_count",
              "ownership_retrain_count", "federated_rounds", "model_upload_accumulated_seconds",
              "model_upload_wall_seconds",
              "incremental_cost_seconds"]
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=fields)
    writer.writeheader()
    for result in results:
        failure = dict(result.get("failure", {}))
        row = {name: result.get(name) for name in fields}
        row["status"] = result.get("status", "completed")
        row["failure_type"] = failure.get("type")
        row["failure_message"] = failure.get("message")
        writer.writerow(row)
    _write_text_atomically(root / "case_metrics.csv", csv_stream.getvalue())
    _write_client_metrics_csv(root, results)
    _write_arrival_metrics_csv(root, results)
    _write_dynamic_join_metrics_csv(root, results)
    _write_metadata_csv(root, results)
    _write_scalability_metrics_csv(root, results)
    _write_markdown_report(root / "REPORT.en.md", results, "en")
    _write_markdown_report(root / "REPORT.zh-CN.md", results, "zh")
    return root


def _write_scalability_metrics_csv(root: Path, results: Sequence[dict[str, object]]) -> None:
    """Write one joined Exp#1 table with time, storage, and communication values.

    写出一张合并时间、存储与通信数值的实验 #1 表。
    """
    fields = [
        "status", "clients", "records_per_client", "input_record_count",
        "duplicate_ratio", "backend_workers", "scheduled_arrival_min_delay_seconds",
        "scheduled_arrival_max_delay_seconds", "scheduled_arrival_delay_sum_seconds",
        "observed_first_connection_after_seconds", "observed_arrival_span_seconds",
        "total_completion_seconds", "dedup_wall_seconds",
        "dedup_arrival_inclusive_wall_seconds", "dedup_active_interval_wall_seconds",
        "dedup_throughput_records_per_second", "dedup_active_interval_throughput_records_per_second",
        "as_native_index_bytes", "client_label_store_bytes", "ks_private_key_bytes",
        "protocol_metadata_storage_bytes", "oprf_communication_bytes",
        "as_control_communication_bytes", "heartbeat_communication_bytes",
        "protocol_metadata_communication_bytes", "model_transport_communication_bytes",
        "ks_oprf_evaluation_compute_seconds", "oprf_rpc_accumulated_seconds",
        "oprf_http_exchange_count", "as_task_count", "as_owner_edge_count",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for result in results:
        if result.get("suite") != "parallel_client_scale":
            continue
        configuration = dict(result.get("configuration") or {})
        metadata = dict(result.get("metadata_bytes") or {})
        communication = dict(result.get("communication_bytes") or {})
        timing = dict(result.get("communication_timing_seconds") or {})
        counts = dict(result.get("metadata_counts") or {})
        arrivals = result.get("client_arrivals", {})
        connected_after = [
            float(timing["connected_after_seconds"])
            for timing in arrivals.values()
            if isinstance(timing, Mapping) and isinstance(
                timing.get("connected_after_seconds"), (int, float)
            )
        ] if isinstance(arrivals, Mapping) else []
        scheduled_delays = [
            float(timing["scheduled_delay_seconds"])
            for timing in arrivals.values()
            if isinstance(timing, Mapping) and isinstance(
                timing.get("scheduled_delay_seconds"), (int, float)
            )
        ] if isinstance(arrivals, Mapping) else []
        dedup_seconds = result.get("dedup_wall_seconds")
        active_interval_seconds = result.get("dedup_active_interval_wall_seconds")
        input_records = result.get("input_record_count")
        throughput = (
            float(input_records) / float(dedup_seconds)
            if isinstance(input_records, int) and isinstance(dedup_seconds, (int, float)) and dedup_seconds > 0
            else None
        )
        active_interval_throughput = (
            float(input_records) / float(active_interval_seconds)
            if isinstance(input_records, int)
            and isinstance(active_interval_seconds, (int, float))
            and active_interval_seconds > 0
            else None
        )
        writer.writerow({
            "status": result.get("status", "completed"),
            "clients": configuration.get("clients"),
            "records_per_client": configuration.get("records_per_client"),
            "input_record_count": input_records,
            "duplicate_ratio": configuration.get("duplicate_ratio"),
            "backend_workers": configuration.get("backend_workers"),
            "scheduled_arrival_min_delay_seconds": min(scheduled_delays, default=0.0),
            "scheduled_arrival_max_delay_seconds": max(scheduled_delays, default=0.0),
            "scheduled_arrival_delay_sum_seconds": sum(scheduled_delays),
            "observed_first_connection_after_seconds": min(connected_after, default=0.0),
            "observed_arrival_span_seconds": (
                max(connected_after) - min(connected_after)
                if len(connected_after) > 1 else 0.0
            ),
            "total_completion_seconds": result.get("total_completion_seconds"),
            "dedup_wall_seconds": dedup_seconds,
            "dedup_arrival_inclusive_wall_seconds": result.get(
                "dedup_arrival_inclusive_wall_seconds"
            ),
            "dedup_active_interval_wall_seconds": active_interval_seconds,
            "dedup_throughput_records_per_second": throughput,
            "dedup_active_interval_throughput_records_per_second": active_interval_throughput,
            "as_native_index_bytes": metadata.get("as_native_index_bytes"),
            "client_label_store_bytes": metadata.get("client_label_store_bytes"),
            "ks_private_key_bytes": metadata.get("ks_private_key_bytes"),
            "protocol_metadata_storage_bytes": metadata.get("protocol_metadata_bytes"),
            "oprf_communication_bytes": communication.get("oprf_communication_bytes"),
            "as_control_communication_bytes": communication.get("as_control_communication_bytes"),
            "heartbeat_communication_bytes": communication.get("heartbeat_communication_bytes"),
            "protocol_metadata_communication_bytes": communication.get("protocol_metadata_communication_bytes"),
            "model_transport_communication_bytes": communication.get("model_transport_communication_bytes"),
            "ks_oprf_evaluation_compute_seconds": timing.get("ks_oprf_evaluation_compute_seconds"),
            "oprf_rpc_accumulated_seconds": timing.get("oprf_rpc_accumulated_seconds"),
            "oprf_http_exchange_count": counts.get("oprf_http_exchange_count"),
            "as_task_count": counts.get("as_task_count"),
            "as_owner_edge_count": counts.get("as_owner_edge_count"),
        })
    _write_text_atomically(root / "exp1_scalability_metrics.csv", stream.getvalue())


def _write_client_metrics_csv(root: Path, results: Sequence[dict[str, object]]) -> None:
    """Flatten per-client real training and GPU observations into a CSV.

    将每客户端真实训练和 GPU 观测展平为 CSV。
    """
    fields = [
        "suite", "variable", "value", "repetition", "client_id", "input_record_count",
        "unique_input_record_count", "hot_record_count", "cold_record_count", "sample_count",
        "optimizer_steps", "mean_loss", "elapsed_seconds", "device", "cuda_available",
        "cuda_name", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
        "physical_gpu_id", "slot_index", "gpu_memory_fraction", "subprocess_elapsed_seconds",
        "mps_partitioning_enabled", "mps_active_thread_percentage",
        "timed_out",
        "submitted", "upload_elapsed_seconds", "trained_task_count", "trained_sample_count",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for result in results:
        for client in result.get("client_metrics", []):
            training = dict(client.get("training", {}))
            cuda = dict(training.get("cuda", {}))
            scheduler = dict(training.get("scheduler", {}))
            submission = dict(client.get("submission", {}))
            writer.writerow({
                "suite": result.get("suite"), "variable": result.get("variable"),
                "value": result.get("value"), "repetition": result.get("repetition"),
                "client_id": client.get("client_id"),
                "input_record_count": client.get("input_record_count"),
                "unique_input_record_count": client.get("unique_input_record_count"),
                "hot_record_count": client.get("hot_record_count"),
                "cold_record_count": client.get("cold_record_count"),
                "sample_count": training.get("sample_count"),
                "optimizer_steps": training.get("optimizer_steps"),
                "mean_loss": training.get("mean_loss"),
                "elapsed_seconds": training.get("elapsed_seconds"),
                "device": training.get("device"), "cuda_available": cuda.get("available"),
                "cuda_name": cuda.get("name"),
                "cuda_peak_allocated_bytes": cuda.get("peak_allocated_bytes"),
                "cuda_peak_reserved_bytes": cuda.get("peak_reserved_bytes"),
                "physical_gpu_id": scheduler.get("physical_gpu_id"),
                "slot_index": scheduler.get("slot_index"),
                "gpu_memory_fraction": scheduler.get("gpu_memory_fraction"),
                "mps_partitioning_enabled": scheduler.get("mps_partitioning_enabled"),
                "mps_active_thread_percentage": scheduler.get("mps_active_thread_percentage"),
                "timed_out": scheduler.get("timed_out"),
                "subprocess_elapsed_seconds": scheduler.get("subprocess_elapsed_seconds"),
                "submitted": submission.get("submitted", False),
                "upload_elapsed_seconds": submission.get("upload_elapsed_seconds"),
                "trained_task_count": submission.get("trained_task_count"),
                "trained_sample_count": submission.get("trained_sample_count"),
            })
    _write_text_atomically(root / "client_metrics.csv", stream.getvalue())


def _write_arrival_metrics_csv(root: Path, results: Sequence[dict[str, object]]) -> None:
    """Persist the planned and observed asynchronous client arrival timeline.

    持久化计划的与实际观测到的异步客户端上线时间线。
    """
    fields = [
        "suite", "variable", "value", "repetition", "client_id",
        "scheduled_delay_seconds", "connected_after_seconds",
        "connect_elapsed_seconds", "first_protocol_started_after_seconds",
        "registration_elapsed_seconds", "claim_elapsed_seconds",
        "protocol_completed_after_seconds",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for result in results:
        arrivals = result.get("client_arrivals", {})
        if not isinstance(arrivals, Mapping):
            continue
        for client_id, timing in arrivals.items():
            if not isinstance(timing, Mapping):
                continue
            writer.writerow({
                "suite": result.get("suite"),
                "variable": result.get("variable"),
                "value": result.get("value"),
                "repetition": result.get("repetition"),
                "client_id": client_id,
                **{field: timing.get(field) for field in fields[5:]},
            })
    _write_text_atomically(root / "client_arrival_metrics.csv", stream.getvalue())


def _write_dynamic_join_metrics_csv(
    root: Path,
    results: Sequence[dict[str, object]],
) -> None:
    """Write late-join protocol time beside the matched end-to-end result.

    将后加入协议时间与匹配的端到端结果一并写出。

    The join protocol runs beside established training, so its local protocol
    duration must not be added to the end-to-end critical path. This table
    records both values explicitly and leaves ``incremental_cost_seconds`` to
    the matched base/join calculation required by the paper. 加入协议会与既有
    训练并行运行，因此其局部协议时长不能直接累加到端到端关键路径。本表明确记录二者，
    而论文定义的 ``incremental_cost_seconds`` 仍由匹配的基础/加入用例计算。
    """
    fields = [
        "status", "suite", "repetition", "joining_client_count",
        "join_dedup_wall_seconds", "join_dedup_accumulated_seconds",
        "joining_oprf_wall_seconds", "joining_oprf_accumulated_seconds",
        "total_completion_seconds", "incremental_cost_seconds",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for result in results:
        if result.get("suite") not in {"dynamic_join_base", "dynamic_join"}:
            continue
        protocol = dict(result.get("dynamic_join_protocol") or {})
        writer.writerow({
            "status": result.get("status", "completed"),
            "suite": result.get("suite"),
            "repetition": result.get("repetition"),
            "joining_client_count": protocol.get("joining_client_count", 0),
            "join_dedup_wall_seconds": protocol.get("dedup_wall_seconds", 0.0),
            "join_dedup_accumulated_seconds": protocol.get(
                "dedup_accumulated_seconds", 0.0
            ),
            "joining_oprf_wall_seconds": protocol.get("joining_oprf_wall_seconds", 0.0),
            "joining_oprf_accumulated_seconds": protocol.get(
                "joining_oprf_accumulated_seconds", 0.0
            ),
            "total_completion_seconds": result.get("total_completion_seconds"),
            "incremental_cost_seconds": result.get("incremental_cost_seconds"),
        })
    _write_text_atomically(root / "dynamic_join_metrics.csv", stream.getvalue())


def _write_metadata_csv(root: Path, results: Sequence[dict[str, object]]) -> None:
    """Flatten byte and count overheads so every auxiliary value is reportable.

    展平字节和计数开销，确保每项辅助值均可报告。
    """
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=[
        "suite", "variable", "value", "repetition", "metric", "amount", "unit",
    ])
    writer.writeheader()
    for result in results:
        common = {key: result.get(key) for key in ("suite", "variable", "value", "repetition")}
        for metric, amount in dict(result.get("metadata_bytes", {})).items():
            writer.writerow({**common, "metric": metric, "amount": amount, "unit": "bytes"})
        for metric, amount in dict(result.get("communication_bytes", {})).items():
            writer.writerow({**common, "metric": metric, "amount": amount, "unit": "bytes"})
        for metric, amount in dict(result.get("communication_timing_seconds", {})).items():
            writer.writerow({**common, "metric": metric, "amount": amount, "unit": "seconds"})
        for metric, amount in dict(result.get("metadata_counts", {})).items():
            writer.writerow({**common, "metric": metric, "amount": amount, "unit": "count"})
    _write_text_atomically(root / "metadata_metrics.csv", stream.getvalue())


def _apply_dynamic_join_cost(results: list[dict[str, object]]) -> None:
    """Attach the paper's T_join - T_base value to aggregate results.

    向动态加入聚合结果附加论文定义的 T_join - T_base 数值。
    """
    baselines = [
        float(result["total_completion_seconds"])
        for result in results
        if result.get("status", "completed") == "completed"
        and result["suite"] == "dynamic_join_base"
        and isinstance(result.get("total_completion_seconds"), (int, float))
    ]
    baseline = baselines[0] if len(baselines) == 1 else None
    for result in results:
        if result.get("status", "completed") == "completed" and result["suite"] == "dynamic_join":
            result["incremental_cost_seconds"] = (
                None if baseline is None else float(result["total_completion_seconds"]) - baseline
            )


def _write_markdown_report(path: Path, results: list[dict[str, object]], language: str) -> None:
    """Create a reviewer-readable result table without inventing conclusions.

    创建便于审阅的结果表，但绝不虚构结论。
    """
    title = "# DwT-FL Evaluation Results" if language == "en" else "# DwT-FL 评估结果"
    note = (
        "Each row is the final case aggregate. Formal four-run cases discard the fastest and slowest "
        "end-to-end observations and average the middle two; simulated mode is protocol-only."
            if language == "en" else "每行对应最终用例聚合值。正式五次运行会剔除端到端最快与最慢观测，"
            "并平均中间三次；模拟模式仅用于协议测试。")
    completed_results = [
        result for result in results if result.get("status", "completed") == "completed"
    ]
    failed_results = [result for result in results if result.get("status") == "failed"]
    rows = [
        title,
        "",
        note,
        "",
        "| Suite | Value | Rounds | Total s | Dedup wall/acc. s | Training wall/acc. s | "
        "Upload acc. s | Recovery s | Retrains | Protocol storage B | Protocol communication B | Model transport B | Model artifacts B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in completed_results:
        metadata = result["metadata_bytes"]
        template = (
            "| {suite} | {value} | {rounds} | {total:.6f} | {dw:.6f}/{da:.6f} | "
            "{tw:.6f}/{ta:.6f} | {upload} | {recovery} | {retrains} | {metadata} | {communication} | {model_transport} | {artifacts} |"
        )
        rows.append(template.format(
            suite=result["suite"], value=result["value"], rounds=result.get("federated_rounds", 1),
            total=result["total_completion_seconds"],
            dw=result["dedup_wall_seconds"], da=result["dedup_accumulated_seconds"],
            tw=result["training_wall_seconds"], ta=result["training_accumulated_seconds"],
            recovery=(
                "N/A" if result["recovery_latency_seconds"] is None
                else f"{result['recovery_latency_seconds']:.6f}"
            ),
            retrains=result.get("ownership_retrain_count", 0),
            upload=f"{result.get('model_upload_accumulated_seconds', 0.0):.6f}",
            metadata=metadata["protocol_metadata_bytes"] if "protocol_metadata_bytes" in metadata else metadata["total_metadata_bytes"],
            communication=dict(result.get("communication_bytes", {})).get("protocol_metadata_communication_bytes", "N/A"),
            model_transport=dict(result.get("communication_bytes", {})).get("model_transport_communication_bytes", "N/A"),
            artifacts=metadata.get("model_artifact_bytes", "N/A"),
        ))
    if failed_results:
        rows.extend([
            "",
            "## Failed cases" if language == "en" else "## 失败用例",
            "",
            (
                "These cases were isolated, cleaned up, and skipped; their raw traceback is in "
                "`failed_cases.json`. No metric values are inferred for a failed case."
                if language == "en" else
                "这些用例已隔离、清理并跳过；其原始 traceback 位于 `failed_cases.json`。"
                "不会为失败用例推断任何指标值。"
            ),
            "",
            "| Suite | Variable | Value | Repetition | Error type | Error message |"
            if language == "en" else
            "| 套件 | 变量 | 取值 | 重复 | 错误类型 | 错误信息 |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ])
        for result in failed_results:
            failure = dict(result.get("failure", {}))
            rows.append(
                "| {suite} | {variable} | {value} | {repetition} | {kind} | {message} |".format(
                    suite=result.get("suite", "N/A"),
                    variable=result.get("variable", "N/A"),
                    value=result.get("value", "N/A"),
                    repetition=result.get("repetition", "N/A"),
                    kind=_markdown_cell(failure.get("type", "Unknown")),
                    message=_markdown_cell(failure.get("message", "")),
                )
            )
    _write_text_atomically(path, "\n".join(rows) + "\n")


def _markdown_cell(value: object) -> str:
    """Render one error value safely inside a Markdown table cell.

    在 Markdown 表格单元格中安全呈现一项错误值。
    """
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _write_run_status(
    root: Path,
    state: str,
    completed_cases: int,
    total_cases: int,
    results: list[dict[str, object]],
    failure: BaseException | None = None,
) -> None:
    """Persist progress and failure context independently of final reports.

    独立于最终报告持久化进度和失败上下文。
    """
    payload: dict[str, object] = {
        "state": state,
        "completed_cases": completed_cases,
        "total_cases": total_cases,
        "succeeded_cases": sum(
            result.get("status", "completed") == "completed" for result in results
        ),
        "failed_cases": sum(result.get("status") == "failed" for result in results),
        "updated_at_unix_seconds": time.time(),
    }
    if results:
        latest = results[-1]
        payload["last_processed_case"] = {
            "suite": latest["suite"],
            "variable": latest["variable"],
            "value": latest["value"],
            "aggregation": latest.get("aggregation"),
            "status": latest.get("status", "completed"),
        }
    if failure is not None:
        payload["failure"] = {
            "type": type(failure).__name__,
            "message": str(failure),
        }
    _write_text_atomically(
        root / "run_status.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_run_metadata(root: Path, plan: EvaluationPlan, prepared_data: Path | None) -> None:
    """Persist host, GPU, and dataset provenance before the first case starts.

    在首个用例开始前持久化主机、GPU 与数据集溯源。
    """
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "service_mode": plan.service_mode,
        "training_mode": plan.training_mode,
        "start_case": plan.start_case,
        "repeat_aggregation": {
            "configured_repetitions": plan.repetitions,
            "formal_policy": (
                "discard_fastest_and_slowest_then_average_middle_two_when_repetitions_is_four"
                if plan.repetitions == 4
                else "arithmetic_mean_diagnostic"
            ),
        },
        "training_job_timeout_seconds": plan.training_job_timeout_seconds,
        "gpu_ids_requested": list(plan.gpu_ids),
        "clients_per_gpu": plan.clients_per_gpu,
        "global_model_download_workers": plan.global_model_download_workers,
        "gpu_memory_fraction_per_client": plan.gpu_memory_fraction_per_client,
        "require_mps_partitioning": plan.require_mps_partitioning,
        "mps_partitioning": mps_partitioning_status(),
        "gpu_resource_policy": {
            "logical_slots_per_gpu": plan.clients_per_gpu,
            "pytorch_memory_fraction_per_client": plan.gpu_memory_fraction_per_client,
            "compute_partitioning": (
                "mps_active_thread_percentage"
                if plan.require_mps_partitioning
                else "shared_cuda_scheduler_not_hard_partitioned"
            ),
        },
        "global_model_distribution_policy": {
            "max_concurrent_downloads": plan.global_model_download_workers,
            "transport_retries": "read_only_global_model_chunks_only",
            "integrity_check": "sha256_after_complete_download",
        },
        "require_cuda": plan.require_cuda,
        "prepared_data_path": None if prepared_data is None else str(prepared_data),
        "gpus": _nvidia_smi_metadata(),
    }
    manifest_path = None if prepared_data is None else prepared_data.with_name("manifest.json")
    if manifest_path is not None and manifest_path.is_file():
        try:
            payload["prepared_dataset_manifest"] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            payload["prepared_dataset_manifest"] = {"status": "invalid_json"}
    try:
        import psutil

        payload["host_resources"] = {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory_bytes": psutil.virtual_memory().total,
        }
    except Exception as error:
        payload["host_resources"] = {"status": "unavailable", "reason": type(error).__name__}
    _write_text_atomically(
        root / "run_metadata.json",
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _nvidia_smi_metadata() -> dict[str, object]:
    """Collect read-only GPU identity without requiring NVIDIA tools on Windows.

    采集只读 GPU 身份信息，且不要求 Windows 上必定存在 NVIDIA 工具。
    """
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "reason": type(error).__name__}
    if completed.returncode != 0:
        return {"status": "unavailable", "reason": completed.stderr.strip() or "nvidia_smi_failed"}
    return {
        "status": "available",
        "devices": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
    }


def _write_text_atomically(path: Path, content: str) -> None:
    """Replace one text artifact atomically on Windows, WSL, and Ubuntu.

    在 Windows、WSL 和 Ubuntu 上原子替换一个文本产物。
    """
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="")
    temporary_path.replace(path)


def _json_plan(plan: EvaluationPlan) -> dict[str, object]:
    """Convert all plan values into portable JSON provenance.

    将所有计划值转换为可移植的 JSON 溯源信息。
    """
    return _json_safe(asdict(plan))


def _json_safe(value: Any) -> Any:
    """Recursively convert path objects before writing cross-platform JSON.

    在写入跨平台 JSON 前递归转换路径对象。
    """
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _directory_bytes(path: Path) -> int:
    """Return recursive file bytes without following links. / 返回递归文件字节数且不跟随链接。"""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _unused_port() -> int:
    """Reserve an ephemeral port number for immediate child-process startup.

    为立即启动的子进程保留一个临时端口号。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_as_ready(host: str, port: int, process: subprocess.Popen[bytes]) -> None:
    """Wait for a real AS response instead of only a listening socket.

    A successful TCP ``connect`` proves only that a socket is bound; it does
    not prove that the HTTP server has initialized its router. The evaluator
    must release concurrent client arrivals only after the read-only metrics
    endpoint responds. 单纯 TCP ``connect`` 成功只能证明套接字已绑定，不能证明
    HTTP 服务已初始化路由。评估器仅在只读 metrics 端点响应后才释放并发客户端。
    """
    deadline = time.monotonic() + 10
    base_url = f"http://{host}:{port}"
    last_error: CommunicationError | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("AS process exited during startup / AS 进程在启动时退出")
        try:
            _request_as_metrics(base_url, timeout_seconds=0.25)
            return
        except CommunicationError as error:
            last_error = error
        time.sleep(0.05)
    detail = "" if last_error is None else f": {type(last_error).__name__}"
    raise TimeoutError(
        "AS did not return a ready response before timeout / AS 未在超时前返回就绪响应"
        f"{detail}"
    )


def _request_as_metrics(base_url: str, *, timeout_seconds: float) -> Mapping[str, object]:
    """Issue one read-only AS health request and validate its response type.

    发起一次只读 AS 健康请求，并校验响应类型。
    """
    response = JsonHttpClient(base_url, timeout_seconds=timeout_seconds).send(
        AggregationServerPath.METRICS.value,
        WireMessage.create(AS_METRICS_REQUEST, {}),
    )
    if response.message_type != AS_METRICS_RESPONSE:
        raise RuntimeError(
            "AS metrics response type is invalid / AS metrics 响应类型无效"
        )
    return dict(response.payload)


def _library_suffix() -> str:
    """Return the platform native-library suffix. / 返回平台原生库后缀。"""
    return ".dll" if sys.platform == "win32" else ".so"
