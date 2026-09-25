"""Deployable Key Server entity for the DwT-FL OPRF service.

DwT-FL OPRF 服务的可部署密钥服务器实体。
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path

from dbtfl.communication import RequestRejected, ThreadedJsonServer, WireMessage
from dbtfl.communication.endpoints import KeyServerPath
from dbtfl.oprf import KeyServerOprfService, OprfKeyStore


KS_METRICS_REQUEST = "ks.evaluation.metrics.request"
KS_METRICS_RESPONSE = "ks.evaluation.metrics.response"


@dataclass(frozen=True, slots=True)
class KeyServerConfig:
    """Network and local-key configuration owned only by the KS process.

    仅由 KS 进程持有的网络与本地密钥配置。
    """

    key_path: Path
    host: str = "0.0.0.0"
    port: int = 18081
    advertised_host: str | None = None
    ssl_context: ssl.SSLContext | None = None

    def __post_init__(self) -> None:
        """Reject invalid deployment binding parameters early.

        尽早拒绝无效的部署绑定参数。
        """
        if not self.host:
            raise ValueError("KS host must not be empty / KS 主机不得为空")
        if not 0 <= self.port <= 65535:
            raise ValueError("KS port must be in 0..65535 / KS 端口必须位于 0..65535")


class KeyServerEntity:
    """Own a KS private scalar and expose only blind OPRF evaluation.

    持有 KS 私有标量，并且仅公开盲化 OPRF 求值。
    """

    def __init__(self, config: KeyServerConfig) -> None:
        """Load the host-local secret and bind, but do not start, the listener.

        加载仅主机本地的密钥并绑定监听器，但不启动服务。
        """
        self.config = config
        material = OprfKeyStore(config.key_path).load_or_create()
        self._oprf_service = KeyServerOprfService(material)
        self._oprf_service.router.add(
            "POST",
            KeyServerPath.METRICS.value,
            self.evaluation_metrics,
        )
        self._server = ThreadedJsonServer(
            config.host,
            config.port,
            self._oprf_service.router,
            ssl_context=config.ssl_context,
            advertised_host=config.advertised_host,
        )

    @property
    def base_url(self) -> str:
        """Return the active HTTP(S) endpoint without exposing KS key material.

        返回活动 HTTP(S) 端点，且不暴露 KS 密钥材料。
        """
        return self._server.base_url

    @property
    def port(self) -> int:
        """Return the actual bound port, including an OS-selected port zero.

        返回实际绑定端口，包括由操作系统选择的零端口。
        """
        return self._server.port

    def evaluation_metrics(self, message: WireMessage) -> WireMessage:
        """Return the private-key storage size without exposing key material.

        返回私钥存储大小，但绝不暴露密钥材料。
        """
        if message.message_type != KS_METRICS_REQUEST or dict(message.payload):
            raise RequestRejected(
                400,
                "invalid_ks_metrics_request",
                "KS metrics request must have an empty payload / KS 指标请求负载必须为空",
            )
        key_path = Path(self.config.key_path)
        metrics = self._oprf_service.metrics_snapshot()
        return WireMessage.create(
            KS_METRICS_RESPONSE,
            {
                "private_key_bytes": key_path.stat().st_size if key_path.is_file() else 0,
                "oprf_evaluated_element_count": metrics["evaluated_element_count"],
                "oprf_evaluation_compute_seconds": metrics["evaluation_compute_seconds"],
            },
            request_id=message.request_id,
        )

    def start(self) -> None:
        """Start the KS listener on its managed daemon thread.

        在受管理的守护线程上启动 KS 监听器。
        """
        self._server.start()

    def close(self) -> None:
        """Stop the KS listener and release its network resources.

        停止 KS 监听器并释放其网络资源。
        """
        self._server.close()

    def __enter__(self) -> "KeyServerEntity":
        """Start the entity when entering a managed lifetime.

        进入受管理生命周期时启动实体。
        """
        self.start()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Close the listener when leaving a managed lifetime.

        离开受管理生命周期时关闭监听器。
        """
        self.close()
