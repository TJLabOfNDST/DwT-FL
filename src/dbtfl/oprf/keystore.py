"""Persistent Ristretto255 secret handling for the KS OPRF service.

KS OPRF 服务的持久化 Ristretto255 私钥处理。
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .group14 import OprfValidationError
from .ristretto255 import OPRF_SUITE_IDENTIFIER, random_scalar, validate_scalar


KEY_FILE_VERSION: Final[int] = 2
"""Version for the native Ristretto255 KS key file. / 原生 Ristretto255 KS 密钥文件版本。"""

GROUP_IDENTIFIER: Final[str] = OPRF_SUITE_IDENTIFIER
"""Explicit cryptographic-suite binding for KS secrets. / KS 私钥的显式密码套件绑定。"""


class OprfKeyStoreError(RuntimeError):
    """Raised when the KS secret file does not meet the selected suite contract.

    当 KS 私钥文件不符合选定套件契约时引发。
    """


@dataclass(frozen=True, slots=True)
class OprfKeyMaterial:
    """Validated 32-byte Ristretto255 scalar held only by the KS process.

    仅由 KS 进程持有的、已验证的 32 字节 Ristretto255 标量。
    """

    scalar: bytes

    def __post_init__(self) -> None:
        """Reject malformed scalar material at the persistence trust boundary.

        在持久化信任边界拒绝格式错误的标量材料。
        """
        validate_scalar(self.scalar)


class OprfKeyStore:
    """Load one Ristretto255 secret or atomically create a new local secret.

    加载一份 Ristretto255 私钥，或原子地创建一份新的本地私钥。
    """

    def __init__(self, path: str | Path) -> None:
        """Bind the store to one concrete local filesystem path.

        将存储绑定到一个具体的本地文件系统路径。
        """
        self.path = Path(path)

    def load_or_create(self) -> OprfKeyMaterial:
        """Load a valid key and never replace malformed or legacy material silently.

        加载有效密钥，绝不静默替换格式错误或旧版材料。
        """
        if self.path.exists():
            return self._load_existing()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        material = OprfKeyMaterial(random_scalar())
        serialized = self._serialize(material)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return self._load_existing()
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(serialized)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return material

    def _load_existing(self) -> OprfKeyMaterial:
        """Load and validate an existing Ristretto255 key without changing it.

        加载并验证已有 Ristretto255 密钥，不修改文件。
        """
        if self.path.is_symlink():
            raise OprfKeyStoreError("KS key path must not be a symlink / KS 密钥路径不得为符号链接")
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OprfKeyStoreError("cannot read KS key file / 无法读取 KS 密钥文件") from error
        if not isinstance(parsed, dict):
            raise OprfKeyStoreError("unsupported KS key-file schema / 不支持的 KS 密钥文件模式")
        if parsed.get("version") == 1:
            raise OprfKeyStoreError(
                "legacy MODP KS key cannot be used with Ristretto255; configure a new key path / "
                "旧 MODP KS 密钥不能用于 Ristretto255；请配置新的密钥路径"
            )
        if parsed.get("version") != KEY_FILE_VERSION or parsed.get("suite") != GROUP_IDENTIFIER:
            raise OprfKeyStoreError("KS key is bound to another OPRF suite / KS 密钥绑定到另一 OPRF 套件")
        serialized_scalar = parsed.get("k_b64")
        if not isinstance(serialized_scalar, str):
            raise OprfKeyStoreError("KS private scalar is invalid / KS 私有标量无效")
        try:
            scalar = base64.b64decode(serialized_scalar.encode("ascii"), validate=True)
            return OprfKeyMaterial(scalar)
        except (UnicodeEncodeError, ValueError, binascii.Error, OprfValidationError) as error:
            raise OprfKeyStoreError("KS private scalar is invalid / KS 私有标量无效") from error

    @staticmethod
    def _serialize(material: OprfKeyMaterial) -> str:
        """Serialize a suite-bound key without exposing derived public material.

        序列化绑定套件的密钥，但不暴露派生公钥材料。
        """
        return json.dumps(
            {
                "version": KEY_FILE_VERSION,
                "suite": GROUP_IDENTIFIER,
                "k_b64": base64.b64encode(material.scalar).decode("ascii"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
