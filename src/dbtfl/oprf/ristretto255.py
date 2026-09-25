"""Constant-time Ristretto255 helpers for the DwT-FL blind OPRF.

用于 DwT-FL 盲 OPRF 的恒定时间 Ristretto255 辅助函数。

The module deliberately accepts only the ``oblivious.ristretto.sodium`` path,
which delegates group operations to libsodium.  It never falls back to the
package's pure-Python arithmetic: a silent fallback would make a deployment's
throughput hardware-dependent and would reintroduce the original bottleneck.
本模块刻意只接受 ``oblivious.ristretto.sodium`` 路径，它将群运算交给
libsodium。绝不回退到该包的纯 Python 算术：静默回退会使部署吞吐量依赖硬件，
并重新引入原有瓶颈。
"""

from __future__ import annotations

import hashlib
from typing import Any, Final

from .group14 import MAX_INPUT_BYTES, OprfValidationError, normalize_record


OPRF_SUITE_IDENTIFIER: Final[str] = "ristretto255-sha512-libsodium-v1"
"""Suite identifier bound to keys and client-local label stores. / 绑定密钥和客户端标签存储的套件标识。"""

LEGACY_OPRF_SUITE_IDENTIFIER: Final[str] = "rfc3526-modp-group14-legacy-v1"
"""Read-only identifier for pre-migration MODP label stores. / 迁移前 MODP 标签存储的只读标识。"""

POINT_BYTES: Final[int] = 32
SCALAR_BYTES: Final[int] = 32
_HASH_DOMAIN: Final[bytes] = b"DwT-FL/OPRF/ristretto255/SHA-512/H1/v1\x00"
_OUTPUT_DOMAIN: Final[bytes] = b"DwT-FL/OPRF/ristretto255/SHAKE-256/output/v1\x00"


class RistrettoBackendUnavailable(RuntimeError):
    """Raised when the required native libsodium Ristretto backend is absent.

    当所需的原生 libsodium Ristretto 后端不可用时引发。
    """


def native_backend_available() -> bool:
    """Return whether a libsodium-backed Ristretto implementation is loaded.

    返回是否已加载基于 libsodium 的 Ristretto 实现。
    """
    try:
        from oblivious.ristretto import sodium
    except ImportError:
        return False
    return sodium is not None


def _backend() -> Any:
    """Return the native backend or fail before handling any OPRF secret.

    返回原生后端，或在接触 OPRF 秘密前失败。
    """
    try:
        from oblivious.ristretto import sodium
    except ImportError as error:
        raise RistrettoBackendUnavailable(
            "the fast OPRF requires oblivious with a libsodium backend; install oblivious[rbcl] and libsodium / "
            "快速 OPRF 需要带 libsodium 后端的 oblivious；请安装 oblivious[rbcl] 和 libsodium"
        ) from error
    if sodium is None:
        raise RistrettoBackendUnavailable(
            "oblivious loaded without libsodium; refusing the slow pure-Python fallback / "
            "oblivious 未加载 libsodium；拒绝使用缓慢的纯 Python 回退"
        )
    return sodium


def random_scalar() -> bytes:
    """Generate one non-zero native Ristretto scalar. / 生成一个非零原生 Ristretto 标量。"""
    value = bytes(_backend().rnd())
    return validate_scalar(value)


def validate_scalar(value: bytes) -> bytes:
    """Validate canonical non-zero scalar bytes. / 验证规范的非零标量字节。"""
    if not isinstance(value, bytes) or len(value) != SCALAR_BYTES:
        raise OprfValidationError("Ristretto scalar must contain 32 bytes / Ristretto 标量必须包含 32 字节")
    try:
        scalar = _backend().scl(value)
    except (TypeError, ValueError) as error:
        raise OprfValidationError("Ristretto scalar encoding is invalid / Ristretto 标量编码无效") from error
    if scalar is None or bytes(scalar) != value or value == b"\x00" * SCALAR_BYTES:
        raise OprfValidationError("Ristretto scalar is not canonical and non-zero / Ristretto 标量不规范或为零")
    return value


def hash_to_group(record: str | bytes) -> bytes:
    """Hash one framed record to a prime-order Ristretto element.

    将一条带帧记录哈希到素数阶 Ristretto 元素。
    """
    encoded = normalize_record(record)
    # ``sodium.pnt`` maps a 64-byte uniform string to Ristretto. Framing keeps
    # records unambiguous before SHA-512. ``sodium.pnt`` 将 64 字节均匀字符串
    # 映射为 Ristretto；先加帧可在 SHA-512 前消除记录歧义。
    framed = _HASH_DOMAIN + len(encoded).to_bytes(4, "big") + encoded
    point = _backend().pnt(hashlib.sha512(framed).digest())
    if point is None:
        raise OprfValidationError("native Ristretto hash-to-group failed / 原生 Ristretto 哈希到群失败")
    return validate_point(bytes(point))


def validate_point(value: bytes) -> bytes:
    """Validate one canonical non-identity Ristretto point encoding.

    验证一个规范且非单位元的 Ristretto 点编码。
    """
    if not isinstance(value, bytes) or len(value) != POINT_BYTES:
        raise OprfValidationError("Ristretto point must contain 32 bytes / Ristretto 点必须包含 32 字节")
    try:
        point = _backend().point.from_bytes(value)
    except (TypeError, ValueError) as error:
        raise OprfValidationError("Ristretto point encoding is invalid / Ristretto 点编码无效") from error
    if point is None or bytes(point) != value or value == b"\x00" * POINT_BYTES:
        raise OprfValidationError("Ristretto point is not canonical and non-identity / Ristretto 点不规范或为单位元")
    return value


def scalar_multiply(scalar_bytes: bytes, point_bytes: bytes) -> bytes:
    """Compute a constant-time native scalar multiplication.

    计算一次恒定时间的原生标量乘法。
    """
    scalar = validate_scalar(scalar_bytes)
    point = validate_point(point_bytes)
    try:
        result = _backend().mul(scalar, point)
    except (TypeError, ValueError) as error:
        raise OprfValidationError("native Ristretto scalar multiplication failed / 原生 Ristretto 标量乘法失败") from error
    return validate_point(bytes(result))


def scalar_inverse(scalar_bytes: bytes) -> bytes:
    """Return the multiplicative inverse of one non-zero scalar.

    返回一个非零标量的乘法逆元。
    """
    scalar = validate_scalar(scalar_bytes)
    try:
        inverse = _backend().inv(scalar)
    except (TypeError, ValueError) as error:
        raise OprfValidationError("native Ristretto scalar inversion failed / 原生 Ristretto 标量求逆失败") from error
    return validate_scalar(bytes(inverse))


def encode_wire_point(value: bytes) -> str:
    """Encode one point compactly for the JSON compatibility envelope.

    为 JSON 兼容信封紧凑编码一个点。
    """
    import base64
    return base64.b64encode(validate_point(value)).decode("ascii")


def decode_wire_point(value: str) -> bytes:
    """Decode and validate one point from the JSON compatibility envelope.

    从 JSON 兼容信封解码并验证一个点。
    """
    import base64
    import binascii
    if not isinstance(value, str):
        raise OprfValidationError("Ristretto point wire value must be text / Ristretto 点线协议值必须是文本")
    try:
        return validate_point(base64.b64decode(value.encode("ascii"), validate=True))
    except (UnicodeEncodeError, ValueError, binascii.Error) as error:
        raise OprfValidationError("Ristretto point wire encoding is invalid / Ristretto 点线协议编码无效") from error


def protected_label(point_bytes: bytes) -> str:
    """Expand the 32-byte OPRF output to the native index's 512-hex contract.

    将 32 字节 OPRF 输出扩展为原生索引要求的 512 位十六进制标签。

    The expansion is domain-separated and preserves equality exactly; it is not
    a second OPRF evaluation. 扩展具有域分离且严格保持相等性；它不是第二次
    OPRF 求值。
    """
    return hashlib.shake_256(_OUTPUT_DOMAIN + validate_point(point_bytes)).hexdigest(256)


def validate_protected_label(label: str) -> str:
    """Validate the native-index-compatible expanded output label.

    验证兼容原生索引的扩展输出标签。
    """
    if not isinstance(label, str) or len(label) != 512 or any(character not in "0123456789abcdef" for character in label):
        raise OprfValidationError("protected label must contain 512 lowercase hexadecimal characters / 保护标签必须包含 512 个小写十六进制字符")
    return label
