"""Paper-compatible OPRF arithmetic in the RFC 3526 group 14 subgroup.

论文兼容的 RFC 3526 第 14 组子群 OPRF 运算。
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Final


# RFC 3526, section 3: 2048-bit MODP group (group 14).  Squaring a non-zero
# field element maps it into the prime-order quadratic-residue subgroup.
# RFC 3526 第 3 节：2048 位 MODP 组（第 14 组）。对非零域元素平方可将其映射到
# 素数阶的二次剩余子群。
MODP_GROUP14_PRIME: Final[int] = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08"
    "8A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D"
    "6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8"
    "A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961"
    "C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA"
    "18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5"
    "C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898"
    "FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
"""The RFC 3526 group 14 safe prime. / RFC 3526 第 14 组安全素数。"""

SUBGROUP_ORDER: Final[int] = (MODP_GROUP14_PRIME - 1) // 2
"""Prime order q of the quadratic-residue subgroup. / 二次剩余子群的素数阶 q。"""

ELEMENT_HEX_LENGTH: Final[int] = 512
"""Fixed lower-hex element encoding length. / 固定小写十六进制元素编码长度。"""

MAX_INPUT_BYTES: Final[int] = 65_535
"""Bound for one application record before hashing. / 哈希前单个应用记录的长度上限。"""

_HASH_DOMAIN: Final[bytes] = b"DwT-FL/OPRF/RFC3526-group14/H1/v1\x00"


class OprfValidationError(ValueError):
    """Raised for non-canonical OPRF inputs or subgroup elements.

    在 OPRF 输入或子群元素不规范时引发。
    """


def normalize_record(record: str | bytes) -> bytes:
    """Encode one client-only record deterministically and enforce its bound.

    确定性编码一条仅由客户端持有的记录并实施长度上限。
    """
    if isinstance(record, str):
        encoded = record.encode("utf-8")
    elif isinstance(record, bytes):
        encoded = record
    else:
        raise TypeError("record must be str or bytes / 记录必须为 str 或 bytes")
    if not encoded:
        raise OprfValidationError("record must not be empty / 记录不得为空")
    if len(encoded) > MAX_INPUT_BYTES:
        raise OprfValidationError(
            f"record exceeds {MAX_INPUT_BYTES} bytes / 记录超过 {MAX_INPUT_BYTES} 字节"
        )
    return encoded


def random_scalar() -> int:
    """Return a cryptographically random non-zero scalar in Z_q.

    返回 Z_q 中密码学随机的非零标量。
    """
    return secrets.randbelow(SUBGROUP_ORDER - 1) + 1


def validate_scalar(value: int) -> int:
    """Validate a private or blinding scalar without normalizing it.

    验证私有或盲化标量，不对其进行归一化。
    """
    if not isinstance(value, int) or not 1 <= value < SUBGROUP_ORDER:
        raise OprfValidationError("scalar is outside Z_q* / 标量不在 Z_q* 内")
    return value


def hash_to_subgroup(record: str | bytes) -> int:
    """Compute the paper's H(m) as a domain-separated subgroup element.

    将论文中的 H(m) 计算为带域分离的子群元素。

    SHAKE-256 rejection sampling avoids modulo bias when mapping to the field.
    SHAKE-256 拒绝采样避免映射到有限域时出现模约减偏差。
    """
    encoded = normalize_record(record)
    framed_record = _HASH_DOMAIN + len(encoded).to_bytes(4, "big") + encoded
    counter = 0
    while True:
        digest = hashlib.shake_256(
            framed_record + counter.to_bytes(4, "big")
        ).digest(ELEMENT_HEX_LENGTH // 2)
        candidate = int.from_bytes(digest, "big")
        if 1 < candidate < MODP_GROUP14_PRIME - 1:
            element = pow(candidate, 2, MODP_GROUP14_PRIME)
            if element != 1:
                return element
        counter += 1
        if counter >= 2**32:
            raise RuntimeError(
                "hash-to-subgroup counter exhausted / 哈希到子群的计数器耗尽"
            )


def is_subgroup_element(element: int) -> bool:
    """Return whether an integer is a non-identity member of the q subgroup.

    返回一个整数是否为 q 阶子群中的非单位元。
    """
    return (
        isinstance(element, int)
        and 1 < element < MODP_GROUP14_PRIME - 1
        and pow(element, SUBGROUP_ORDER, MODP_GROUP14_PRIME) == 1
    )


def encode_element(element: int) -> str:
    """Return one canonical lower-hex subgroup encoding for the native index.

    为原生索引返回一个规范的小写十六进制子群编码。
    """
    if not is_subgroup_element(element):
        raise OprfValidationError(
            "value is not a valid subgroup element / 值不是有效子群元素"
        )
    return f"{element:0{ELEMENT_HEX_LENGTH}x}"


def decode_element(encoded_element: str) -> int:
    """Decode and validate one canonical lower-hex subgroup encoding.

    解码并验证一个规范的小写十六进制子群编码。
    """
    is_canonical = (
        isinstance(encoded_element, str)
        and len(encoded_element) == ELEMENT_HEX_LENGTH
    )
    if not is_canonical or any(
        character not in "0123456789abcdef" for character in encoded_element
    ):
        raise OprfValidationError(
            "element must contain exactly 512 lowercase hexadecimal characters / "
            "元素必须恰好包含 512 个小写十六进制字符"
        )
    element = int(encoded_element, 16)
    if not is_subgroup_element(element):
        raise OprfValidationError("element is outside the q subgroup / 元素不在 q 阶子群内")
    return element
