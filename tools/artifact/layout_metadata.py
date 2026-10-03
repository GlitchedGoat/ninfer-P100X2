"""Pure byte-layout geometry used by framing, sharding, and exact converters."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import prod
import operator
import struct
import sys
from types import MappingProxyType
from typing import Sequence, TypeAlias


from .numeric import (
    DirectFormat,
    Fp8RowFormat,
    Fp8BlockFormat,
    Nvfp4Format,
    NumericFormat,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)


PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128
_PACK_TEMP_BYTES = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    alignment: int
    formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class RowSplitGeometry:
    n: int
    k: int
    k_pad: int
    groups_per_row: int
    base_bytes_per_group: int
    high_bytes_per_group: int
    base_row_bytes: int
    high_row_bytes: int
    scale_row_bytes: int
    base_offset: int
    base_bytes: int
    high_offset: int
    high_bytes: int
    scale_offset: int
    scale_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class BlockScaleGeometry:
    n: int
    k: int
    groups_per_row: int
    k_tiles: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    weight_divisor_offset: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowScaleGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class Fp8BlockGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


CONTIGUOUS_LE_V1 = Layout(
    "contiguous-le-v1", 256, frozenset(("BF16", "FP32", "I32"))
)
ROW_SPLIT_K128_V1 = Layout(
    "row-split-k128-v1",
    256,
    frozenset(("Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S", "W8G32_F16S")),
)
BLOCKSCALE_K16_M128X4_V1 = Layout(
    "blockscale-k16-m128x4-v1",
    256,
    frozenset(("NVFP4",)),
)
ROW_SCALE_V1 = Layout(
    "row-scale-v1",
    256,
    frozenset(("FP8_E4M3FN_ROW_BF16S",)),
)
GGML_K256_V1 = Layout("ggml-k256-v1", 256, frozenset(("GGML_K",)))
BLOCKSCALE_M128_K128_V1 = Layout(
    "blockscale-m128-k128-v1", 256, frozenset(("FP8_E4M3FN_BLOCK128_BF16S",))
)

LAYOUTS = MappingProxyType(
    {
        layout.name: layout
        for layout in (
            CONTIGUOUS_LE_V1,
            ROW_SPLIT_K128_V1,
            BLOCKSCALE_K16_M128X4_V1,
            ROW_SCALE_V1,
            GGML_K256_V1,
            BLOCKSCALE_M128_K128_V1,
        )
    }
)


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tensor layout: {name!r}") from None


def _format(value: str | NumericFormat) -> NumericFormat:
    if isinstance(value, str):
        return get_format(value)
    registered = get_format(value.name)
    if value != registered:
        raise ValueError(f"numeric format does not match registered {value.name!r}")
    return registered


def _layout(value: str | Layout) -> Layout:
    if isinstance(value, str):
        return get_layout(value)
    registered = get_layout(value.name)
    if value != registered:
        raise ValueError(f"layout does not match registered {value.name!r}")
    return registered


def _shape(value: Sequence[int], *, rank: int | None = None) -> tuple[int, ...]:
    dims = []
    for dim in value:
        if isinstance(dim, bool):
            raise ValueError("shape dimensions must be positive integers")
        try:
            item = operator.index(dim)
        except TypeError:
            raise ValueError("shape dimensions must be positive integers") from None
        if item <= 0:
            raise ValueError("shape dimensions must be positive integers")
        dims.append(item)
    result = tuple(dims)
    if rank is not None and len(result) != rank:
        raise ValueError(f"layout requires rank {rank}, got rank {len(result)}")
    return result


def row_split_geometry(
    format: str | QuantFormat, shape: Sequence[int]
) -> RowSplitGeometry:
    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row-split-k128-v1 requires a grouped quantized format")
    n, k = _shape(shape, rank=2)
    k_pad = align_up(k, K_ALIGNMENT)
    groups_per_row = k_pad // spec.group_size
    base_bytes_per_group = spec.group_size if spec.bits == 8 else spec.group_size // 2
    high_bytes_per_group = (
        0 if spec.bits in (4, 8) else spec.group_size * (spec.bits - 4) // 8
    )
    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2
    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k_pad,
        groups_per_row=groups_per_row,
        base_bytes_per_group=base_bytes_per_group,
        high_bytes_per_group=high_bytes_per_group,
        base_row_bytes=base_row_bytes,
        high_row_bytes=high_row_bytes,
        scale_row_bytes=scale_row_bytes,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def block_scale_geometry(
    format: str | Nvfp4Format, shape: Sequence[int]
) -> BlockScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("blockscale-k16-m128x4-v1 requires NVFP4")
    n, k = _shape(shape, rank=2)
    if n % 128 != 0 or k % 64 != 0:
        raise ValueError(
            "blockscale-k16-m128x4-v1 requires N divisible by 128 "
            "and K divisible by 64"
        )
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * k // spec.group_size
    weight_divisor_offset = scale_plane_offset + scale_plane_bytes
    return BlockScaleGeometry(
        n=n,
        k=k,
        groups_per_row=k // spec.group_size,
        k_tiles=k // 64,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        weight_divisor_offset=weight_divisor_offset,
        payload_bytes=weight_divisor_offset + 4,
    )


def row_scale_geometry(
    format: str | Fp8RowFormat, shape: Sequence[int]
) -> RowScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError("row-scale-v1 requires a row-scaled FP8 format")
    n, k = _shape(shape, rank=2)
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * 2
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def fp8_block_geometry(
    format: str | Fp8BlockFormat, shape: Sequence[int]
) -> Fp8BlockGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8BlockFormat):
        raise ValueError("blockscale-m128-k128-v1 requires block-scaled FP8")
    n, k = _shape(shape, rank=2)
    if n % 128 or k % 128:
        raise ValueError("blockscale-m128-k128-v1 requires N and K divisible by 128")
    codes = n * k
    offset = align_up(codes, PLANE_ALIGNMENT)
    scales = (n // 128) * (k // 128) * 2
    return Fp8BlockGeometry(n, k, codes, offset, scales, offset + scales)


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
    *,
    stored_bytes: int | None = None,
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is GGML_K256_V1:
        n, k = _shape(shape, rank=2)
        if k % 256:
            raise ValueError("ggml-k256-v1 requires K divisible by 256")
        minimum = align_up(n * 8, 256) + n * (k // 256) * 144
        maximum = align_up(n * 8, 256) + n * (k // 256) * 210
        if stored_bytes is None or not minimum <= stored_bytes <= maximum:
            raise ValueError("ggml-k256-v1 requires its exact descriptor-derived byte size")
        if (stored_bytes - minimum) % ((k // 256) * 66):
            raise ValueError("ggml-k256-v1 byte size does not describe whole Q6_K rows")
        return stored_bytes
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous-le-v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous-le-v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row-split-k128-v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCKSCALE_K16_M128X4_V1:
        if not isinstance(numeric_spec, Nvfp4Format):
            raise ValueError("blockscale-k16-m128x4-v1 requires NVFP4")
        return block_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_SCALE_V1:
        if not isinstance(numeric_spec, Fp8RowFormat):
            raise ValueError("row-scale-v1 requires a row-scaled FP8 format")
        return row_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCKSCALE_M128_K128_V1:
        return fp8_block_geometry(numeric_spec, shape).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")


def validate_ggml_k_payload(shape: Sequence[int], payload: bytes | memoryview) -> None:
    n, k = _shape(shape, rank=2)
    encoded_size(GGML_K256_V1, "GGML_K", shape, stored_bytes=len(payload))
    code_offset = align_up(n * 8, 256)
    cursor = 0
    for row in range(n):
        descriptor = struct.unpack_from("<Q", payload, row * 8)[0]
        if descriptor >> 1 != cursor:
            raise ValueError("GGML_K row descriptors are not canonical contiguous rows")
        cursor += (k // 256) * (210 if descriptor & 1 else 144)
    if code_offset + cursor != len(payload):
        raise ValueError("GGML_K row descriptors disagree with payload byte size")
    if any(payload[n * 8:code_offset]):
        raise ValueError("GGML_K descriptor padding must be zero")
