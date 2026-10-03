"""Exact, CPU-streaming conversion of Qwen3.8-27B block-128 FP8 Text/MTP.

Preserves E4M3FN codes and BF16 block multipliers. No requantization, Torch,
activation calibration, or GPU is used. The output is the native TP4/SM70
execution package; Vision and DFlash are deliberately not included.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import time

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec
from tools.artifact.layout_metadata import fp8_block_geometry

FP8 = "FP8_E4M3FN_BLOCK128_BF16S"
BLOCK = "blockscale-m128-k128-v1"
DIRECT = "contiguous-le-v1"
RESOURCES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json")


class Source:
    """Read exact file ranges from the explicitly selected safetensors index."""
    def __init__(self, directory: Path):
        self.directory = directory
        self.index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        for shard in sorted(set(self.index.values())):
            with (directory / shard).open("rb") as stream:
                length, = struct.unpack("<Q", stream.read(8))
                self.headers[shard] = (8 + length, json.loads(stream.read(length)))

    def metadata(self, name):
        shard = self.index[name]
        origin, header = self.headers[shard]
        return self.directory / shard, origin, header[name]

    def require(self, name, dtype, shape):
        _, _, meta = self.metadata(name)
        if meta["dtype"] != dtype or tuple(meta["shape"]) != tuple(shape):
            raise ValueError(f"{name}: expected {dtype} {shape}, got {meta}")
        width = {"BF16": 2, "F32": 4, "F8_E4M3": 1}[dtype]
        elements = 1
        for extent in shape:
            elements *= extent
        if meta["data_offsets"][1] - meta["data_offsets"][0] != elements * width:
            raise ValueError(f"{name}: inconsistent safetensors byte range")

    def chunks(self, name, begin=0, length=None):
        path, origin, meta = self.metadata(name)
        first, last = meta["data_offsets"]
        if length is None:
            length = last - first - begin
        if begin < 0 or length < 0 or begin + length > last - first:
            raise ValueError(f"{name}: invalid exact source byte range")
        with path.open("rb") as stream:
            stream.seek(origin + first + begin)
            while length:
                data = stream.read(min(length, 4 << 20))
                if not data:
                    raise ValueError(f"{name}: truncated tensor payload")
                yield data
                length -= len(data)

    def raw(self, name):
        return b"".join(self.chunks(name))


@dataclass(frozen=True)
class Rows:
    source: str
    begin: int
    count: int


@dataclass(frozen=True)
class Entry:
    spec: TensorSpec
    pieces: tuple[Rows, ...]
    transform: str = "rows"


def attention_rows(prefix):
    # HF q_proj stores [Q256,G256] within each head. Move whole 128-row
    # blocks, so each output code block keeps its original scale row.
    query = tuple(Rows(prefix + "q_proj.weight", head * 512, 256) for head in range(24))
    gate = tuple(Rows(prefix + "q_proj.weight", head * 512 + 256, 256) for head in range(24))
    return query + (Rows(prefix + "k_proj.weight", 0, 1024),) + gate + (
        Rows(prefix + "v_proj.weight", 0, 1024),)


def inventory():
    entries = []
    def add(name, shape, fmt, pieces, transform="rows"):
        entries.append(Entry(TensorSpec(name, shape, fmt, BLOCK if fmt == FP8 else DIRECT),
                             tuple(pieces), transform))
    def direct(name, source, shape, fmt="BF16", transform="rows"):
        add(name, shape, fmt, [Rows(source, 0, shape[0])], transform)
    def matrix(name, source, n, k):
        add(name, (n, k), FP8, [Rows(source, 0, n)])
    direct("text/token_embedding", "model.language_model.embed_tokens.weight", (248320, 5120))
    for layer in range(64):
        dst = f"text/layers/{layer}/"
        src = f"model.language_model.layers.{layer}."
        direct(dst + "input_norm", src + "input_layernorm.weight", (5120,))
        if layer % 4 == 3:
            attn = src + "self_attn."
            add(dst + "attention/query_key_gate_value", (14336, 5120), FP8, attention_rows(attn))
            direct(dst + "attention/query_norm", attn + "q_norm.weight", (256,))
            direct(dst + "attention/key_norm", attn + "k_norm.weight", (256,))
            matrix(dst + "attention/output", attn + "o_proj.weight", 5120, 6144)
        else:
            gdn = src + "linear_attn."
            direct(dst + "gdn/a_log", gdn + "A_log", (48,), "FP32", "bf16_to_fp32")
            direct(dst + "gdn/dt_bias", gdn + "dt_bias", (48,), "FP32", "bf16_to_fp32")
            direct(dst + "gdn/convolution", gdn + "conv1d.weight", (4, 10240), transform="conv")
            add(dst + "gdn/a_b_projection", (96, 5120), "BF16", [
                Rows(gdn + "in_proj_a.weight", 0, 48), Rows(gdn + "in_proj_b.weight", 0, 48)])
            add(dst + "gdn/query_key_value_z", (16384, 5120), FP8, [
                Rows(gdn + "in_proj_qkv.weight", 0, 10240), Rows(gdn + "in_proj_z.weight", 0, 6144)])
            direct(dst + "gdn/norm", gdn + "norm.weight", (128,))
            matrix(dst + "gdn/output", gdn + "out_proj.weight", 5120, 6144)
        direct(dst + "post_attention_norm", src + "post_attention_layernorm.weight", (5120,))
        add(dst + "mlp/gate_up", (34816, 5120), FP8, [
            Rows(src + "mlp.gate_proj.weight", 0, 17408), Rows(src + "mlp.up_proj.weight", 0, 17408)])
        matrix(dst + "mlp/down", src + "mlp.down_proj.weight", 5120, 17408)
    direct("text/final_norm", "model.language_model.norm.weight", (5120,))
    direct("text/output_head", "lm_head.weight", (248320, 5120))
    # An exact subset for the optional optimized proposal head. Target verification
    # always uses the complete vocabulary. --lm-head-draft selects this shortlist;
    # omitting it uses the full head for proposals as well.
    direct("text/draft_head", "lm_head.weight", (131072, 5120))
    add("text/draft_head_token_ids", (131072,), "I32", [], "draft_ids")
    direct("mtp/input_projection", "mtp.fc.weight", (5120, 10240))
    direct("mtp/embedding_norm", "mtp.pre_fc_norm_embedding.weight", (5120,))
    direct("mtp/hidden_norm", "mtp.pre_fc_norm_hidden.weight", (5120,))
    direct("mtp/layer/input_norm", "mtp.layers.0.input_layernorm.weight", (5120,))
    add("mtp/layer/attention/query_key_gate_value", (14336, 5120), FP8,
        attention_rows("mtp.layers.0.self_attn."))
    direct("mtp/layer/attention/query_norm", "mtp.layers.0.self_attn.q_norm.weight", (256,))
    direct("mtp/layer/attention/key_norm", "mtp.layers.0.self_attn.k_norm.weight", (256,))
    matrix("mtp/layer/attention/output", "mtp.layers.0.self_attn.o_proj.weight", 5120, 6144)
    direct("mtp/layer/post_attention_norm", "mtp.layers.0.post_attention_layernorm.weight", (5120,))
    add("mtp/layer/mlp/gate_up", (34816, 5120), FP8, [
        Rows("mtp.layers.0.mlp.gate_proj.weight", 0, 17408),
        Rows("mtp.layers.0.mlp.up_proj.weight", 0, 17408)])
    matrix("mtp/layer/mlp/down", "mtp.layers.0.mlp.down_proj.weight", 5120, 17408)
    direct("mtp/final_norm", "mtp.norm.weight", (5120,))
    return tuple(entries)


def preflight(source, entries):
    config = json.loads((source.directory / "config.json").read_text())
    text = config["text_config"]
    expected = {"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
                "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
                "mtp_num_hidden_layers": 1, "vocab_size": 248320,
                "rms_norm_eps": 1e-6, "max_position_embeddings": 262144}
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"text_config.{key}: expected {value}")
    quant = config["quantization_config"]
    if (quant.get("quant_method") != "fp8" or quant.get("fmt") != "e4m3"
            or quant.get("weight_block_size") != [128, 128]):
        raise ValueError("requires the original E4M3FN block-128 checkpoint")
    for entry in entries:
        spec = entry.spec
        for piece in entry.pieces:
            _, _, meta = source.metadata(piece.source)
            shape = tuple(meta["shape"])
            if entry.transform == "conv":
                source.require(piece.source, "BF16", (10240, 1, 4))
            elif entry.transform == "bf16_to_fp32":
                source.require(piece.source, "BF16", spec.shape)
            else:
                dtype = "F8_E4M3" if spec.format == FP8 else "BF16"
                if shape[1:] != spec.shape[1:] or piece.begin + piece.count > shape[0]:
                    raise ValueError(f"{piece.source}: invalid row mapping for {spec.name}")
                source.require(piece.source, dtype, shape)
                if spec.format == FP8:
                    n, k = shape
                    if piece.begin % 128 or piece.count % 128:
                        raise ValueError("FP8 row fusion must preserve complete scale blocks")
                    source.require(piece.source.removesuffix(".weight") + ".weight_scale_inv",
                                   "BF16", (n // 128, k // 128))


def payload(source, entry):
    spec = entry.spec
    if entry.transform == "draft_ids":
        yield struct.pack("<131072i", *range(131072))
        return
    if entry.transform == "bf16_to_fp32":
        raw = source.raw(entry.pieces[0].source)
        # Expanding BF16 to FP32 inserts 16 zero low bits, with no arithmetic.
        yield b"".join(b"\0\0" + raw[i:i+2] for i in range(0, len(raw), 2))
        return
    if entry.transform == "conv":
        raw = source.raw(entry.pieces[0].source)
        yield b"".join(raw[(channel * 4 + tap) * 2:(channel * 4 + tap) * 2 + 2]
                       for tap in range(4) for channel in range(10240))
        return
    stride = (spec.shape[1] if len(spec.shape) == 2 else 1) * (1 if spec.format == FP8 else 2)
    for piece in entry.pieces:
        yield from source.chunks(piece.source, piece.begin * stride, piece.count * stride)
    if spec.format == FP8:
        geometry = fp8_block_geometry(FP8, spec.shape)
        yield bytes(geometry.scale_plane_offset - geometry.code_plane_bytes)
        scale_stride = spec.shape[1] // 128 * 2
        for piece in entry.pieces:
            scale = piece.source.removesuffix(".weight") + ".weight_scale_inv"
            yield from source.chunks(scale, piece.begin // 128 * scale_stride,
                                     piece.count // 128 * scale_stride)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; select a new explicit destination")
    source = Source(args.model)
    entries = inventory()
    preflight(source, entries)
    resources = [(name, (args.model / name).read_bytes()) for name in RESOURCES]
    if any(not data for _, data in resources):
        raise ValueError("all six frontend resources must be nonempty")
    specs = [entry.spec for entry in entries] + [
        ResourceSpec("frontend/" + name, "raw-bytes-v1", len(data)) for name, data in resources]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    with ArtifactWriter(args.out, ArtifactIdentity("qwen3.8-27b", "fp8"), specs) as writer:
        for entry in entries:
            writer.write(entry.spec.name, payload(source, entry))
            print(entry.spec.name, flush=True)
        for name, data in resources:
            writer.write("frontend/" + name, data)
    report = {"identity": "qwen3.8-27b/fp8", "source": str(args.model.resolve()),
              "output": str(args.out.resolve()), "bytes": args.out.stat().st_size,
              "seconds": time.monotonic() - begin, "tensors": len(entries),
              "code_and_scale_transform": "exact row moves; no requantization",
              "draft_head": "exact first 131072 BF16 output-head rows",
              "execution": "SM70 TP4 Text/MTP; no Vision/DFlash"}
    args.out.with_suffix(".ninfer.conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
