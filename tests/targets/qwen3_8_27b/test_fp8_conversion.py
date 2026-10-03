"""Exact source-to-container word mapping; independent expected row/block order."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.convert.qwen3_8_27b.convert_fp8 import BLOCK, DIRECT, FP8, Entry, Rows, Source, attention_rows, payload


class Fp8ConversionTest(unittest.TestCase):
    def source(self, directory, values):
        header, data = {}, bytearray()
        for name, (dtype, shape, raw) in values.items():
            begin = len(data)
            data.extend(raw)
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(data)]}
        encoded = json.dumps(header).encode()
        (directory / "fixture.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
        (directory / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {name: "fixture.safetensors" for name in values}}))
        return Source(directory)

    def test_q_gate_fusion_and_container(self):
        # Distinguish every 128-row block and K block; expected ordering is generated
        # from the HF head layout, not from the converter's Rows list.
        values = {}
        k = 256
        for tag, n in (("q", 12288), ("k", 1024), ("v", 1024)):
            codes = b"".join(bytes([(r // 128 + c // 128 + ord(tag)) % 126])
                             for r in range(n) for c in range(k))
            scales = b"".join(struct.pack("<H", 0x3a00 + r + c + ord(tag))
                              for r in range(n // 128) for c in range(k // 128))
            values[tag + "_proj.weight"] = ("F8_E4M3", (n, k), codes)
            values[tag + "_proj.weight_scale_inv"] = ("BF16", (n//128, k//128), scales)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self.source(directory, values)
            spec = TensorSpec("fused", (14336, k), FP8, BLOCK)
            actual = b"".join(payload(source, Entry(spec, attention_rows(""))))
            order = [("q", head*512 + local) for head in range(24) for local in range(256)]
            order += [("k", r) for r in range(1024)]
            order += [("q", head*512 + 256 + local) for head in range(24) for local in range(256)]
            order += [("v", r) for r in range(1024)]
            expected_codes = b"".join(values[tag+"_proj.weight"][2][r*k:(r+1)*k] for tag,r in order)
            expected_scales = b"".join(values[tag+"_proj.weight_scale_inv"][2][r//128*4:r//128*4+4]
                                       for tag,r in order[::128])
            expected = expected_codes + expected_scales
            self.assertEqual(actual, expected)
            destination = directory / "fixture.ninfer"
            with ArtifactWriter(destination, ArtifactIdentity("qwen3.8-27b", "fp8"), [spec]) as writer:
                writer.write(spec.name, actual)
            with Artifact(destination) as artifact:
                self.assertEqual(artifact.identity.weights_id, "fp8")
                self.assertEqual(bytes(artifact.payload(spec.name)), expected)

    def test_control_expansion_and_convolution(self):
        control = struct.pack("<3H", 0x3f80, 0x8000, 0xbfa1)
        conv = b"".join(struct.pack("<H", (channel*7+tap)&0xffff)
                        for channel in range(10240) for tap in range(4))
        with tempfile.TemporaryDirectory() as tmp:
            source = self.source(Path(tmp), {"control": ("BF16", (3,), control),
                "conv": ("BF16", (10240,1,4), conv)})
            entry = Entry(TensorSpec("control", (3,), "FP32", DIRECT),
                          (Rows("control",0,3),), "bf16_to_fp32")
            self.assertEqual(b"".join(payload(source,entry)), struct.pack("<3I",0x3f800000,0x80000000,0xbfa10000))
            entry = Entry(TensorSpec("conv", (4,10240), "BF16", DIRECT),
                          (Rows("conv",0,4),), "conv")
            expected = b"".join(struct.pack("<H", (channel*7+tap)&0xffff)
                                for tap in range(4) for channel in range(10240))
            self.assertEqual(b"".join(payload(source,entry)),expected)


if __name__ == "__main__":
    unittest.main()
