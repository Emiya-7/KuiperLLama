#!/usr/bin/env python3
"""Export the text tower of a Qwen3.5 checkpoint to the Kuiper .bin format.

Qwen3.5 ships as a multimodal checkpoint (Qwen3_5ForConditionalGeneration): the
weights we want live under `model.language_model.*`, alongside a `model.visual.*`
vision tower and `mtp.*` multi-token-prediction layers that text-only inference
does not use. Those two groups are skipped.

The model is a hybrid: `full_attention_interval` (4) means layers 3, 7, 11, ...
are GQA attention and the rest are Gated DeltaNet. The two kinds carry different
weights, so the writer walks layers in order and branches on the type.

Reads safetensors directly -- no torch, no transformers, no numpy -- so this runs
anywhere Python 3.8+ does. Version 3 can preserve large embedding/projection
matrices as BF16 while widening small parameters to FP32. Activations and model
state remain FP32 in Kuiper.

Usage:
  python3 export.py --model_dir /path/to/Qwen3.5-4B --output qwen35_4b.bin
  python3 export.py --model_dir ... --output ... --max_seq_len 8192
"""

import argparse
import json
import os
import struct
import sys

MAGIC = ord("K") | (ord("3") << 8) | (ord("5") << 16) | (ord("D") << 24)
VERSION = 3
MATRIX_FP32 = 0
MATRIX_BF16 = 1

LM = "model.language_model."


# ---------------------------------------------------------------- safetensors

class SafeTensors:
    """Lazy multi-shard safetensors reader. Keeps only one shard header in memory."""

    def __init__(self, model_dir):
        self.dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = "model.safetensors"
            if not os.path.exists(os.path.join(model_dir, single)):
                raise FileNotFoundError(f"no safetensors found under {model_dir}")
            self.weight_map = None
            self._single = single
        self._headers = {}
        self._files = {}

    def _shard_of(self, name):
        if self.weight_map is None:
            # Single-file checkpoint: the header is the only index, so consult it
            # rather than assuming every name resolves.
            hdr, _ = self._header(self._single)
            if name not in hdr:
                raise KeyError(name)
            return self._single
        if name not in self.weight_map:
            raise KeyError(name)
        return self.weight_map[name]

    def _header(self, shard):
        if shard not in self._headers:
            f = open(os.path.join(self.dir, shard), "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            self._headers[shard] = (hdr, 8 + n)
            self._files[shard] = f
        return self._headers[shard]

    def names(self):
        if self.weight_map is not None:
            return list(self.weight_map.keys())
        hdr, _ = self._header(self._single)
        return list(hdr.keys())

    def has(self, name):
        try:
            self._shard_of(name)
            return True
        except KeyError:
            return False

    def info(self, name):
        shard = self._shard_of(name)
        hdr, _ = self._header(shard)
        e = hdr[name]
        return e["shape"], e["dtype"]

    def raw(self, name):
        """Return (bytes, shape, dtype) for one tensor."""
        shard = self._shard_of(name)
        hdr, base = self._header(shard)
        e = hdr[name]
        start, end = e["data_offsets"]
        f = self._files[shard]
        f.seek(base + start)
        return f.read(end - start), e["shape"], e["dtype"]

    def close(self):
        for f in self._files.values():
            f.close()


# ------------------------------------------------------------ dtype widening

def to_fp32_bytes(buf, dtype):
    """Widen a raw tensor buffer to little-endian fp32 bytes."""
    if dtype == "F32":
        return buf
    if dtype == "BF16":
        # bf16 is the top 16 bits of fp32: interleave zero bytes below each pair.
        out = bytearray(len(buf) * 2)
        out[2::4] = buf[0::2]
        out[3::4] = buf[1::2]
        return bytes(out)
    if dtype == "F16":
        import array
        import struct as _s
        # No numpy: decode with struct into a float array, then re-pack.
        n = len(buf) // 2
        vals = array.array("f", [0.0]) * n
        for i in range(n):
            vals[i] = _s.unpack_from("<e", buf, i * 2)[0]
        return vals.tobytes()
    raise ValueError(f"unsupported dtype {dtype}")


def to_bf16_bytes(buf, dtype):
    """Convert a raw tensor buffer to little-endian BF16 with RNE rounding."""
    if dtype == "BF16":
        return buf
    if dtype == "F32":
        out = bytearray(len(buf) // 2)
        for index, (bits,) in enumerate(struct.iter_unpack("<I", buf)):
            rounded = bits + 0x7FFF + ((bits >> 16) & 1)
            struct.pack_into("<H", out, index * 2, (rounded >> 16) & 0xFFFF)
        return bytes(out)
    if dtype == "F16":
        out = bytearray(len(buf))
        for index, (value,) in enumerate(struct.iter_unpack("<e", buf)):
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            rounded = bits + 0x7FFF + ((bits >> 16) & 1)
            struct.pack_into("<H", out, index * 2, (rounded >> 16) & 0xFFFF)
        return bytes(out)
    raise ValueError(f"unsupported dtype {dtype}")


# ------------------------------------------------------------------- writing

class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.bytes_written = 0
        self.tensors = 0

    def tensor(self, st, name, expect_shape=None, output_dtype="fp32"):
        buf, shape, dtype = st.raw(name)
        if expect_shape is not None and list(shape) != list(expect_shape):
            raise ValueError(f"{name}: shape {shape} != expected {expect_shape}")
        if output_dtype == "fp32":
            data = to_fp32_bytes(buf, dtype)
        elif output_dtype == "bf16":
            data = to_bf16_bytes(buf, dtype)
        else:
            raise ValueError(f"unsupported output dtype {output_dtype}")
        self.f.write(data)
        self.bytes_written += len(data)
        self.tensors += 1
        return shape

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="local HF snapshot directory")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help="clamp for the RoPE cache; the config's 262144 would need ~1GB of tables",
    )
    ap.add_argument("--dry_run", action="store_true", help="validate shapes, write nothing")
    ap.add_argument(
        "--weight_dtype",
        choices=("fp32", "bf16"),
        default="bf16",
        help="storage dtype for embedding and projection matrices (default: bf16)",
    )
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg_all = json.load(f)
    cfg = cfg_all.get("text_config", cfg_all)

    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    n_layer = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    head_num = cfg["num_attention_heads"]
    kv_head_num = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    interval = cfg["full_attention_interval"]
    nk = cfg["linear_num_key_heads"]
    nv = cfg["linear_num_value_heads"]
    kd = cfg["linear_key_head_dim"]
    vd = cfg["linear_value_head_dim"]
    conv_k = cfg["linear_conv_kernel_dim"]
    eps = cfg["rms_norm_eps"]

    rope = cfg.get("rope_parameters", {})
    theta = float(rope.get("rope_theta", cfg.get("rope_theta", 10000.0)))
    prf = float(rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0)))
    rotary_dim = int(head_dim * prf)

    tie = bool(cfg.get("tie_word_embeddings", cfg_all.get("tie_word_embeddings", False)))

    st = SafeTensors(args.model_dir)
    # tie_word_embeddings in the config is advisory; trust the file.
    has_lm_head = st.has("lm_head.weight")
    tie = not has_lm_head

    k_dim = nk * kd
    v_dim = nv * vd
    conv_dim = k_dim * 2 + v_dim
    q_proj_out = head_num * head_dim * 2  # second half is the output gate
    kv_dim = kv_head_num * head_dim

    layer_types = [
        "full" if ((i + 1) % interval) == 0 else "linear" for i in range(n_layer)
    ]
    n_full = layer_types.count("full")

    print(f"model      : {cfg_all.get('architectures')}")
    print(f"hidden={hidden} inter={inter} layers={n_layer} vocab={vocab}")
    print(f"full attn  : heads={head_num} kv={kv_head_num} head_dim={head_dim} "
          f"rotary_dim={rotary_dim} theta={theta:g}")
    print(f"linear attn: k_heads={nk} v_heads={nv} kd={kd} vd={vd} conv_k={conv_k} "
          f"conv_dim={conv_dim}")
    print(f"layers     : {n_full} full / {n_layer - n_full} linear (interval {interval})")
    print(f"tie_emb    : {tie} (lm_head.weight present: {has_lm_head})")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"matrix dtype: {args.weight_dtype} (activations and small parameters stay fp32)")

    if args.dry_run:
        # Verify every tensor we intend to read exists with the expected shape.
        problems = []

        def check(name, shape):
            if not st.has(name):
                problems.append(f"MISSING {name}")
                return
            got, _ = st.info(name)
            if list(got) != list(shape):
                problems.append(f"SHAPE   {name}: {got} != {shape}")

        check(LM + "embed_tokens.weight", [vocab, hidden])
        check(LM + "norm.weight", [hidden])
        if has_lm_head:
            check("lm_head.weight", [vocab, hidden])
        for i, t in enumerate(layer_types):
            p = f"{LM}layers.{i}."
            check(p + "input_layernorm.weight", [hidden])
            check(p + "post_attention_layernorm.weight", [hidden])
            check(p + "mlp.gate_proj.weight", [inter, hidden])
            check(p + "mlp.up_proj.weight", [inter, hidden])
            check(p + "mlp.down_proj.weight", [hidden, inter])
            if t == "full":
                a = p + "self_attn."
                check(a + "q_proj.weight", [q_proj_out, hidden])
                check(a + "k_proj.weight", [kv_dim, hidden])
                check(a + "v_proj.weight", [kv_dim, hidden])
                check(a + "o_proj.weight", [hidden, head_num * head_dim])
                check(a + "q_norm.weight", [head_dim])
                check(a + "k_norm.weight", [head_dim])
            else:
                a = p + "linear_attn."
                check(a + "in_proj_qkv.weight", [conv_dim, hidden])
                check(a + "in_proj_z.weight", [v_dim, hidden])
                check(a + "in_proj_a.weight", [nv, hidden])
                check(a + "in_proj_b.weight", [nv, hidden])
                check(a + "conv1d.weight", [conv_dim, 1, conv_k])
                check(a + "A_log", [nv])
                check(a + "dt_bias", [nv])
                check(a + "norm.weight", [vd])
                check(a + "out_proj.weight", [hidden, v_dim])
        if problems:
            print(f"\n{len(problems)} problem(s):")
            for p in problems[:40]:
                print("  " + p)
            return 1
        print("\ndry run OK: all tensors present with expected shapes")
        return 0

    w = Writer(args.output)
    hdr = struct.pack(
        "<19i2f",
        MAGIC, VERSION,
        hidden, inter, n_layer, vocab, args.max_seq_len,
        head_num, kv_head_num, head_dim, rotary_dim, interval,
        nk, nv, kd, vd, conv_k,
        1 if tie else 0,
        MATRIX_BF16 if args.weight_dtype == "bf16" else MATRIX_FP32,
        theta, eps,
    )
    w.f.write(hdr)
    w.bytes_written += len(hdr)

    # Weight order below is the order Qwen35Model::create_param_layers reads.
    # Global tensors first, then layers in index order so a layer's weights are
    # contiguous on disk and the loader can stream straight through.
    def matrix(name, shape):
        w.tensor(st, name, shape, args.weight_dtype)

    matrix(LM + "embed_tokens.weight", [vocab, hidden])
    w.tensor(st, LM + "norm.weight", [hidden])

    for i, t in enumerate(layer_types):
        p = f"{LM}layers.{i}."
        w.tensor(st, p + "input_layernorm.weight", [hidden])
        if t == "full":
            a = p + "self_attn."
            matrix(a + "q_proj.weight", [q_proj_out, hidden])
            matrix(a + "k_proj.weight", [kv_dim, hidden])
            matrix(a + "v_proj.weight", [kv_dim, hidden])
            w.tensor(st, a + "q_norm.weight", [head_dim])
            w.tensor(st, a + "k_norm.weight", [head_dim])
            matrix(a + "o_proj.weight", [hidden, head_num * head_dim])
        else:
            a = p + "linear_attn."
            matrix(a + "in_proj_qkv.weight", [conv_dim, hidden])
            matrix(a + "in_proj_z.weight", [v_dim, hidden])
            matrix(a + "in_proj_a.weight", [nv, hidden])
            matrix(a + "in_proj_b.weight", [nv, hidden])
            w.tensor(st, a + "conv1d.weight", [conv_dim, 1, conv_k])
            w.tensor(st, a + "A_log", [nv])
            w.tensor(st, a + "dt_bias", [nv])
            w.tensor(st, a + "norm.weight", [vd])
            matrix(a + "out_proj.weight", [hidden, v_dim])
        w.tensor(st, p + "post_attention_layernorm.weight", [hidden])
        matrix(p + "mlp.gate_proj.weight", [inter, hidden])
        matrix(p + "mlp.up_proj.weight", [inter, hidden])
        matrix(p + "mlp.down_proj.weight", [hidden, inter])
        if (i + 1) % 8 == 0 or i + 1 == n_layer:
            print(f"  layer {i + 1}/{n_layer}  {w.bytes_written / 1e9:.2f} GB")

    if not tie:
        matrix("lm_head.weight", [vocab, hidden])

    w.close()
    st.close()
    print(f"\nwrote {args.output}: {w.tensors} tensors, {w.bytes_written / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
