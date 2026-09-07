#!/usr/bin/env python3
"""Build a small synthetic Qwen3.5 checkpoint for the unit tests.

Structurally identical to the published 4B/9B models -- full_attention_interval 4,
2 value heads per key head, partial_rotary_factor 0.25, tied embeddings -- but with
small hidden/intermediate sizes so the tests stay fast. The vocabulary is kept at
the real 248320 so the shipped tokenizer.json can be used unchanged.

Writes a safetensors file plus config.json, then leaves it to
tools/export_qwen35/export.py to produce the .bin the tests load.

Needs numpy. Usage:
    python3 test/test_model/make_tiny_qwen35.py --out_dir /tmp/tiny35
    python3 tools/export_qwen35/export.py --model_dir /tmp/tiny35 \
        --output /tmp/tiny35.bin --max_seq_len 128
"""

import argparse
import json
import os
import struct

import numpy as np

TINY = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=8,
    vocab_size=248320,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    full_attention_interval=4,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    rope_parameters=dict(rope_theta=1e7, partial_rotary_factor=0.25, rope_type="default"),
)

LM = "model.language_model."


def tensor_specs(c):
    hid, inter, nl = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    voc, iv = c["vocab_size"], c["full_attention_interval"]
    hn, kvn, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
    kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
    ck = c["linear_conv_kernel_dim"]

    k_dim, v_dim = nk * kd, nv * vd
    conv_dim = k_dim * 2 + v_dim
    q_proj_out = hn * hd * 2  # second half is the output gate
    kv_dim = kvn * hd

    specs = [(LM + "embed_tokens.weight", [voc, hid]), (LM + "norm.weight", [hid])]
    for i in range(nl):
        p = f"{LM}layers.{i}."
        specs.append((p + "input_layernorm.weight", [hid]))
        if (i + 1) % iv == 0:
            a = p + "self_attn."
            specs += [
                (a + "q_proj.weight", [q_proj_out, hid]),
                (a + "k_proj.weight", [kv_dim, hid]),
                (a + "v_proj.weight", [kv_dim, hid]),
                (a + "q_norm.weight", [hd]),
                (a + "k_norm.weight", [hd]),
                (a + "o_proj.weight", [hid, hn * hd]),
            ]
        else:
            a = p + "linear_attn."
            specs += [
                (a + "in_proj_qkv.weight", [conv_dim, hid]),
                (a + "in_proj_z.weight", [v_dim, hid]),
                (a + "in_proj_a.weight", [nv, hid]),
                (a + "in_proj_b.weight", [nv, hid]),
                (a + "conv1d.weight", [conv_dim, 1, ck]),
                (a + "A_log", [nv]),
                (a + "dt_bias", [nv]),
                (a + "norm.weight", [vd]),
                (a + "out_proj.weight", [hid, v_dim]),
            ]
        specs.append((p + "post_attention_layernorm.weight", [hid]))
        specs += [
            (p + "mlp.gate_proj.weight", [inter, hid]),
            (p + "mlp.up_proj.weight", [inter, hid]),
            (p + "mlp.down_proj.weight", [hid, inter]),
        ]
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    c = TINY
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": c,
            },
            f,
            indent=2,
        )

    rng = np.random.default_rng(args.seed)
    header, blobs, offset = {}, [], 0
    for name, shape in tensor_specs(c):
        count = int(np.prod(shape))
        # Qwen3.5 decoder/final/QK RMSNorm weights are zero-centered: the
        # operator multiplies by (1 + weight), so these values stay near zero.
        # GDN's gated RMSNorm is a different operator with an ordinary scale
        # initialized near one, and it is fp32 in the real checkpoints.
        is_gated_norm = name.endswith("linear_attn.norm.weight")
        is_f32 = name.endswith("A_log") or is_gated_norm
        arr = (rng.standard_normal(count) * 0.05).astype("<f4")
        if is_gated_norm:
            arr += 1.0
        if is_f32:
            raw, dtype = arr.tobytes(), "F32"
        else:
            # Truncate to bf16 the same way the published weights are stored.
            raw = (arr.view("<u4") >> 16).astype("<u2").tobytes()
            dtype = "BF16"
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        blobs.append(raw)

    hj = json.dumps(header).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    out = os.path.join(args.out_dir, "model.safetensors")
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)

    print(f"wrote {out}: {len(header)} tensors, {offset / 1e6:.1f} MB")
    print("Copy a real tokenizer.json into this directory, then run export.py.")


if __name__ == "__main__":
    main()
