#!/usr/bin/env python3
"""Compare raw Kuiper and Transformers Qwen3.5 alignment traces."""

import argparse
import json
import os

import numpy as np


def load_metadata(path):
    with open(os.path.join(path, "metadata.json")) as file:
        return json.load(file)


def load_raw(path, name, dtype, shape):
    value = np.fromfile(os.path.join(path, name), dtype=dtype)
    expected = int(np.prod(shape))
    if value.size != expected:
        raise ValueError(f"{path}/{name}: {value.size} values, expected {expected}")
    return value.reshape(shape)


def errors(actual, reference):
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    scale = max(float(np.max(np.abs(reference))), 1e-12)
    max_abs = float(np.max(difference))
    return max_abs, float(np.mean(difference)), max_abs / scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kuiper_dir", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--relative_tolerance", type=float, default=2e-3)
    args = parser.parse_args()

    kuiper = load_metadata(args.kuiper_dir)
    reference = load_metadata(args.reference_dir)
    for key in (
        "sequence_length",
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "generation_length",
    ):
        if kuiper[key] != reference[key]:
            raise ValueError(f"metadata mismatch for {key}: {kuiper[key]} != {reference[key]}")

    seq_len = kuiper["sequence_length"]
    hidden_size = kuiper["hidden_size"]
    layers = kuiper["num_hidden_layers"]
    vocab_size = kuiper["vocab_size"]
    kuiper_tokens = load_raw(args.kuiper_dir, "tokens.i32", "<i4", (seq_len,))
    reference_tokens = load_raw(args.reference_dir, "tokens.i32", "<i4", (seq_len,))
    if not np.array_equal(kuiper_tokens, reference_tokens):
        raise ValueError(f"token mismatch: {kuiper_tokens.tolist()} != {reference_tokens.tolist()}")

    generation_length = kuiper["generation_length"]
    kuiper_generated = load_raw(
        args.kuiper_dir, "generated.i32", "<i4", (generation_length,)
    )
    reference_generated = load_raw(
        args.reference_dir, "generated.i32", "<i4", (generation_length,)
    )
    generation_matches = np.array_equal(kuiper_generated, reference_generated)

    print("layer   max_abs       mean_abs      rel_max")
    first_divergence = None
    for layer in range(layers + 1):
        name = f"hidden_{layer:03d}.f32"
        actual = load_raw(args.kuiper_dir, name, "<f4", (seq_len, hidden_size))
        expected = load_raw(args.reference_dir, name, "<f4", (seq_len, hidden_size))
        max_abs, mean_abs, rel_max = errors(actual, expected)
        print(f"{layer:5d} {max_abs:12.5e} {mean_abs:12.5e} {rel_max:12.5e}")
        if first_divergence is None and rel_max > args.relative_tolerance:
            first_divergence = layer

    actual_logits = load_raw(args.kuiper_dir, "logits.f32", "<f4", (seq_len, vocab_size))
    expected_logits = load_raw(
        args.reference_dir, "logits.f32", "<f4", (seq_len, vocab_size)
    )
    max_abs, mean_abs, rel_max = errors(actual_logits, expected_logits)
    print(f"logits max_abs={max_abs:.5e} mean_abs={mean_abs:.5e} rel_max={rel_max:.5e}")
    print(
        f"generated_match={generation_matches} first_divergence={first_divergence}\n"
        f"kuiper_generated={kuiper_generated.tolist()}\n"
        f"reference_generated={reference_generated.tolist()}"
    )

    if not generation_matches or first_divergence is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
