#!/usr/bin/env python3
"""Dump official Transformers Qwen3.5 hidden states and logits as fp32 raw files."""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig


DEFAULT_PROMPT = "<|im_start|>user\nWhat is AI?<|im_end|>\n<|im_start|>assistant\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="reference compute dtype (default: fp32; use bf16 for large models)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with open(os.path.join(args.model_dir, "config.json")) as file:
        serialized_config = json.load(file)
    text_config = Qwen3_5TextConfig(
        **serialized_config.get("text_config", serialized_config)
    )
    torch_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    # Loading the causal LM directly avoids materializing the unused vision and
    # MTP towers from the multimodal checkpoint. Transformers maps the
    # model.language_model.* checkpoint prefix onto the text model here.
    model = Qwen3_5ForCausalLM.from_pretrained(
        args.model_dir,
        config=text_config,
        dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.eval()
    text_model = model.model
    hidden = [None] * (len(text_model.layers) + 1)
    hooks = []

    def capture(index):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            hidden[index] = value.detach().float().cpu().numpy()[0].copy()

        return hook

    for index, layer in enumerate(text_model.layers):
        hooks.append(layer.register_forward_hook(capture(index)))
    hooks.append(text_model.norm.register_forward_hook(capture(len(text_model.layers))))

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=0,
        )
    for hook in hooks:
        hook.remove()

    with torch.no_grad():
        generated_sequence = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            min_new_tokens=10,
            do_sample=False,
            use_cache=True,
        )

    logits = output.logits.detach().float().cpu().numpy()[0]
    # AutoTokenizer also injects audio-only tokens from tokenizer_config.json.
    # Kuiper's text-only tokenizer reads tokenizer.json, whose highest added ID
    # is </think>. Derive the same contiguous text sampling range here instead
    # of using tokenizer.vocab_size (base tokens only) or len(tokenizer)
    # (base + text + unsupported audio tokens).
    with open(os.path.join(args.model_dir, "tokenizer.json")) as file:
        serialized_tokenizer = json.load(file)
    tokenizer_size = max(
        len(serialized_tokenizer["model"]["vocab"]),
        max(token["id"] for token in serialized_tokenizer["added_tokens"]) + 1,
    )
    next_token = int(np.argmax(logits[-1, :tokenizer_size]))
    generated = generated_sequence[0, len(token_ids) :].detach().cpu().numpy()
    if len(generated) != 10:
        raise RuntimeError(f"Transformers generated {len(generated)} tokens, expected 10")
    np.asarray(token_ids, dtype="<i4").tofile(os.path.join(args.output_dir, "tokens.i32"))
    np.asarray(generated, dtype="<i4").tofile(os.path.join(args.output_dir, "generated.i32"))
    np.asarray(logits, dtype="<f4").tofile(os.path.join(args.output_dir, "logits.f32"))
    for index, value in enumerate(hidden):
        if value is None:
            raise RuntimeError(f"hidden-state hook {index} did not run")
        np.asarray(value, dtype="<f4").tofile(
            os.path.join(args.output_dir, f"hidden_{index:03d}.f32")
        )

    metadata = {
        "implementation": "transformers",
        "dtype": args.dtype,
        "sequence_length": len(token_ids),
        "hidden_size": int(text_model.config.hidden_size),
        "num_hidden_layers": len(text_model.layers),
        "vocab_size": int(logits.shape[-1]),
        "base_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_vocab_size": tokenizer_size,
        "transformers_tokenizer_size": len(tokenizer),
        "generation_length": int(len(generated)),
        "next_token": next_token,
        "next_token_text": tokenizer.decode([next_token]),
        "generated_text": tokenizer.decode(generated.tolist()),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
