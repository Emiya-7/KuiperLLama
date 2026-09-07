# Qwen3.5 real-checkpoint alignment

These tools compare Kuiper's token-by-token CPU inference with the official
Transformers FP32 eager implementation. They dump the same prompt tokens,
post-decoder hidden states, final normalized hidden state, and logits on both
sides, then report the numerical error at every layer. They also require the
first 10 greedy generated tokens to match exactly.

## Build

Configure the project with `QWEN35_SUPPORT=ON`, then build the trace runner:

```bash
cmake --build build --target qwen35_trace -j2
```

Export the Hugging Face checkpoint with a sequence length large enough for the
prompt:

```bash
python3 tools/export_qwen35/export.py \
  --model_dir /path/to/Qwen3.5-0.8B \
  --output /tmp/qwen35-0.8b-bf16.bin \
  --max_seq_len 128 \
  --weight_dtype bf16
```

`bf16` is the default and stores embedding/projection matrices at half the
FP32 size while retaining FP32 activations, accumulation, recurrent state, and
small parameters. Use `--weight_dtype fp32` when a full-FP32 checkpoint is
needed. The loader remains compatible with version-2 FP32 exports.

## Generate and compare traces

```bash
./build/demo/qwen35_trace \
  /tmp/qwen35-0.8b-bf16.bin \
  /path/to/Qwen3.5-0.8B/tokenizer.json \
  /tmp/qwen35-kuiper-trace

python3 tools/verify_qwen35/hf_reference.py \
  --model_dir /path/to/Qwen3.5-0.8B \
  --output_dir /tmp/qwen35-hf-trace

python3 tools/verify_qwen35/compare_traces.py \
  --kuiper_dir /tmp/qwen35-kuiper-trace \
  --reference_dir /tmp/qwen35-hf-trace
```

The default prompt is:

```text
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant
```

Pass a fourth positional argument to `qwen35_trace` and `--prompt` to the
Python reference when checking another prompt. Both sides must receive exactly
the same bytes.

The comparison fails if tokenization differs, any of the first 10 generated
tokens differs, or any hidden-state layer exceeds the configured relative tolerance.
Raw trace files are little-endian `int32`/`float32`; `metadata.json` records the
shapes required to read them.
