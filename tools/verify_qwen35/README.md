# Qwen3.5 real-checkpoint alignment

These tools compare Kuiper's token-by-token CPU or CUDA inference with another
Kuiper trace or the official Transformers eager implementation. They dump the
same prompt tokens, post-decoder hidden states, final normalized hidden state,
and logits on every side, then report the numerical error at every layer. They
also require the first 10 greedy generated tokens to match exactly.

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
  /tmp/qwen35-kuiper-cpu \
  --device cpu

./build/demo/qwen35_trace \
  /tmp/qwen35-0.8b-bf16.bin \
  /path/to/Qwen3.5-0.8B/tokenizer.json \
  /tmp/qwen35-kuiper-cuda \
  --device cuda

python3 tools/verify_qwen35/hf_reference.py \
  --model_dir /path/to/Qwen3.5-0.8B \
  --output_dir /tmp/qwen35-hf-trace

python3 tools/verify_qwen35/compare_traces.py \
  --kuiper_dir /tmp/qwen35-kuiper-cuda \
  --reference_dir /tmp/qwen35-kuiper-cpu

python3 tools/verify_qwen35/compare_traces.py \
  --kuiper_dir /tmp/qwen35-kuiper-cuda \
  --reference_dir /tmp/qwen35-hf-trace
```

`qwen35_trace` defaults to `--device cpu` for backward compatibility. In CUDA
mode it copies every observed tensor on the model-owned stream and synchronizes
that stream before reusing the activation buffer, so each file contains the
completed layer result rather than an in-flight device value. The selected
device is also recorded in `metadata.json`.

The default prompt is:

```text
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant
```

Pass a positional prompt after the output directory (before or after
`--device`) and `--prompt` to the Python reference when checking another prompt.
Both sides must receive exactly the same bytes.

The comparison fails if tokenization differs, any of the first 10 generated
tokens differs, or any hidden-state layer exceeds the configured relative tolerance.
Raw trace files are little-endian `int32`/`float32`; `metadata.json` records the
shapes required to read them.

The reference tool loads only the checkpoint's text tower, leaving the vision
and MTP weights on disk. It uses FP32 by default for strict numerical alignment.
For models that do not fit in host memory after FP32 widening, pass
`--dtype bf16`; trace files are still widened to FP32 before they are written.
Because Transformers rounds intermediate activations to BF16 while Kuiper keeps
them in FP32, use `--relative_tolerance 2e-2` when comparing that BF16 reference.
The generated-token sequence must still match exactly.

For the real 4B BF16-matrix checkpoint, the validated comparisons are:

```bash
./build/demo/qwen35_trace \
  /tmp/qwen35_4b_stage4_bf16.bin \
  /path/to/Qwen3.5-4B/tokenizer.json \
  /tmp/qwen35-4b-cuda \
  --device cuda

python3 tools/verify_qwen35/compare_traces.py \
  --kuiper_dir /tmp/qwen35-4b-cuda \
  --reference_dir /tmp/qwen35-4b-cpu \
  --relative_tolerance 2e-3

python3 tools/verify_qwen35/compare_traces.py \
  --kuiper_dir /tmp/qwen35-4b-cuda \
  --reference_dir /tmp/qwen35-4b-hf-bf16 \
  --relative_tolerance 2e-2
```
