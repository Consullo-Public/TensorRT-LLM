# Qwen3.8-27B DFlash2 evaluation suite

This directory contains the public evaluation tools used to select and reproduce the production concurrency policy. The scripts use Python's standard library and the TensorRT-LLM OpenAI-compatible API.

## Current tools

- `benchmark_concurrency.py` runs fixed-output concurrent decode trials and records raw per-request timing and aggregate completion tokens/second.
- `benchmark_controlled_matrix.py` synchronizes request starts; measures streaming TTFT, TPOT, end-to-end latency, and aggregate throughput; captures DFlash acceptance metrics; and records TensorRT-LLM's isolated context GPU-forward duration from `/perf_metrics`.
- `make_exact_prompts.py` generates semantically consistent prompts with exact post-chat-template token counts.
- `learn_spec_policy.py` learns request-shape policies only when repeated failure-free trials exceed a predefined throughput effect threshold.
- `compare_profiles.py` compares two result artifacts by batch size.
- `summarize_results.py` renders a result artifact as Markdown or CSV.
- `smoke_test_openai.py` checks `/v1/models` and performs a short deterministic completion.
- `collect_environment.py` captures non-secret software and GPU provenance.
- `verify_artifact_hashes.py` validates files against a SHA-256 manifest.

Additional safe concurrency-search and peak-memory tools will be added with the reconstructed production patch stack.

## Example

```bash
python examples/consullo/qwen38_dflash2/smoke_test_openai.py \
  --base-url http://127.0.0.1:5057

python examples/consullo/qwen38_dflash2/benchmark_concurrency.py \
  --label dflash2-production \
  --base-url http://127.0.0.1:5057 \
  --model qwen3.8-27b \
  --batches 1,2,4,8,12,15 \
  --output-tokens 512 \
  --trials 5 \
  --output results/dflash2-production.json

python examples/consullo/qwen38_dflash2/benchmark_controlled_matrix.py \
  --label dflash2-controlled \
  --base-url http://127.0.0.1:5057 \
  --model qwen3.8-27b \
  --batches 1,8,12,15 \
  --output-tokens 512 \
  --trials 5 \
  --output results/dflash2-controlled.json

python examples/consullo/qwen38_dflash2/learn_spec_policy.py \
  results/policy-learning \
  --minimum-trials 5 \
  --minimum-relative-effect 0.03 \
  --output results/policy-learning/learned-policy.json
```

The warmup is excluded from summaries. Aggregate decode throughput is:

```text
sum(completion tokens returned by every request) / batch wall-clock seconds
```

## Evidence rules

- Preserve raw trials and failures; do not report only the fastest run.
- Record the exact source commits, artifact hashes, model revisions, driver, CUDA, PyTorch, and GPU.
- Compare target-only and speculative configurations with the same prompts, output lengths, trial count, and concurrency.
- Compare system envelopes separately: the maximum safe target-only concurrency can be higher because DFlash draft weights and private context/KV caches consume VRAM.
- Use synchronized starts, the same shape warmup, and an exactly tokenized prompt when comparing request shapes.
- Treat `ctx_gpu_forward_time` as isolated context GPU-forward duration, not as total wall-clock prefill. TTFT additionally includes queueing, scheduling, transport, and first-token work.
- Require at least five failure-free trials and a 3% aggregate-throughput effect before changing a learned policy; retain smaller changes as `KEEP_BASELINE`.
- Do not claim that the `minsky` result generalizes to other machines or workloads.
- Never place API tokens, model credentials, private paths, or environment-variable values in result artifacts.
