# Request-shape speculative policy experiment

This directory records controlled trials on `minsky` on August 25, 2026, using one NVIDIA GeForce RTX 5090 (Blackwell, SM120), Qwen3.8-27B target weights in NVFP4, FP8 target KV cache, and the DFlash2 draft transformer dynamically converted to NVFP4. All recommendation-eligible rows use an exactly tokenized prompt, synchronized request starts, deterministic decoding, one excluded shape warmup, and five measured failure-free trials.

The production-derived prompt sizes are 3,200 and 3,900 tokens (the observed typical range), 4,000 tokens (the representative benchmark), and 16,000 tokens (the source-heavy stress case). Outputs are fixed at 128 tokens for 3.2K–4K inputs and 64 tokens for 16K inputs.

## Maximum-aggregate system envelopes

These rows compare the best capacity-qualified concurrency of each profile, rather than forcing equal batches when DFlash's draft and private caches reduce available VRAM.

| Input tokens | Profile | Safe concurrent requests | Aggregate decode tok/s p50 | TTFT p50 / p95 (s) | GPU context-forward p50 / p95 (ms) | Accepted length | Acceptance rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3,200 | Target-only | 15 | **256.638** | 2.734 / 4.351 | 358.56 / 363.32 | — | — |
| 3,200 | DFlash2 K6 | 4 | 226.709 | 1.078 / 1.289 | 280.62 / 393.60 | 3.239 | 37.32% |
| 3,900 | Target-only | 15 | **223.545** | 3.166 / 5.407 | 367.88 / 370.26 | — | — |
| 3,900 | DFlash2 K6 | 4 | 186.291 | 1.192 / 1.572 | 384.45 / 401.55 | 3.534 | 42.23% |
| 4,000 | Target-only | 15 | **218.763** | 3.182 / 5.583 | 369.66 / 372.44 | — | — |
| 4,000 | DFlash2 K6 | 4 | 195.423 | 1.192 / 1.600 | 385.75 / 400.06 | 3.607 | 43.45% |
| 16,000 | Target-only | 8 | 29.033 | 9.275 / 16.130 | 1,893.46 / 1,902.67 | — | — |
| 16,000 | DFlash2 K6 | 2 | 29.050 | 3.018 / 3.943 | 1,771.43 / 2,029.03 | 4.828 | 63.79% |

At 3.2K, 3.9K, and 4K, target-only wins maximum aggregate throughput by 13.2%, 20.0%, and 11.9%, respectively, because it sustains batch 15 while the long-context DFlash profile sustains batch 4. At 16K, the aggregate rates differ by only 0.06%, far below the predefined 3% policy threshold; target-only sustains four times the concurrency, while DFlash has much lower TTFT. The aggregate-throughput learner therefore keeps target-only as the baseline for these system envelopes. A latency-first router could make a different explicit choice.

## Prefill interpretation and NVFP4 confirmation

TensorRT-LLM selected `trtllm::nvfp4_gemm::gemm` during profile startup on SM120. This confirms that NVFP4 linear-layer matrix multiplications in the target model use the Blackwell hardware-backed TensorRT-LLM path during both prefill and decode. Attention, normalization, sampling, and other non-GEMM operations are not NVFP4 GEMMs.

The benchmark drains `/perf_metrics` before each trial and records `time_breakdown_metrics.ctx_gpu_forward_time` afterward. The table reports that isolated GPU context-forward duration separately from client TTFT. It must not be described as total wall-clock prefill: TTFT also includes queueing, chunk scheduling, transport, and first-token work.

## Draft-length experiment

With the earlier exactly 90-token input and 512-token output at fixed concurrency, K4 improved batch-12 throughput from the K7 baseline of 1,284.547 to **1,389.144 tok/s (+8.14%)**. K6 improved batch-15 throughput from 1,154.855 to **1,252.897 tok/s (+8.49%)**. Both exceed the 3% effect threshold across five trials; K5 was dominated. These results establish that accepted-position survival can justify shorter drafts, but only within the tested short-prompt shape.

## Capacity failures retained as evidence

- A DFlash long-context profile configured for batch 15 could not start because its dense private context-cache allocation requested approximately 5 GiB after the other model allocations.
- The batch-4/max-token-2048 DFlash profile starts and completes 3.2K–4K trials. At 16K, batch 2 completes, but a prior batch-4 capacity probe failed with `Draft KV cache context resize failed ... could not resize to 16005 tokens` and terminated the executor loop. The failing capacity probe is retained; it is not used as a performance result.
- One-trial capacity probes are retained to document feasibility boundaries but are excluded by `learn_spec_policy.py`, which requires at least five successful trials and zero failed trials.

The next high-value implementation opportunity is a lazily allocated or page-backed DFlash private context cache. The current dense allocation scales with configured maximum batch and sequence length, which can erase the aggregate benefit of otherwise high speculative acceptance. This is an implementation hypothesis, not a measured improvement.

## Reproduction and policy gate

`learned-policy.json` was produced with:

```bash
python examples/consullo/qwen38_dflash2/learn_spec_policy.py \
  examples/consullo/qwen38_dflash2/results/minsky-rtx5090-2026-08-25-policy-learning \
  --minimum-trials 5 \
  --minimum-relative-effect 0.03 \
  --output examples/consullo/qwen38_dflash2/results/minsky-rtx5090-2026-08-25-policy-learning/learned-policy.json
```

The raw JSON retains every request observation, server timing record, failure, DFlash acceptance counter, and full-precision summary. The prompt manifest records exact post-chat-template token counts without exposing local paths.
