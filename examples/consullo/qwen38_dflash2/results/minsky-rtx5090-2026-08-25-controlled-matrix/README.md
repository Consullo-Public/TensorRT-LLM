# Controlled target-only and DFlash2 matrix

This directory records a controlled comparison performed on `minsky` on August 25, 2026. The GPU was an NVIDIA GeForce RTX 5090 (Blackwell, SM120). The target and DFlash2 profiles used the same exactly 90-token prompt, synchronized request starts, temperature `0`, `ignore_eos=true`, exactly 512 output tokens per request, one excluded 64-token shape warmup, and five measured trials per batch size.

## Results

| Concurrent requests | Target-only aggregate tok/s | DFlash2 aggregate tok/s | Gain | DFlash2 client tok/s | Target / DFlash2 TTFT (ms) | Acceptance rate | Accepted length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46.00 | 118.63 | +157.88% | 118.63 | 119.85 / 94.40 | 22.57% | 2.580 |
| 8 | 361.45 | 885.28 | +144.93% | 110.67 | 125.22 / 115.73 | 20.45% | 2.431 |
| 12 | 529.67 | 1,067.79 | +101.60% | 89.00 | 152.56 / 154.29 | 18.08% | 2.265 |
| 15 | 656.85 | 1,154.85 | +75.82% | 92.27 | 157.68 / 157.98 | 23.50% | 2.645 |

Values are five-trial medians. Aggregate decode throughput is total decoded tokens divided by concurrent-batch wall time. Client throughput is the median individual request rate. TTFT is measured from HTTP request release to the first streamed content or reasoning token.

The acceptance metrics do not support the hypothesis that speculative decoding stops paying once batching fills this GPU. DFlash2 remained 75.82% faster at batch 15, and accepted length was slightly higher at batch 15 than batch 1. Per-client throughput fell from 118.63 to 92.27 tokens/second from batch 1 to 15, while aggregate throughput grew 9.74 times.

## Request-shape selector experiment

| Shape | Selector policy under test | Baseline tok/s | Test tok/s | Change | Accepted length change | TTFT change |
|---:|---|---:|---:|---:|---:|---:|
| 8 | `8,12,15` | 885.28 | 887.17 | +0.21% | 2.431 → 2.450 | 115.73 → 116.43 ms |
| 12 | `12,15` | 1,067.79 | **1,284.55** | **+20.30%** | 2.265 → 2.768 | 154.29 → 154.81 ms |

The batch-12 change is large, stable (1,283.94–1,286.40 tok/s), and accompanied by higher acceptance. The batch-8 change is too small to distinguish from normal run variation. The evidence-backed production policy is therefore `TRTLLM_DFLASH2_SELECTOR_BATCH_SIZES=12,15`.

## Production-policy percentiles

The production-policy view combines the selector-enabled batch-12 artifact with batches 1, 8, and 15 from the post-fix matrix. Batch 15 already used the selector. Every row retains the identical 90-token prompt and 512-token forced output.

### Per-client latency

| Concurrent requests | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms/token) | E2E p50 / p95 (ms) |
|---:|---:|---:|---:|
| 1 | 94.40 / 104.85 | 8.261 / 8.265 | 4,315.79 / 4,327.35 |
| 8 | 115.73 / 117.48 | 8.827 / 8.833 | 4,626.23 / 4,630.44 |
| 12 | 154.81 / 156.59 | 9.053 / 9.061 | 4,782.12 / 4,784.49 |
| 15 | 157.98 / 160.88 | 10.550 / 12.570 | 5,548.93 / 6,582.41 |

TTFT, TPOT, and E2E percentiles use the individual client-request observations pooled across all five trials for each batch size.

### Speculation and throughput

| Concurrent requests | Acceptance length p50 / p95 | Acceptance rate p50 / p95 | Aggregate decode tok/s p50 / p95 |
|---:|---:|---:|---:|
| 1 | 2.580 / 2.586 | 22.57% / 22.66% | 118.63 / 118.69 |
| 8 | 2.431 / 2.431 | 20.45% / 20.45% | 885.28 / 886.41 |
| 12 | 2.768 / 2.768 | 25.25% / 25.25% | 1,284.55 / 1,286.31 |
| 15 | 2.645 / 2.646 | 23.50% / 23.52% | 1,154.85 / 1,167.56 |

Acceptance and aggregate-throughput percentiles use the five trial-level observations. For throughput and acceptance, p95 is the favorable high tail; for latency, p95 is the unfavorable slow tail. Values are rounded only for presentation; the JSON artifacts retain full precision.

## V2 dual-cache failure and fix

Before the fix, a synchronized batch of fifteen 512-token requests reached a tail state in which thirteen completion-pending requests retained KV pages and two generation requests were deferred. The V2 scheduler raised a false deadlock and terminated the executor. Host-cache offload and a larger KV fraction did not correct it.

Two lifecycle defects were fixed:

1. The scheduler now reserves main and DFlash-private generation KV capacity atomically and rolls the main allocation back if the draft allocation cannot fit. The later draft preparation step recognizes this reservation instead of allocating the same draft length twice.
2. Deadlock detection now treats in-flight, completion-pending, evicted, and recompute-paused requests as forward-progress paths. Completion-pending requests can release pages later in the executor cycle and must not trigger a deadlock.

After the patch, all five batch-15 controlled trials completed. Their median was 1,154.85 aggregate tok/s and their range was 1,149.18–1,169.75 tok/s.

`dflash2-production.json` preserves the pre-fix failed run. Its zero-token summaries are failure evidence, not performance measurements. `dflash2-production-patched.json` is the valid post-fix controlled matrix.

## Artifacts

- `target-only.json`: complete target-only matrix.
- `dflash2-production-patched.json`: complete post-fix DFlash2 matrix.
- `dflash2-selector-b12.json`: selector-enabled batch-12 trials.
- `dflash2-selector-b8.json`: selector-enabled batch-8 trials.
- `dflash2-production.json`: pre-fix batch-15 failure evidence.

Raw artifacts include every request's TTFT, end-to-end latency, decoded-token count, completion status, and Prometheus counter deltas. No fastest-run selection was used.

## Applicability to longer contexts

This is a decode-heavy, short-context benchmark: 90 input tokens are short, while 512 forced output tokens are not. It strongly validates steady-state decode throughput, speculative acceptance on this prompt family, and the V2 cache lifecycle at the chosen shapes. It does not establish that concurrency 15, the `12,15` selector policy, or the current KV-cache fraction is optimal for 1K–8K-token inputs.

The NVFP4 compute path, DFlash private-cache attention, and dual-cache correctness fix should remain applicable at longer contexts. Shape policy is less portable: longer prompts increase prefill work and KV occupancy, can lower the safe concurrent-request count, and may change accepted length. The measured production profile currently uses `max_input_len=1024` and `max_seq_len=4096`; it is not a 4K–8K-input profile. Broad qualification should first test inputs `90,512,1024` under the current limits, then use a separately capacity-qualified profile to test `4096,8192`. Cross those input lengths with output lengths `128,512` and feasible concurrency levels while separately reporting TTFT, inter-token latency, aggregate decode rate, acceptance length, peak VRAM, and request success. Until that matrix is run, `12,15` is a production policy for this measured decode-heavy envelope rather than a universal optimum.

## Remaining opportunities

The next useful experiment is a policy learner keyed by request shape, prompt length, and observed accepted-position survival rather than batch size alone. It should change policy only after repeated trials exceed a predefined effect-size threshold. A second experiment should test draft lengths below seven at batch 12 and 15: the last draft position survives only about 5–6% of proposals, so a shorter draft may reduce wasted verification work. Neither is proven by this matrix.
