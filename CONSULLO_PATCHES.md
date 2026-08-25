# Consullo Blackwell inference patches

## Purpose

This fork hosts independently installable TensorRT-LLM changes intended to improve effective NVIDIA Blackwell hardware utilization for Qwen3.8-27B inference. The planned production stack combines NVFP4 target and draft execution, DFlash2 speculative decoding, private-cache attention, Blackwell kernel corrections, and request-shape-aware CUDA graphs.

“Full Blackwell hardware utilization” is the optimization objective. Every release will state the actual execution paths and measured results rather than claiming universal saturation.

## Status

The upstream `main` branch remains available for synchronization. This public branch now contains the evaluation suite and the first production-derived reliability patches. The remaining composite Python overlay and locally built native wheel are still being reconstructed into reviewable commits.

No Consullo release tag is supported until a wheel is rebuilt from public tagged sources and passes the coordinated clean-install qualification on `minsky`.

## Planned patch series

1. Qwen3.8/Qwen3.5 quantized-checkpoint loader corrections.
2. SM120 attention-kernel correctness fixes.
3. DFlash2 core integration and private-cache attention.
4. Disabled-selector checkpoint filtering.
5. Dynamic CUDA graph eviction and recapture.
6. NVFP4 draft-transformer execution.
7. KV-cache lifecycle, backpressure, and runtime reliability fixes.
8. Versioned concurrency evaluation and environment capture.

Each patch will identify its provenance, tests, effect, and upstream issue or pull-request candidate.

## Controlled reference result

The August 25 controlled matrix on a GeForce RTX 5090 measured 1,154.85 median aggregate decode tokens/second at 15 concurrent requests, compared with 656.85 tokens/second for the target-only control. A request-shape selector at batch 12 further raised that shape from 1,067.79 to 1,284.55 tokens/second. These results are host- and workload-specific reference evidence, not a performance guarantee.

The public release must reproduce at least 1,097.11 tokens/second—the 95% threshold of the new batch-15 controlled median—without OOM, request loss, server death, or persistent KV-cache leakage.

The V2 scheduler now treats main and DFlash-private generation-cache allocation as one transaction and does not declare deadlock while completion-pending requests can release pages. This fixes the observed 15-request tail failure without requiring host-cache offload.

## Related projects

- [Public installer and release documentation](https://github.com/Consullo-Public/qwen38-dflash2-tensorrt-llm)
- [Consullo FlashAttention fork](https://github.com/Consullo-Public/flash-attention)
- [Upstream DFlash](https://github.com/z-lab/dflash)
- [Upstream TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

This is a community-maintained fork and is not supported or endorsed by NVIDIA, Alibaba/Qwen, Z Lab, or the upstream maintainers.
