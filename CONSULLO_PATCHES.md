# Consullo Blackwell inference patches

## Purpose

This fork hosts independently installable TensorRT-LLM changes intended to improve effective NVIDIA Blackwell hardware utilization for Qwen3.8-27B inference. The planned production stack combines NVFP4 target and draft execution, DFlash2 speculative decoding, private-cache attention, Blackwell kernel corrections, and request-shape-aware CUDA graphs.

“Full Blackwell hardware utilization” is the optimization objective. Every release will state the actual execution paths and measured results rather than claiming universal saturation.

## Status

The upstream `main` branch remains available for synchronization. This public landing branch contains documentation only. The proven production patch stack is being reconstructed into reviewable commits because the running system includes a composite Python overlay and a locally built native wheel that are not completely represented by the earlier development branch.

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

## Reference result

The existing private production baseline on a GeForce RTX 5090 selected 15 concurrent requests. Its median aggregate decode rate was 1126.77 tokens/second, compared with 770.46 tokens/second for the target-only control under the recorded workload. These results are host- and workload-specific reference evidence, not a performance guarantee.

The first public release must reproduce at least 1070.43 tokens/second—the 95% acceptance threshold—without OOM, request loss, server death, or persistent KV-cache leakage.

## Related projects

- [Public installer and release documentation](https://github.com/Consullo-Public/qwen38-dflash2-tensorrt-llm)
- [Consullo FlashAttention fork](https://github.com/Consullo-Public/flash-attention)
- [Upstream DFlash](https://github.com/z-lab/dflash)
- [Upstream TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

This is a community-maintained fork and is not supported or endorsed by NVIDIA, Alibaba/Qwen, Z Lab, or the upstream maintainers.

