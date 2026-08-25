# Consullo public-fork addendum

These instructions supplement the TensorRT-LLM upstream instructions below for work in `Consullo-Public/TensorRT-LLM`. Read `llms.txt`, `CONSULLO_PATCHES.md`, and `examples/consullo/qwen38_dflash2/README.md` before changing the Consullo patch stack, evaluation tools, benchmark claims, or release metadata.

For Consullo-only files and branches, this addendum takes precedence when an upstream instruction assumes NVIDIA authorship, NVIDIA infrastructure, or an upstream pull-request destination. Continue following upstream coding, testing, and DCO requirements wherever they apply.

## Fork mission and status

- Optimize Qwen3.8-27B inference on NVIDIA Blackwell consumer GPUs using an NVFP4 target, NVFP4 DFlash2 draft transformer, private-cache attention, CUDA-graph policy, and KV-cache reliability changes.
- Draft NVFP4 is intended to reduce weight residency so more RTX 5090 VRAM remains available for the KV cache and concurrent requests.
- The default `codex/public-landing` branch is not a qualified production release.
- The historical production runtime includes changes not yet reconstructed in public source. Never claim source equivalence until hashes, builds, and Minsky requalification prove it.

## Required Consullo workflow

1. Establish the upstream base and actual public branch before editing.
2. Preserve upstream changes and keep Consullo changes in narrow, reviewable commits.
3. Identify whether a change belongs to model loading, DFlash2, private-cache attention, CUDA graphs, native kernels, KV-cache lifecycle, runtime reliability, or evaluation.
4. Record provenance, tests, correctness implications, performance hypothesis, rollback, and candidate upstream disposition.
5. Run the smallest deterministic CPU/unit checks first.
6. Run GPU tests only with explicit access authorization and an available model cache.
7. Preserve raw results and failures; never select only the fastest trial.

## Consullo benchmark rules

- The reference GPU is an NVIDIA GeForce RTX 5090, Blackwell SM120.
- The published workload uses 102–104 input tokens and 512 forced decode tokens per request.
- Aggregate decode throughput is total returned completion tokens divided by concurrent-batch wall time.
- The historical five-trial medians are 736.39 tok/s at 8 requests, 972.27 at 12, and 1126.77 at 15; target-only at 15 is 770.46.
- Do not infer long-context performance, quality equivalence, a universal concurrency optimum, or per-request throughput from these values.
- Changes to `examples/consullo/qwen38_dflash2/` must keep result formats versioned, preserve failures, sanitize credentials and unique identifiers, and avoid service mutation.

## Public-release constraints

- Do not publish checkpoints, private paths, credentials, backup files, generated binaries, hydrated LFS objects, or unreconstructable wheels.
- Newly authored files require accurate authorship; do not automatically apply an NVIDIA copyright to independently authored Consullo code.
- Preserve Apache-2.0 notices and all notices required by derived DFlash MIT code.
- A supported tag requires public-source builds, hashes, SBOM, public-only installation, exact model revisions, rollback tests, and the acceptance criteria in `llms.txt`.
- Never attach a privileged self-hosted GPU runner to public pull-request workflows.
- GPU qualification on `minsky` requires explicit owner coordination and exclusive access. Do not stop or relocate the running stack without approval and a verified recovery snapshot.

## Git destinations

- NVIDIA upstream: `https://github.com/NVIDIA/TensorRT-LLM`
- Consullo public fork: `https://github.com/Consullo-Public/TensorRT-LLM`
- Keep the upstream synchronization branch free of Consullo-only commits.
- Use protected Consullo feature or release branches and pull requests for publication work.

---

# TensorRT-LLM upstream agent instructions

TensorRT-LLM: open-source library for optimized LLM inference on NVIDIA GPUs.
Python and C++ codebase with PyTorch and AutoDeploy execution paths.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it — it contains developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**
- Read and follow `CODING_GUIDELINES.md` for ALL code changes (C++ and Python)
- NVIDIA copyright header on ALL new files (update year on modified files)
- `git commit -s` (DCO sign-off required). Never attribute AI tools in sign-off line. Always rely on `git` to do the sign off instead of directly adding sign off in commit message.
- Do not add co-authors to the git commit message unless explicitly instructed to do so by the user.
- `pre-commit` hooks run on commit — if files are modified by hooks, re-stage and commit again
- LLM args or nested-config changes must run `python3 scripts/generate_llm_args_golden_manifest.py` and commit
  `tensorrt_llm/usage/llm_args_golden_manifest.json`; new fields require telemetry/privacy CODEOWNER approval
- PR title format: `[JIRA/NVBUG/None][type] description` (e.g., `[TRTLLM-5516][perf] optimize cuda graph padding`)
- Set `LLM_MODELS_ROOT` env var when running tests that need model weights

## Common Commands

| Task | Command |
|------|---------|
| Unit tests | `pytest tests/unittest/` |
| Specific test | `pytest tests/unittest/llmapi/test_llm_args.py` |
| Pattern match | `pytest tests/unittest -k "test_llm_args"` |
| Integration tests | `LLM_MODELS_ROOT=/path/to/models pytest tests/integration/defs/...` |
| Serve model | `trtllm-serve <hf_model> --port 8000` |
| Serve with config | `trtllm-serve <hf_model> --config config.yaml` |
| Benchmark | `trtllm-bench --model <hf_model> throughput --dataset <path>` |
| Find CI stage for test | `python scripts/test_to_stage_mapping.py --tests "test_name"` |

### Installation & Build

Building TensorRT-LLM requires Docker and may involve compiling C++ components.
See the [Installation Guide](docs/source/installation/installation-guide.md) for pre-built release containers and pip install,
[build from source](docs/source/installation/build-from-source.md) for development builds,
and [Container Images](docs/source/installation/containers.md) for information about the container images.

### Reference Configs

`examples/configs/database/` contains pareto-optimized serving configurations
across multiple models, GPUs, ISL/OSL combinations, and concurrency levels.
Use these as starting points for deployment and benchmarking rather than hand-tuning parameters.
See [deployment guides](docs/source/deployment-guide/) for model-specific walkthroughs.

## Architecture

See [architecture diagram](.github/tava_architecture_diagram.md) for the full Mermaid diagram.

### Backends

| Backend | Status | Entry Point | Key Path |
|---------|--------|-------------|----------|
| **PyTorch** | Default | `TorchLlmArgs` | `_torch/pyexecutor/` → `PyExecutor` → PyTorch Engine |
| **AutoDeploy** | Beta | `_torch/auto_deploy/` shim | `_torch/auto_deploy/shim/ad_executor.py` → adapts `PyExecutor` → graph transforms + torch.export |

### Shared C++ Core (via Nanobind)

Both backends share these C++ components:
- **Scheduling pipeline**: Scheduler → BatchManager (in-flight batching) → KV Cache Manager
- **Decoding pipeline**: Decoder (token generation orchestration) → Sampling

### Request Flow
```text
HuggingFace Model → LLM API → Executor (PyTorch/AutoDeploy)
    → Scheduler → Model Forward → Decoder → Sampling → Generated Tokens
```

### Serving
- `trtllm-serve`: OpenAI-compatible REST + gRPC server, supports all backends
- **Disaggregated serving**: separates prefill (context) and decode (generation) across GPUs
  - KV cache exchange via NIXL (default), UCX, or MPI

## Key Files

| File | Role |
|------|------|
| `tensorrt_llm/llmapi/llm.py` | Main API entry point |
| `tensorrt_llm/llmapi/llm_args.py` | Complete configuration schema (Pydantic) |
| `tensorrt_llm/llmapi/llm_utils.py` | Model loading, model-specific default overrides |
| `tensorrt_llm/models/modeling_utils.py` | Base classes for all models (`PretrainedConfig`, `PretrainedModel`) |
| `tensorrt_llm/executor/executor.py` | Execution abstraction (`GenerationExecutor`) |
| `tensorrt_llm/models/automodel.py` | Auto-discovery and model registry |
| `tensorrt_llm/_torch/models/` | PyTorch backend model implementations (distinct from the top-level `models/` package) |
| `tensorrt_llm/_torch/modules/ATTENTION_DEVELOPER_GUIDE.md` | Attention, MLA, backend families, sparse backends, metadata contracts, and KV-cache behavior - **read before modifying `tensorrt_llm/_torch/modules/attention.py`, `tensorrt_llm/_torch/modules/mla.py`, or `tensorrt_llm/_torch/attention_backend/`** |
| `tensorrt_llm/_torch/modules/fused_moe/MOE_DEVELOPER_GUIDE.md` | MoE architecture, backends, communication, development patterns — **read before modifying MoE code** |
| `CODING_GUIDELINES.md` | C++ and Python coding standards (referenced throughout, must read before contributing) |

## Design Patterns

| Pattern | Key Points |
|---------|------------|
| **Config hierarchy** | `BaseLlmArgs` → `TorchLlmArgs`, model-specific defaults override generics, Pydantic validation |
| **Model architecture** | Each model: `Config` (inherits `PretrainedConfig`) + `ForCausalLM` (inherits `PretrainedModel`) |
| **Model defaults** | Architecture-specific overrides in `llm_utils.py` (attention kernels, quant, spec decoding, cache) |
| **Attention backends** | `TorchLlmArgs.attn_backend` selects kernel: `TRTLLM` (default), `FlashInfer`, `FlashAttention` |
| **Distributed execution** | Tensor/pipeline parallelism via `Mapping` class, multiple backends (MPI, Ray, RPC) |
| **Auto-discovery** | Models self-register via `automodel.py`, resolved by HF config `architectures` field |

## VisualGen

VisualGen is a vertical alongside LLM for Diffusion-Transformer (DiT)-based image/video generation
(text-to-image, text-to-video, image-to-video). It is **not** an LLM backend — it has
its own engine, args, params, and outputs — but shares ops and kernels with the
PyTorch backend where it makes sense (attention, quantization, parallelism).

Key entry points:
- Public Python API: `from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams`.
- Serving CLI: `trtllm-serve --model <HF id> --visual_gen_args <YAML path>`.

Key files:
- `tensorrt_llm/_torch/visual_gen/ENGINEERING_CRITERIA.md`: **Engineering criteria for any change under `tensorrt_llm/visual_gen/` or `tensorrt_llm/_torch/visual_gen/`** — API discipline, feature/test/lossy-vs-lossless requirements, examples & docs rules. Read before modifying anything in those trees.
- `tensorrt_llm/visual_gen/`: VisualGen public Python API. **User-facing surface — before modifying anything here, pause and confirm with the user that a public API change is actually intended; do not infer it from the surrounding task.**
- `tensorrt_llm/_torch/visual_gen/`: VisualGen internal implementation. All non-user-facing code belongs here.

## Anti-Patterns / Gotchas

- **Pre-commit modifies files in-place** — if hooks fail, files are already modified. Re-stage (`git add`) and commit again.
- **Protected APIs exist** — changes to LLM API signatures will fail `tests/unittest/api_stability` tests. Get code owner review.
- **Integration tests need GPUs + models** — always set `LLM_MODELS_ROOT` and ensure GPU access. Unit tests don't.
- **Copyright year** — update to current year when modifying existing files; add full header to new files.
- **Avoid broad exception handling** — catch specific exceptions, not bare `except:` (see `CODING_GUIDELINES.md`).
- **One concern per PR** — avoid scope creep. If a PR touches unrelated areas, split it.
- **User-facing configuration classes** - when editing or defining any user-facing configuration classes (particularly `BaseLlmArgs` or any class used in its fields), you **MUST** follow the Pydantic guidelines in `CODING_GUIDELINES.md`.

## Development Workflow

1. Set up build environment (see [installation docs](docs/source/installation/))
2. Make changes following `CODING_GUIDELINES.md`
3. Test locally with `pytest`

## Branching policy and PRs

- The main repository (`upstream`) is located at https://github.com/NVIDIA/TensorRT-LLM/
- Branches should always be pushed to the user-specified fork (usually `origin`)
- If pushing fails to due pre-push pre-commits hooks getting updated, just re-push immediately
- PRs should be opened on the main repository
   - Target `main` unless fixing a release branch bug
   - See `CONTRIBUTING.md` for full PR policies

### GitHub CLI authentication (`GH_CONFIG_DIR`)

The `gh` CLI uses `~/.config/gh` by default for authentication. Different GitHub hosts or forks may require a different config directory. **Before running any `gh` command** (e.g., `gh pr create`, `gh api`, `gh pr comment`):

1. Check if the user has specified a custom `GH_CONFIG_DIR` (e.g., in `CLAUDE.local.md` or environment). If so, use it.
2. If not explicitly set, default to `~/.config/gh`; do not ask for confirmation.
3. Prefix all `gh` commands with the resolved config dir: `GH_CONFIG_DIR=<path> gh ...`
4. If the command fails due to missing authentication or the wrong GitHub host/account, report the failure and ask for the correct `GH_CONFIG_DIR`.

## CI / Testing

See [CI overview](docs/source/developer-guide/ci-overview.md) for full details.

| Layer | Location | Notes |
|-------|----------|-------|
| Unit tests | `tests/unittest/` | Run in pre-merge CI; some tests require GPU |
| API stability | `tests/unittest/api_stability/` | Protects committed API signatures |
| Integration tests | `tests/integration/defs/` | Requires GPU + `LLM_MODELS_ROOT` |
| Test lists | `tests/integration/test_lists/test-db/` | Per-GPU YAML files (`l0_a10.yml`, `l0_h100.yml`, etc.) |
| Test waives | `tests/integration/test_lists/waives.txt` | Skip known-failing tests with NVBug links |
| Performance | See [benchmarking guide](docs/source/developer-guide/perf-benchmarking.md) | `trtllm-bench` and `trtllm-serve` benchmarks |

### Triggering CI

CI is triggered by posting comments on the PR. Basic commands:
- `/bot run` — trigger the standard CI pipeline
- `/bot run --disable-fail-fast` — run all stages even if earlier ones fail (only add when explicitly needed)
- `/bot run --extra-stage "DGX_B200-4_GPUs-AutoDeploy-1, DGX_H100-4_GPUs-AutoDeploy-1"` — include AutoDeploy CI stages (use for AutoDeploy-related PRs)

For a full list of up-to-date bot commands, post `/bot help` as a PR comment and check the bot's reply.

### Trouble Shooting

- Use `TLLM_LOG_LEVEL_BY_MODULE` to enable per-module log filtering (e.g., `"debug:_torch,runtime;info:serve"`); see [Module-Level Logging](docs/source/developer-guide/overview.md#module-level-logging) for details.

## Key Documentation

| Topic | Path |
|-------|------|
| Coding guidelines | `CODING_GUIDELINES.md` |
| Architecture overview | `docs/source/developer-guide/overview.md` |
| PyTorch backend | `docs/source/torch/arch_overview.md` |
| Adding a new model | `docs/source/torch/adding_new_model.md` |
| AutoDeploy | `docs/source/features/auto_deploy/auto-deploy.md` |
| Disaggregated serving | `docs/source/features/disagg-serving.md` |
| Speculative decoding | `docs/source/features/speculative-decoding.md` |
| Quantization | `docs/source/features/quantization.md` |
| Parallelism strategies | `docs/source/features/parallel-strategy.md` |
| KV cache | `docs/source/features/kvcache.md` |
| API change guidelines | `docs/source/developer-guide/api-change.md` |
| Feature compatibility matrix | `docs/source/features/feature-combination-matrix.md` |
| Supported models | `docs/source/models/supported-models.md` |
| Deployment guides | `docs/source/deployment-guide/` |
| Examples & customization | `docs/source/examples/` |
| Performance analysis | `docs/source/developer-guide/perf-analysis.md` |
