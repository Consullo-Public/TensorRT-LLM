# Minsky RTX 5090 reference results — 2026-08-21

These are raw historical benchmark artifacts produced by the original evaluation harness. They are preserved without retroactively changing their schema or measurements.

| Profile | Batch | Trials | Median tok/s | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Target only | 15 | 5 | 770.46 | 770.25 | 768.52 | 771.80 |
| DFlash2 NVFP4 draft | 8 | 5 | 736.39 | 716.68 | 628.56 | 757.31 |
| DFlash2 NVFP4 draft | 12 | 5 | 972.27 | 965.17 | 927.68 | 987.03 |
| DFlash2 NVFP4 draft | 15 | 5 | 1126.77 | 1137.53 | 1093.36 | 1199.87 |

At batch 15, the DFlash2 configuration improved the median aggregate decode rate by 46.25% relative to the target-only control. The measurements used 512 forced completion tokens per request and a deterministic temperature-zero workload.

## Interpretation limits

- The target-only artifact covers batch 15; it is not a complete target-only concurrency sweep.
- These results establish behavior for the recorded host and workload, not a universal optimum.
- The original artifact schema predates `format_version` and `created_at` fields.
- The first supported public release must repeat the clean-install evaluation and publish new versioned artifacts.

See `environment.json` for software, hardware, source, and checkpoint provenance known from the production record.

