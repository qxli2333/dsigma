# Single A100 validation and performance

These measurements describe the original GPU implementation at `ce39d36`,
before merging upstream 1.2.3. Upstream changed cosmology defaults, little-h
units and source preparation. The merge was checked with the CPU/GPU regression
suite; the complete real-catalog timing campaign has not been repeated for
1.2.3. The catalog benchmark runners in `examples/gpu_validation.py` and
`examples/gpu_cpu_benchmark.py` use the legacy input adapters and must be run
from the frozen checkout recorded here.

The complete calibrated type1/UNIONS 1.6.6 acceptance passed on 2026-10-29.
All five mass bins, 15 radial bins and 64 jackknife regions were tested for both
components. Every individual jackknife sample was compared before covariance.
GPU was faster than a single CPU process in this run; it did **not** outperform
the 256-process production CPU pipeline end to end.

## Frozen inputs and numerical gates

- 18,783 calibrated type1 lenses and all 88,638,229 UNIONS sources.
- Mass edges: 5.00, 7.47, 7.84, 8.16, 8.50, 11.00. Radial edges:
  `geomspace(0.02, 20, 16)/0.6766` comoving Mpc; Planck18 cosmology.
- Source columns: RA, Dec, `e1_leak_corrected`, `e2_leak_corrected`, `w_iv`;
  source mean redshift 0.66 and the existing UNIONS SOM-weighted redshift distribution.
- Seed 42; CPU-assigned lens regions reused on the GPU and in the independent
  production CPU comparison. Source array SHA-256 fingerprints match exactly.
- Pair counts match exactly. Each floating statistic passed
  `abs(error) <= 1e-10*max(1, abs(CPU sum))`, a gate at least as strict as
  `1e-10*max(1, sum(abs(pair contributions)))`.
- Signals, all samples and covariances passed `rtol=1e-8, atol=1e-10`, with
  identical NaN and signed Inf locations. Tolerances were not relaxed.
- The cross sign matches the CPU source rotation `(e_2, -e_1)`.

The read-only reference adapter has SHA-256 `738f9e1b86003c8d17cebc3aca0bb9114dce4e49ed7994e537493af22de8500c`.
[Machine-readable results](gpu_validation_summary.json) include versions, exact
input fingerprints, configuration, timings and artifact locations. Large source
files, numerical artifacts, environments and profiler captures remain under
`/pscratch/sd/q/qinxunli/dsigma_gpu_validation`.

## Measured timings

Times are seconds, measured with GPU synchronization around each GPU stage.
The GPU used one NVIDIA A100-PCIE-40GB on login35. The production CPU benchmark
used a Slurm CPU node with 256 allocated CPUs (job 60515378); its independent
memory run was job 60515627. BLAS and OpenMP each used one thread per process.
The main GPU pairing measurement did not overlap the later subset GPU regressions.
These are individual measurements on shared infrastructure, not repeated-trial
estimates of a universal speedup.

| Stage | Single CPU process | CPU, 256 processes | Single GPU |
| --- | ---: | ---: | ---: |
| Source read/preparation | 57.738 | 23.146 | 57.738 (CPU) |
| Tangential precompute | 117.637 | 39.954 | both components below |
| Cross precompute | 124.664 | 41.416 | included in same pass |
| Precompute, both components | 242.301 | 81.370 | 78.068 |
| Main pipeline, both components | 303.429 | 107.447 | 145.178 |

The main pipeline includes source reading, precomputation, CPU region assignment,
stacking and jackknife for all ten profiles; it excludes validation-only duplicate
computations and the separate random-catalog regression. CPU and GPU I/O timings
were taken on different nodes. Removing their respective source-read stages gives
245.691 s (single CPU),
84.301 s (256 CPU) and
87.440 s (GPU).
A separate 4-process tangential precomputation took 78.676 s.

Pair-kernel compilation into an initially empty scratch cache took
0.318 s. The first GPU stack took
1.123 s and the first GPU jackknife took
6.771 s, including additional CuPy JIT and library
initialization. Those cold costs are included in the GPU pipeline total.

The following synchronized stage timings use already-resident statistics. Each
row batches all 64 samples. The warm jackknife column includes region aggregation,
all sample calculations, covariance and final host result transfer. The first
warm repeat retained additional startup overhead; most later profiles took 5–9 ms.

| Mass bin | Lenses after pair filter | Region aggregation, ms | Batched stack, ms | Covariance, ms | Warm tangential jackknife, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5.00_7.47 | 2470 | 4.328 | 0.238 | 0.279 | 38.171 |
| 7.47_7.84 | 3989 | 4.482 | 0.206 | 0.366 | 5.767 |
| 7.84_8.16 | 4245 | 4.216 | 0.189 | 0.721 | 5.590 |
| 8.16_8.50 | 4180 | 4.021 | 0.167 | 0.287 | 5.943 |
| 8.50_11.00 | 3899 | 4.095 | 0.187 | 0.297 | 8.720 |

## Memory and profiler evidence

- Acceptance-process peak host RSS: 26.404 GiB.
  This includes CPU reference calculations and is not an isolated GPU-production
  memory measurement. The CPU reference shares source columns for the rotated
  component and releases the host preparation cache before that comparison.
- Peak CuPy pool reservation in the acceptance process:
  12.248 GiB, sampled every 0.1 s.
- Device-global sampled peak: 13.612 GiB;
  this includes contexts and overlapping subset-regression processes, so it is
  not a per-process allocation metric.
- Production CPU process-family peak PSS:
  15.376 GiB, sampled every 2 s in the separate
  allocated-node run. RSS and PSS are different metrics and are reported separately.

`jackknife_64_profile.nsys-rep` and its exported SQLite/CSV contain a cold and a
warm 64-region, 15-bin jackknife. Each run launches exactly one `grouped_finite`
kernel with 53 blocks of 128 threads, covering region × statistic × bin, followed
by one FP64 GEMM for covariance. There is no per-region Python stacking loop or
per-sample host result copy. The grouped kernel durations were 27.680 and
56.096 microseconds; covariance GEMMs were 7.776 and 7.008 microseconds. These
kernel durations exclude Python launch and profiler overhead and should not be
compared directly with the complete jackknife timings above.

## Regression coverage and reproducibility

The local suite passed **32 tests**, including the independent TreeCorr reference,
Astropy-bearing brute-force sums, Cython parity, correction combinations, every
jackknife sample, covariance, RA wrap, polar coordinates, bin edges, zero/invalid
weights, non-finite values, empty retained bins, non-contiguous regions, missing
random regions, secondary catalogs, more than 64 regions, streaming, chunk sizes,
FITS byte ordering and int64 counts above 2**53. The CPU remains the default.
Ruff passed and a wheel containing the CUDA source was built successfully.

The full-source acceptance has 47 named gates in addition to exact
pair-count assertions. An independent 256-process CPU run passed 30 profile/sample/
covariance comparisons across all ten outputs; its region labels also match.
The existing random catalog was tested with 1,000 UNIONS randoms and deterministic
shared region IDs (including regions absent from randoms). DES and DECADE use
fixed source subsets before preparation; their survey corrections are retained.

| Survey regression | Lenses | Prepared sources | Status |
| --- | ---: | ---: | --- |
| desy3 | 3,757 | 196,617 | passed |
| decade_ngc | 10,321 | 200,000 | passed |
| decade_sgc | 1,784 | 200,000 | passed |

Reproduce with `examples/gpu_validation.py`; omit `--source-limit` for the full
UNIONS catalog. Use `examples/gpu_cpu_benchmark.py` in a 256-CPU Slurm allocation
for the production comparison, with `PYTHONPATH` pointing to this checkout.
`examples/gpu_profile_jackknife.py` reproduces the 64-region profiler workload.
The source artifacts are recorded in each scratch report. Benchmark code hashes
and final reviewed code hashes are both retained; later table-column ordering,
import cleanup and balanced-tail batching were covered by targeted GPU tests and
do not change the measured 64-region array-return path.
