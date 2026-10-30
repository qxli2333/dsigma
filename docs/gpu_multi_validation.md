# Single-host multi-GPU validation

The multi-GPU pairing implementation adds `MultiGPUSession` without rerunning
historical survey acceptance. CPU remains the default; the existing single-GPU
session remains available.

## Completed checks

- Four-A100 job 60517411: all 37 GPU regression tests passed, including all
  dual-device cases, with no skips. The complete job finished successfully
  in 2 minutes 7 seconds on nid008336 (exit code 0).
- Coverage includes CPU/single/multi-session pair statistics, exact int64 counts,
  FP64 tolerances, calibration, tomography, source streaming, source reuse,
  original row order, cross profiles, every jackknife sample, covariance,
  device output, invalid devices, worker failures and session cleanup.
- CPU end-to-end regression tests (`tests/test_full.py`): 8 passed, including
  TreeCorr, brute-force sums, units, comoving conversion, calibration and n(z).
- Ruff and `git diff --check` passed. Importing the backend module leaves CuPy
  unloaded, preserving CPU-only import behavior.
- The synthetic benchmark completed its small smoke run and its default
  4,096-lens / 500,000-source workload on one A100. CPU subset statistics,
  full-catalog statistics, both profiles, all samples and covariances passed.

## Local single-device measurements

Seed 42, 15 physical radial bins, 32 jackknife regions, and three warm repeats.
Numbers below are median seconds on a shared login-node A100; they are not
multi-GPU speedup measurements. First-use costs and memory samples are retained
in the JSON artifact. Existing CUDA compilation caches were reused.

| Path | Precompute | Precompute plus both profiles and jackknife |
| --- | ---: | ---: |
| Existing `GPUSession` | 0.707 | 0.719 |
| `MultiGPUSession([0])` | 1.087 | 1.106 |

Worker scheduling and host gathering add overhead in the new session. Retain
`GPUSession` for ordinary single-GPU use. These timings do not predict scaling
on multiple GPUs or on larger scientific catalogs.

Artifacts are under `/pscratch/sd/q/qinxunli/dsigma_multi_gpu/`:
`single_gpu_benchmark.json`, `single_gpu_benchmark.log`, `smoke.json`, and
`local_multi_tests.log`. JSON records code SHA-256 hashes, versions, configuration,
first-use/warm timings and sampled per-device memory. Worker busy times are
instrumented only for the multi-session path and overlap across workers.

## Completed four-device validation

Slurm job **60517411** completed on a node with four NVIDIA A100-SXM4-80GB GPUs.
`benchmark.json` has `completed: true`; every correctness check passed for
all first-use and warm trials. Its recorded SHA-256 hashes identify the pre-optimization
GPU implementation, CUDA source, precompute module and benchmark script.
Subsequent host-task optimizations are tracked in `gpu_multi_optimization.md`.

The workload used the same seed, 4,096 lenses, 500,000 sources, 15 radial bins,
32 jackknife regions and three warm repeats. Warm median seconds on that node:

| Path | Precompute | Precompute plus both profiles and jackknife |
| --- | ---: | ---: |
| Existing `GPUSession`, 1 GPU | 0.576 | 0.586 |
| `MultiGPUSession`, 1 GPU | 0.689 | 0.699 |
| `MultiGPUSession`, 2 GPUs | 0.665 | 0.675 |
| `MultiGPUSession`, 4 GPUs | 0.681 | 0.691 |

Correctness passed, but this synthetic workload shows no speedup over the
existing single-device implementation. Two GPUs improve precompute latency by
about 3.4% relative to the new session using one GPU; four GPUs offer no further
improvement. Host scheduling and synchronization can limit these short tasks;
these measurements alone do not establish their relative contributions or
predict performance on larger catalogs. No historical survey acceptance was
rerun.

Sampled peak live CuPy pool use was about 74 MiB for the existing single-device
path, 69 MiB for the one-device multi session, 131–139 MiB per device for two
GPUs and 139 MiB per device for four GPUs. These are sampled live pool values,
not total reserved memory or isolated source-memory measurements.

The completed artifacts are `slurm-60517411.log` and `benchmark.json` under
`/pscratch/sd/q/qinxunli/dsigma_multi_gpu/`. Reproduce from the repository root:

```bash
mkdir -p /pscratch/sd/q/qinxunli/dsigma_multi_gpu
sbatch examples/gpu_multi_device.slurm
```

## Optimized production follow-up

Verification job **60521476** completed successfully after host-task
vectorization and geometry caching: **38 GPU tests passed with no skips**.
Uninstrumented synthetic comparisons at unchanged default chunk 128 reduced
four-device precompute from 0.628 to 0.186 seconds for 4,096 lenses / 500,000
sources, and from 7.042 to 0.805 seconds for 16,384 lenses / 2,000,000 sources.
All scientific comparison gates passed. See `gpu_multi_optimization.md` for
same-node baselines, full timings, scaling limits and artifact locations.
