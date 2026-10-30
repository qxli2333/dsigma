# GPU optimization findings

The validated host-task optimizations have now been applied to production
GPU pairing. The original experiments and raw results live under
`/pscratch/sd/q/qinxunli/dsigma_multi_gpu/optimization/`.

## Highest priority: host task construction

`GPUSession.pair` builds `(lens, source-start, source-end)` tuples in nested
Python loops and converts the list to an int64 array at every flush. A profiled
4,096-lens / 500,000-source call recorded about 1.47 million list appends and
1.48 million calls to `len`. Under cProfile, list-to-array conversion accumulated
0.437 seconds; pixel-center conversion accumulated 0.240 seconds. Profiler
instrumentation changes timings, so these values describe hot paths rather
than an additive decomposition of uninstrumented wall time.

CUDA events measured approximately 106 ms across 32 pair-kernel launches.
This excludes reduction kernels and transfers and must not be interpreted as
total GPU time. Together with the prototype results below, the measurements
identify host preparation/dispatch as a major optimization target.

An isolated prototype replaces tuple appends with a preallocated NumPy int64
buffer filled by source-range slices. It preserves the original flush
boundaries and source order. The prototype also converts HEALPix angles to raw
radians before trigonometric operations, reducing Quantity wrapper overhead.
The CUDA kernel, FP64 policy, count dtype and weighting rules are unchanged.

Same shared A100-PCIE-40GB host, seed 42, 4,096 lenses, 500,000 sources,
three warm repeats; median total precompute seconds:

| Implementation | lens_chunk | Seconds | Speed relative to original, chunk 128 |
| --- | ---: | ---: | ---: |
| Original multi session, one GPU | 128 | 1.176 | 1.00x |
| Original multi session, one GPU | 512 | 1.079 | 1.09x |
| Vectorized tasks plus numeric angles | 128 | 0.467 | 2.52x |
| Vectorized tasks plus numeric angles | 512 | 0.370 | 3.18x |

All experiment outputs passed the existing count and floating-statistic gates.
With the prototype injected in memory, the GPU regression suite passed:
**34 passed, 3 dual-device cases skipped** on this one-GPU host. The two host
optimizations were measured together; their individual benefits have not been
isolated. Shared-node timings have noise and do not establish multi-GPU scaling.

Implementation priority: vectorize task construction and numeric pixel geometry,
then retain the full scientific regression suite, including source streaming and
flush boundaries. Python thread contention is a plausible scaling limit;
actual GIL wait time has not been measured.

## Dispatch granularity and repeated setup

Every outer lens task invokes the complete single-device pairing method,
including pixel queries, unit conversions, lens uploads and output allocation.
At chunk 128 there are 32 outer calls for 4,096 lenses. Increasing the chunk
reduces repeated setup, but the optimum depends on device count and memory.
In particular, chunk 2048 creates only two tasks for this workload, leaving
some devices idle in a four-device session.

Separate outer dispatch granularity from the inner GPU working tile size.
Cache numeric pixel centers and immutable conversion constants. Preserve enough
tasks for load balance and keep working allocations within each device budget.
Tune on multiple catalog sizes rather than changing defaults from a single-GPU
measurement.

## Result gathering

The original synthetic output transfers about 3.75 MiB of pair statistics in
32 tasks, with eight synchronous column transfers per task. The old instrumented
benchmark recorded summed worker gather time rising from roughly 5 ms on one
GPU to 417 ms on four GPUs. These are overlapping worker wall times, not pure
PCIe bandwidth measurements; they cannot be added to end-to-end latency.

A prototype stacks uint64 views of each output column and performs one transfer
per task, retaining the exact int64/FP64 bit patterns. In the local sweep,
packing yielded small and inconsistent improvements and was slower at chunk
2048. It adds packing kernels and copies, so packing alone is a lower priority
than host task generation. A later experiment can test pinned host output and
asynchronous transfers with explicit completion and buffer lifetime handling.

## Benchmark and kernel follow-up

- Measure end-to-end latency without per-task timing synchronizations or memory
  sampling; collect profiling and memory traces in separate runs. The original
  benchmark adds a synchronization after every worker pairing call.
- Profile region reductions and scatter launches after host improvements. Fusing
  reduction/scatter into contiguous result buffers could reduce launch count.
- The pair kernel recomputes geometry for each radial bin. A bin-assignment
  kernel could reduce repeated work, but it must preserve `(lower, upper]`
  membership and FP64 scientific gates. This remains an unmeasured opportunity.
- Host precompute before pairing was about 80 ms and post-pair profiles about
  10 ms on the previous four-A100 node. Their optimization priority is below
  the demonstrated task-construction bottleneck for this workload.

## Reproduction and pending multi-device experiments

Local artifacts: `local.json`, `worker-1-0.txt`, `vectorized_local.json`, and
`vectorized_tests.log`. Scripts: `diagnose_pairing.py`, `vectorized_tasks.py`,
and `compare_tasks.py` in the scratch directory above. Experiments monkeypatch
functions within their own Python process; they do not edit production files.

Four-device diagnostic Slurm job **60519318** ended with exit code 1 after
4 minutes 37 seconds. All chunk/gather sweep comparisons passed, but the
subsequent concurrent cProfile stage raised
`ValueError: Another profiling tool is already active`. The vectorized and
larger-workload comparisons were not reached. This diagnostic failure is not
a failed scientific correctness assertion.

Completed uninstrumented four-device sweep, 4,096 lenses / 500,000 sources,
median warm precompute seconds:

| lens_chunk | Original gather | Packed gather |
| --- | ---: | ---: |
| 32 | 0.945 | 0.961 |
| 128 | 0.687 | 0.691 |
| 512 | 0.638 | 0.640 |
| 2048 | 0.624 | 0.623 |

Chunk 2048 creates only two tasks and therefore uses at most two GPUs for
pairing. Chunk 512 improves on chunk 128 by about 7.1%; packed gathering has no
material advantage in this sweep. `four_gpu.json` retains the completed sweep
rows but lacks a final completion flag because profiling subsequently failed.

Follow-up job **60521355** completed successfully using `resume.slurm` to run only the unfinished uninstrumented
original/vectorized comparisons on the small and larger workloads. It avoids
the conflicting concurrent profiling stage and writes
`four_gpu_vectorized.json` and `larger_vectorized.json`. Larger-workload
chunk/gather tuning remains unmeasured.

## Applied implementation

`GPUSession.pair` now fills a bounded NumPy task buffer instead of constructing
individual Python tuples. Its flush boundaries, source order and CUDA kernels
are unchanged. Pixel-center trigonometry operates on numeric radians and is
batched per call; immutable HEALPix geometry and conversion constants are cached
by nside. Multi-device cost estimation uses the same numeric geometry helpers.
Default chunk sizes remain unchanged.

The benchmark now disables per-worker timing synchronizations and memory
sampling by default. Explicit `--profile-workers` and `--sample-memory` flags
retain diagnostic measurements in a separate run. This changes timing
instrumentation only, not scientific outputs.

Local production checks: **35 passed, 3 dual-device cases skipped**. Final
four-device verification: **38 passed with no skips**. The new
regression forces source streaming and multiple launches within one lens to
check the buffer rewrite across flush boundaries. Benchmark smoke checks cover
both default timing and optional diagnostic modes.

The prototype comparison in job **60521355** completed its small workload:

| Multi session | Original, chunk 128 | Prototype, chunk 128 | Prototype, chunk 512 |
| --- | ---: | ---: | ---: |
| 1 GPU | 0.685 s | 0.278 s | 0.236 s |
| 4 GPUs | 0.675 s | 0.256 s | 0.206 s |

These results validate the host optimization direction, not the final production
revision with additional geometry caching. The larger prototype comparison also completed successfully: default-chunk
four-device precompute fell from 7.045 s to 0.879 s; the one-device prototype
took 1.805 s. All statistics passed the comparison gates. The old single-device method was frozen in `original_pair.py`
before editing production sources, preserving the comparison baseline across
subprocesses.

Production verification job **60521476** completed successfully in 4 minutes
45 seconds (exit code 0). All 38 GPU regression tests passed with no skips.
It completed an uninstrumented 1/2/4-device benchmark and comparisons
of the actual optimized method against the frozen original method on both
catalog sizes. Outputs: `production_benchmark.json`,
`production_comparison.json`, `production_larger.json`, and
`slurm-60521476.log` in the optimization scratch directory.

### Production timing on the verification node

The completed uninstrumented small-workload benchmark reports warm medians of
0.164 s for `GPUSession`, 0.221 s for a one-device multi session, 0.187 s for two
devices and 0.191 s for four devices. Pipeline medians, including both profiles
and jackknife, were 0.175 / 0.230 / 0.197 / 0.200 s respectively. For this small
workload, the existing single-device interface is still the fastest choice.

The same-node frozen-original versus production-method comparison completed:

| Devices | Chunk | Frozen original pair method | Optimized production method |
| --- | ---: | ---: | ---: |
| 1 | 128 | 0.644 s | 0.221 s |
| 1 | 512 | 0.588 s | 0.186 s |
| 4 | 128 | 0.628 s | 0.186 s |
| 4 | 512 | 0.585 s | 0.155 s |

Both sides use the current multi-session scheduler; the original pair method
is loaded from its frozen source. Every correctness gate passed. The default
four-device improvement is 3.38x. The old and new methods use identical chunk
settings in each row. The verification benchmark's recorded implementation
SHA-256 hashes match the applied source files.

### Completed larger production comparison

All production output JSON files have `completed: true`; recorded source hashes
match the applied implementation and every comparison gate passed.
At 16,384 lenses and 2,000,000 sources, warm median precompute seconds:

| Devices | Chunk | Frozen original pair method | Optimized production method |
| --- | ---: | ---: | ---: |
| 1 | 128 | 7.210 s | 1.726 s |
| 1 | 512 | 6.876 s | 1.396 s |
| 4 | 128 | 7.042 s | 0.805 s |
| 4 | 512 | 6.831 s | 0.632 s |

At unchanged default chunk 128, the four-device speedup is **8.74x** versus the
original pair method; four devices are **2.14x** faster than the optimized
one-device multi session. Chunk 512 is an explicit tuning option; defaults
remain unchanged. These are fixed-seed synthetic measurements with three warm
repeats, not a real-survey end-to-end performance claim. Large-scale source
streaming performance was not benchmarked; streaming correctness is covered by
the regression suite.
