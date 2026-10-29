Single NVIDIA GPU (optional)
===========================

Install ``pip install '.[gpu]'`` in a CUDA 12 environment. CuPy is loaded lazily;
CPU users do not need CUDA. CUDA runtime, NVRTC and cuBLAS must be discoverable.
On Perlmutter use the appropriate CUDA module and a single A100 allocation.
CPU execution remains the default.

``precompute``, ``tangential_shear``, ``excess_surface_density`` and
``jackknife_resampling`` accept ``backend='gpu'``. Astropy inputs, in-place
precomputation, output tables, units, cosmology preparation and correction order
follow the CPU implementation. GPU statistics use FP64, counts use int64, and
CUDA pair kernels disable FMA contraction and fast math. Pair bin membership
follows the CPU's ``(lower, upper]`` convention and its chord-distance cap.
The weighting exponent is rounded to float32 to match the Cython interface;
all pair computations thereafter use FP64.

Reuse a source catalog with a session::

    from dsigma.gpu import GPUSession
    from dsigma.precompute import precompute
    from dsigma.stacking import excess_surface_density
    from dsigma.jackknife import jackknife_resampling

    with GPUSession() as session:
        result = precompute(lenses, sources, bins, backend='gpu',
                            session=session, cross=True, return_device=True)
        signal = excess_surface_density(result, backend='gpu')
        cross = excess_surface_density(result, backend='gpu', component='x')
        covariance, samples = jackknife_resampling(
            excess_surface_density, result, backend='gpu', return_samples=True)

A session owns a snapshot of source preparation. Keep its source table,
redshift distribution and preparation options immutable; create a new session
when any change. Lens catalogs and radial bins may change. Source buffers,
HEALPix sorting and the CPU spatial index are reused. ``close()`` releases the
session's references; the CuPy allocator may retain freed buffers for reuse.
Preparation, source pixels and cosmology remain on the CPU. Candidate pixels
are processed in bounded source segments and lens chunks without a pair matrix.
The session reserves at most 70% of initially free memory for its live working
arrays, excluding caller-owned results and CuPy's retained allocator cache.
Sources larger than half that budget are staged from host memory in pixel
ranges. Lower ``memory_fraction``, ``source_chunk`` or ``lens_chunk`` for other
memory consumers. The host must still accommodate the prepared source catalog.

``return_device=True`` returns a ``DeviceCatalog`` containing statistics,
weights and any existing region labels. Device masks select lenses;
``with_regions(labels)`` attaches CPU-assigned labels and ``to_table()``
explicitly transfers these columns to Astropy. Original coordinate and ancillary
columns remain in the input table. Stacking and resampling can also return CuPy
arrays with ``return_device=True``. ``return_table=True`` always returns an
Astropy table on the host.

``cross=True`` accumulates both components in one pass. The cross sign matches
replacing ``(e_1, e_2)`` with ``(e_2, -e_1)`` in the CPU source catalog. Matrix
response uses the same geometry as the existing rotated-source workflow.

Jackknife processing
--------------------

GPU jackknife supports the two built-in profiles and their new
``tangential_shear_difference`` and ``excess_surface_density_difference``
functions. Differences accept ``table_l_2`` and ``table_r_2``; the corrections
are applied independently before taking the difference. Arbitrary Python
functions remain supported by the CPU backend.

The GPU groups weighted sufficient statistics by region. Pair counts remain
unweighted. All leave-one-out profiles in a batch are evaluated together,
including nonlinear corrections, boost and random subtraction. Non-finite
contributions are tracked with masks, preserving NaN/Inf semantics when their
region is removed. Region labels may be non-contiguous and random catalogs may
lack regions. The sorted unique labels of the first lens catalog define all
samples, including deletion in secondary catalogs. The default batch size is
64; larger region sets are processed in batches without per-sample host copies.
The covariance uses centered FP64 matrix multiplication and the factor
``(N-1)/N``. At least two regions are required.

Validation and benchmarking
---------------------------

``tests/test_gpu.py`` compares Cython and GPU statistics, all correction
combinations, every jackknife sample, covariance, cross rotation, source
redshift distributions, calibration, bin boundaries, chunk sizes, polar and
RA-wrap geometry, missing random regions, two catalog pairs and NaN/Inf values.
GPU tests skip when CuPy or a CUDA device is unavailable.

``examples/gpu_single_device.py`` is a standalone synthetic demonstration.
The frozen pre-1.2.3 ``examples/gpu_validation.py`` is a Perlmutter acceptance runner using existing
calibrated type1 inputs and read-only survey adapters. Its default output root
is ``/pscratch/sd/q/qinxunli/dsigma_gpu_validation/acceptance``. Run the full
UNIONS source catalog without ``--source-limit``; use that flag for deterministic
DES/DECADE regressions. The runner fixes seed 42, radial bins, mass bins,
corrections and shared jackknife labels, checks every sample before covariance,
and writes version/timing manifests and scientific artifacts to scratch.

To reproduce the original report, use its recorded checkout and run::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python examples/gpu_validation.py \
      --cpu-jobs 4 --output /pscratch/sd/q/qinxunli/dsigma_gpu_validation/unions_full

Use the production CPU process count only in an allocation with enough CPUs.
NVTX ranges identify precomputation, region aggregation, batched corrections
and covariance for ``nsys profile --trace=cuda,nvtx``. Compilation is timed
separately; use an empty ``CUPY_CACHE_DIR`` on scratch to measure cold compilation.
Timing results are measurements, with no assumed speedup.

Measured acceptance results are recorded in ``docs/gpu_validation.md`` with a
machine-readable manifest in ``docs/gpu_validation_summary.json``. In the initial
complete UNIONS run, GPU was faster than one CPU process but did not outperform
the 256-process production CPU pipeline end to end.

Upstream compatibility
----------------------

Both backends use the upstream default cosmology and units: bare radial bins
mean Mpc/h, density profiles carry little-h solar-mass/pc² units, and shear
output columns are named ``gt`` and ``gt_raw``. Device arrays contain numerical
values in the catalog's units; ``DeviceCatalog.to_table()`` restores column
units. Host density profiles, jackknife samples and covariance carry units.
Set redshift selection through source ``z_l_max``; the explicit
``lens_source_cut`` keyword remains a compatibility override for older GPU
examples. Omitting it follows the upstream source-column defaults.

CPU jackknife now compresses input catalogs by default. GPU always performs its
own region aggregation. To compare NaN/Inf or non-positive-weight edge cases
against direct CPU weighted sums, pass ``compress=False`` to CPU jackknife;
upstream compression can change those exceptional-value semantics.
