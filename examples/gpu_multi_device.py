"""Synthetic single-host 1/2/4 GPU correctness and timing benchmark.

Run with PYTHONPATH=. python examples/gpu_multi_device.py --devices 0 1 2 3
Outputs JSON; catalogs, generated with seed 42, are not saved. No survey inputs.
"""
import argparse
from contextlib import contextmanager, nullcontext
import json
import hashlib
from pathlib import Path
import platform
from threading import Event, Thread, Lock
from time import perf_counter

import numpy as np
from astropy import units as u
from astropy.table import Table
import astropy

from dsigma import __version__, stacking
from dsigma.gpu import GPUSession, MultiGPUSession, cupy
from dsigma.jackknife import jackknife_resampling
from dsigma.precompute import precompute


def catalogs(nlens, nsource):
    rng = np.random.default_rng(42)
    lenses = Table(dict(ra=rng.uniform(0, 8, nlens), dec=rng.uniform(0, 8, nlens),
                        z=rng.uniform(.1, .4, nlens), w_sys=np.ones(nlens),
                        field_jk=np.arange(nlens) % 32))
    sources = Table(dict(ra=rng.uniform(-1, 9, nsource), dec=rng.uniform(-1, 9, nsource),
                         z=rng.uniform(.5, 1, nsource), w=rng.uniform(.5, 1.5, nsource),
                         e_1=rng.normal(0, .2, nsource), e_2=rng.normal(0, .2, nsource)))
    return lenses, sources


def compare(left, right):
    for key in left.colnames:
        if key.startswith('sum '):
            a, b = np.asarray(left[key]), np.asarray(right[key])
            if key == 'sum 1':
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)
                np.testing.assert_array_equal(np.isposinf(a), np.isposinf(b))
                np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))


@contextmanager
def memory_samples(devices):
    cp = cupy()
    stop = Event()
    peaks = {str(d): dict(device_used_bytes=0, pool_used_bytes=0,
                          pool_reserved_bytes=0) for d in devices}
    errors = []

    def sample():
        try:
            while not stop.is_set():
                for device in devices:
                    with cp.cuda.Device(device):
                        free, total = cp.cuda.runtime.memGetInfo()
                        pool = cp.get_default_memory_pool()
                        for key, value in dict(device_used_bytes=total-free,
                                               pool_used_bytes=pool.used_bytes(),
                                               pool_reserved_bytes=pool.total_bytes()).items():
                            peaks[str(device)][key] = max(peaks[str(device)][key], value)
                stop.wait(.02)
        except Exception as exc:
            errors.append(str(exc))

    thread = Thread(target=sample)
    thread.start()
    try:
        yield peaks
    finally:
        stop.set()
        thread.join()
        if errors:
            peaks['sampling_errors'] = errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--lenses', type=int, default=4096)
    parser.add_argument('--sources', type=int, default=500000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile-workers', action='store_true',
                        help='Instrument worker stages with extra synchronizations.')
    parser.add_argument('--sample-memory', action='store_true',
                        help='Sample memory in a background thread; use a separate timing run.')
    args = parser.parse_args()
    if args.lenses < 64 or args.sources < 1 or args.repeats < 1:
        parser.error('Use at least 64 lenses, one source and one repetition.')
    cp = cupy()
    # Validate selection before allocating the benchmark catalogs.
    with MultiGPUSession(args.devices):
        pass
    root = Path(__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in ('dsigma/gpu.py', 'dsigma/gpu_pairs.cu',
                           'dsigma/precompute.py', 'examples/gpu_multi_device.py')}
    report = dict(completed=False, code_sha256=hashes, versions=dict(dsigma=__version__, python=platform.python_version(),
                                numpy=np.__version__, astropy=astropy.__version__,
                                cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion()),
                  config=dict(devices=args.devices, lenses=args.lenses, sources=args.sources,
                              repeats=args.repeats, seed=42, nside=64, lens_chunk=128,
                              profile_workers=args.profile_workers, sample_memory=args.sample_memory,
                              bins_mpc=np.geomspace(.1, 5, 16).tolist()),
                  devices={str(d): cp.cuda.runtime.getDeviceProperties(d)['name'].decode()
                           for d in args.devices}, runs=[])
    start = perf_counter()
    lenses, sources = catalogs(args.lenses, args.sources)
    report['catalog_generation_seconds'] = perf_counter()-start
    bins = np.geomspace(.1, 5, 16)*u.Mpc
    # A fresh CPU oracle on a small subset, not a historical acceptance rerun.
    cpu = precompute(lenses[:64].copy(), sources[:2000], bins, nside=64)
    oracle = precompute(lenses[:64].copy(), sources[:2000], bins, nside=64, backend='gpu')
    compare(cpu, oracle)
    report['cpu_subset_parity'] = 'passed'
    baseline = None
    profile_baseline = None
    configurations = [('single', [args.devices[0]])]
    configurations += [('multi', args.devices[:n]) for n in (1, 2, 4) if n <= len(args.devices)]
    for kind, devices in configurations:
        for d in args.devices:
            with cp.cuda.Device(d):
                cp.get_default_memory_pool().free_all_blocks()
        start = perf_counter()
        with cp.cuda.Device(devices[0]):
            session = GPUSession() if kind == 'single' else MultiGPUSession(devices)
        initialization = perf_counter()-start
        row = dict(kind=kind, devices=devices, initialization_seconds=initialization, trials=[])
        try:
            for repeat in range(args.repeats+1):
                stages = {}
                original_pair = session.pair
                worker_times = []
                gather_times = []
                metrics_lock = Lock()
                originals = []
                original_gather = None
                if kind == 'multi' and args.profile_workers:
                    # Synchronize worker pairing before timing host transfers.
                    # Busy times are summed across workers, not additive wall stages.
                    def instrument(worker, stream):
                        real = worker.pair

                        def measured(*a, **kw):
                            begin = perf_counter()
                            result = real(*a, **kw)
                            stream.synchronize()
                            with metrics_lock:
                                worker_times.append(perf_counter()-begin)
                            return result

                        originals.append((worker, real))
                        worker.pair = measured

                    for worker, stream in session._workers:
                        instrument(worker, stream)
                    original_gather = session._gather

                    def measured_gather(*a):
                        begin = perf_counter()
                        original_gather(*a)
                        with metrics_lock:
                            gather_times.append(perf_counter()-begin)

                    session._gather = measured_gather

                def timed_pair(*a, **kw):
                    stages['host_preparation_seconds'] = perf_counter()-start
                    pair_start = perf_counter()
                    output = original_pair(*a, **kw)
                    for d in devices:
                        with cp.cuda.Device(d):
                            cp.cuda.runtime.deviceSynchronize()
                    stages['pair_and_gather_seconds'] = perf_counter()-pair_start
                    return output

                session.pair = timed_pair
                try:
                    memory = memory_samples(devices) if args.sample_memory else nullcontext({})
                    with memory as peaks, cp.cuda.Device(devices[0]):
                        start = perf_counter()
                        result = precompute(lenses.copy(), sources, bins, backend='gpu',
                                            session=session, cross=True, nside=64)
                        precompute_time = perf_counter()-start
                        profile_start = perf_counter()
                        profiles = []
                        for component in ('t', 'x'):
                            signal = stacking.excess_surface_density(
                                result, backend='gpu', component=component)
                            covariance, samples = jackknife_resampling(
                                stacking.excess_surface_density, result,
                                backend='gpu', component=component, return_samples=True)
                            profiles.extend([signal, covariance, samples])
                        cp.cuda.runtime.deviceSynchronize()
                        profile_time = perf_counter()-profile_start
                finally:
                    session.pair = original_pair
                    for worker, real in originals:
                        worker.pair = real
                    if original_gather is not None:
                        session._gather = original_gather
                if baseline is None:
                    baseline = result
                else:
                    compare(baseline, result)
                numeric_profiles = [u.Quantity(value).value for value in profiles]
                if profile_baseline is None:
                    profile_baseline = numeric_profiles
                else:
                    for expected, actual in zip(profile_baseline, numeric_profiles):
                        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10, equal_nan=True)
                        np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
                        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
                row['trials'].append(dict(phase='cold' if repeat == 0 else 'warm',
                                          **stages, finalization_seconds=precompute_time-sum(stages.values()),
                                          precompute_seconds=precompute_time,
                                          profiles_seconds=profile_time,
                                          pipeline_seconds=precompute_time+profile_time,
                                          worker_pair_busy_seconds=sum(worker_times) if originals else None,
                                          worker_gather_busy_seconds=sum(gather_times) if originals else None,
                                          memory_sampled_peak=peaks, parity='passed'))
                print(f'{kind} {len(devices)} GPU trial {repeat}: {precompute_time:.3f}s precompute',
                      flush=True)
        finally:
            with cp.cuda.Device(devices[0]):
                session.close()
        report['runs'].append(row)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
    report['completed'] = True
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
