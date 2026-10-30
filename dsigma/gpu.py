"""Optional FP64 GPU backend. CuPy is imported only on GPU use.

A GPUSession owns a snapshot of one source catalog. Do not mutate that catalog
between calls; create a new session when source data or preparation options change.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from queue import Empty, Queue
from threading import Lock
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np


def cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError("GPU backend requires CuPy (install dsigma[gpu]).") from exc
    return cp


def check_backend(backend):
    if backend not in ('cpu', 'gpu'):
        raise ValueError("backend must be 'cpu' or 'gpu'")


def _nvtx(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            cp = cupy()
            cp.cuda.nvtx.RangePush(name)
            try:
                return function(*args, **kwargs)
            finally:
                cp.cuda.nvtx.RangePop()
        return wrapped
    return decorate


@lru_cache(maxsize=16)
def _segment_kernel(device, integer):
    cp = cupy()
    return cp.RawKernel(Path(__file__).with_name('gpu_pairs.cu').read_text(),
                        'segmented_int64' if integer else 'segmented_double',
                        options=('--std=c++11', '--fmad=false'))


@lru_cache(maxsize=8)
def _finite_kernel(device):
    cp = cupy()
    return cp.RawKernel(Path(__file__).with_name('gpu_pairs.cu').read_text(),
                        'grouped_finite', options=('--std=c++11', '--fmad=false'))


def _segmented_sum(values, starts):
    cp = cupy()
    integer = np.issubdtype(values.dtype, np.integer)
    values = cp.ascontiguousarray(values, dtype=cp.int64 if integer else cp.float64)
    starts = cp.asarray(starts, dtype=cp.int64)
    width = int(np.prod(values.shape[1:]))
    output = cp.empty((len(starts),) + values.shape[1:], dtype=values.dtype)
    size = len(starts)*width
    _segment_kernel(cp.cuda.Device().id, integer)(
        ((size+127)//128,), (128,),
        (values, starts, np.int64(len(values)), np.int64(len(starts)),
         np.int64(width), output))
    return output


def _to_device(value):
    cp = cupy()
    if hasattr(value, '__cuda_array_interface__'):
        return cp.asarray(value)
    array = np.asarray(value)
    # FITS columns may use big-endian dtypes, unsupported by CuPy directly.
    return cp.asarray(array.astype(array.dtype.newbyteorder('='), copy=False))


class DeviceCatalog:
    """Device-resident precompute columns with explicit host conversion."""
    def __init__(self, columns, meta=None, units=None):
        self.units = dict(units or {})
        self.columns = columns
        self.meta = deepcopy(meta or {})

    @property
    def colnames(self):
        return list(self.columns)

    def __len__(self):
        return len(next(iter(self.columns.values())))

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.columns[item]
        return DeviceCatalog({k: v[item] for k, v in self.columns.items()}, self.meta, self.units)

    def with_regions(self, labels):
        labels = _to_device(labels)
        if labels.shape != (len(self),):
            raise ValueError('Region labels must have one entry per lens.')
        return DeviceCatalog(dict(self.columns, field_jk=labels), self.meta, self.units)

    @classmethod
    def from_table(cls, table):
        if isinstance(table, cls):
            return table
        return cls({k: _to_device(table[k]) for k in table.colnames
                    if k.startswith('sum ') or k in ('w_sys', 'field_jk')}, table.meta,
                   {k: table[k].unit for k in table.colnames if getattr(table[k], 'unit', None)})

    def to_table(self):
        from astropy.table import Table
        cp = cupy()
        table = Table({k: cp.asnumpy(v) for k, v in self.columns.items()},
                      meta=deepcopy(self.meta))
        for key, unit in self.units.items():
            if key in table.colnames:
                table[key].unit = unit
        return table


@lru_cache(maxsize=16)
def _pair_geometry(nside):
    from astropy import constants as c
    from astropy import units as u
    from astropy_healpix import HEALPix
    hp = HEALPix(nside, order='ring')
    pixrad = np.deg2rad(1.05 * hp.pixel_resolution.to_value(u.deg))
    factor = (1e-6 * c.c**2 / (4 * np.pi * c.G)).to_value(u.Msun / u.pc)
    return hp, pixrad, factor


def _pixel_xyz(hp, pixels):
    from astropy import units as u
    lon, lat = hp.healpix_to_lonlat(pixels)
    lon, lat = lon.to_value(u.rad), lat.to_value(u.rad)
    return np.column_stack((np.cos(lon)*np.cos(lat),
                            np.sin(lon)*np.cos(lat), np.sin(lat)))


class GPUSession:
    """Reuse source buffers and HEALPix spatial index on the current device.

    Allocations made by this session are limited to memory_fraction of the free
    device memory at construction. source_chunk controls deterministic local
    accumulation; lens_chunk bounds intermediate output storage.
    """
    def __init__(self, memory_fraction=0.7, source_chunk=256, lens_chunk=128):
        cp = cupy()
        if not 0 < memory_fraction <= 0.7:
            raise ValueError('memory_fraction must be in (0, 0.7].')
        if source_chunk < 1 or lens_chunk < 1:
            raise ValueError('Chunk sizes must be positive.')
        self.device = cp.cuda.Device().id
        self.budget = int(cp.cuda.runtime.memGetInfo()[0] * memory_fraction)
        self.source_chunk = int(source_chunk)
        self.lens_chunk = int(lens_chunk)
        self.source = None
        self.source_token = None
        self.tree = None
        self.kernel = cp.RawKernel(Path(__file__).with_name('gpu_pairs.cu').read_text(),
                                   'pairs', options=('--std=c++11', '--fmad=false'))
        self._source_table = None
        self._prepared_source = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.source = None
        self.source_token = None
        self.tree = None
        self.zbin = None
        self._source_table = None
        self._prepared_source = None
        self._table_n = None
        self._cut = None

    def pair(self, ul, nl, us, ns, edges, lens, source, keys, nside,
             comoving, weighting, source_token):
        from scipy.spatial import cKDTree
        cp = cupy()
        if cp.cuda.Device().id != self.device:
            raise ValueError('Use the session on the device where it was created.')
        skeys = ['z', 'd_com', 'sin ra', 'cos ra', 'sin dec', 'cos dec',
                 'w', 'e_1', 'e_2', 'z_l_max', 'm', 'e_rms', 'm_sel',
                 'R_11', 'R_12', 'R_21', 'R_22']
        lkeys = skeys[:6]
        hp, pixrad, factor = _pair_geometry(nside)
        lens_xyz = _pixel_xyz(hp, ul)
        if self.source_token != source_token:
            if self.source_token is not None:
                raise ValueError('A session is bound to one source preparation; create a new session.')
            needed = len(source['z']) * (len(skeys) * 8 + 8)
            if needed <= self.budget // 2:
                self.source = cp.empty((len(skeys), len(source['z'])), dtype=cp.float64)
                for i, key in enumerate(skeys):
                    if key in source:
                        self.source[i] = cp.asarray(source[key])
                    else:
                        self.source[i].fill(0)
                self.zbin = cp.asarray(source.get('z_bin', np.zeros(len(source['z']), dtype=np.int64)))
            else:
                # Large catalogs remain in host memory and selected pixel ranges
                # are staged per tile. The spatial index is still reused.
                self.source = None
                self.zbin = cp.empty(0, dtype=cp.int64)
            if self.tree is None:
                self.tree = cKDTree(_pixel_xyz(hp, us))
            self.source_token = source_token
        ld = cp.asarray(np.stack([lens[k] for k in lkeys]))
        nb = edges.shape[1] - 1
        eff = lens.get('sigma_crit_eff', np.empty(0))
        nz = len(eff) // len(lens['z'])
        eff = cp.asarray(eff)
        ed = cp.asarray(edges)
        # Fixed stat layout shared with the CUDA kernel.
        statkeys = ['sum w_ls', 'sum w_ls e_t', 'sum w_ls z_s',
                    'sum w_ls e_t sigma_crit', 'sum w_ls sigma_crit',
                    'sum w_ls m', 'sum w_ls (1 - e_rms^2)',
                    'sum w_ls m_sel', 'sum w_ls R_T', 'sum w_ls e_x',
                    'sum w_ls e_x sigma_crit']
        out = {k: cp.zeros((len(lens['z']), nb), dtype=cp.int64 if k == 'sum 1'
                           else cp.float64) for k in keys}
        resident = ((0 if self.source is None else self.source.nbytes) + self.zbin.nbytes + ld.nbytes + eff.nbytes
                    + ed.nbytes + sum(a.nbytes for a in out.values()))
        if resident > self.budget * .8:
            raise MemoryError('Catalog results exceed GPU session memory budget.')
        starts_l = np.r_[0, nl[:-1]]
        starts_s = np.r_[0, ns[:-1]]
        # Combine independent pixels into launches to amortize launch overhead.
        # Tasks remain sorted by lens, then pixel/range; segment loops fix the reduction order.
        bytes_per_task = 2 * nb * (11*8+8)
        if self.source is None:
            bytes_per_task += 2 * self.source_chunk * 144
        max_tasks = max(1, min(131072, (self.budget-resident)//bytes_per_task))
        # Keep the original flush boundaries and source order without creating
        # a Python tuple for every lens/source segment.
        pending = np.empty((max_tasks, 3), dtype=np.int64)
        n_pending = 0
        def flush():
            nonlocal n_pending
            if not n_pending:
                return
            # Streaming rewrites source offsets; own the host launch snapshot.
            host_tasks = pending[:n_pending].copy()
            ntask = len(host_tasks)
            lens_ids, starts = np.unique(host_tasks[:, 0], return_index=True)
            sd, zd, ns_device = self.source, self.zbin, len(source['z'])
            if sd is None:
                indices = np.concatenate([np.arange(a, b) for _, a, b in host_tasks])
                sd = cp.empty((len(skeys), len(indices)), dtype=cp.float64)
                for i, key in enumerate(skeys):
                    if key in source:
                        sd[i] = cp.asarray(source[key][indices])
                    else:
                        sd[i].fill(0)
                zd = cp.asarray(source['z_bin'][indices] if 'z_bin' in source
                                else np.zeros(len(indices), dtype=np.int64))
                ends = np.cumsum(host_tasks[:, 2]-host_tasks[:, 1])
                host_tasks[:, 1] = np.r_[0, ends[:-1]]
                host_tasks[:, 2] = ends
                ns_device = len(indices)
            tasks = cp.asarray(host_tasks)
            partial = cp.empty((ntask, 11, nb), dtype=cp.float64)
            counts = cp.empty((ntask, nb), dtype=cp.int64)
            size = ntask * nb
            self.kernel(((size+127)//128,), (128,),
                        (ld, sd, ed, eff, zd, tasks,
                         np.int64(len(lens['z'])), np.int64(ns_device),
                         np.int64(ntask), np.int32(nb), np.int32(nz),
                         np.int32(comoving), np.float64(np.float32(weighting)),
                         np.float64(factor), partial, counts))
            starts_d, ids_d = cp.asarray(starts), cp.asarray(lens_ids)
            out['sum 1'][ids_d] += _segmented_sum(counts, starts_d)
            summed = _segmented_sum(partial, starts_d)
            for j, key in enumerate(statkeys):
                if key in out:
                    out[key][ids_d] += summed[:, j, :]
            n_pending = 0
        lenses_pending = 0
        for p, (lo, hi) in enumerate(zip(starts_l, nl)):
            radius = np.sqrt(np.max(edges[lo:hi, -1])) + 2 * pixrad
            candidates = sorted(self.tree.query_ball_point(lens_xyz[p], radius))
            ranges = [(a, min(a + self.source_chunk, ns[s]))
                      for s in candidates for a in range(starts_s[s], ns[s], self.source_chunk)]
            if not ranges:
                continue
            range_array = np.asarray(ranges, dtype=np.int64)
            for il in range(lo, hi):
                position = 0
                while position < len(ranges):
                    take = min(len(ranges)-position, max_tasks-n_pending)
                    pending[n_pending:n_pending+take, 0] = il
                    pending[n_pending:n_pending+take, 1:] = range_array[position:position+take]
                    n_pending += take
                    position += take
                    if n_pending >= max_tasks:
                        flush()
                        lenses_pending = 0
                lenses_pending += 1
                if lenses_pending >= self.lens_chunk:
                    flush()
                    lenses_pending = 0
        flush()
        return out


class MultiGPUSession:
    """Parallel lens pairing on explicitly selected GPUs on one host.

    Source preparation and the spatial index are shared in host memory. Each
    device owns a persistent worker, stream and single-device source cache.
    Pair results are gathered on the host; ``return_device=True`` in precompute
    uploads the complete result to ``devices[0]``. Use that device's context for
    subsequent device operations. Do not overlap precompute calls on one session.
    """
    def __init__(self, devices, memory_fraction=0.7, source_chunk=256, lens_chunk=128):
        devices = tuple(devices)
        if not devices or any(isinstance(d, (bool, np.bool_)) or
                              not isinstance(d, (int, np.integer)) for d in devices):
            raise ValueError('devices must be a non-empty sequence of integer device IDs.')
        if len(set(devices)) != len(devices):
            raise ValueError('devices must not contain duplicate IDs.')
        if not 0 < memory_fraction <= 0.7:
            raise ValueError('memory_fraction must be in (0, 0.7].')
        if source_chunk < 1 or lens_chunk < 1:
            raise ValueError('Chunk sizes must be positive.')
        cp = cupy()
        count = cp.cuda.runtime.getDeviceCount()
        if any(d < 0 or d >= count for d in devices):
            raise ValueError('Device IDs must refer to visible CUDA devices.')
        self.devices = tuple(int(d) for d in devices)
        self.device = self.devices[0]
        self.lens_chunk = int(lens_chunk)
        self.source_token = None
        self._prepared_source = None
        self._source_table = self._table_n = self._cut = None
        self.tree = None
        self._lock = Lock()
        self._closed = False
        self._executors = [ThreadPoolExecutor(max_workers=1) for _ in devices]
        self._workers = []

        def initialize(device):
            with cp.cuda.Device(device):
                stream = cp.cuda.Stream(non_blocking=True)
                with stream:
                    worker = GPUSession(memory_fraction, source_chunk, lens_chunk)
                return worker, stream

        futures = [executor.submit(initialize, device)
                   for executor, device in zip(self._executors, self.devices)]
        try:
            # Collect every successful worker even when another initializer fails.
            error = None
            for future in futures:
                try:
                    self._workers.append(future.result())
                except Exception as exc:
                    self._workers.append(None)
                    if error is None:
                        error = exc
            if error is not None:
                raise error
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args):
        self.close()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('MultiGPUSession is closed.')

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            cp = cupy()

            def release(entry):
                if entry is not None:
                    worker, stream = entry
                    with cp.cuda.Device(worker.device):
                        stream.synchronize()
                        worker.close()

            futures = [executor.submit(release, entry)
                       for executor, entry in zip(self._executors, self._workers)]
            try:
                for future in futures:
                    future.result()
            finally:
                for executor in self._executors:
                    executor.shutdown(wait=True)
                self._workers.clear()
                self.source_token = self.tree = self._prepared_source = None
                self._source_table = self._table_n = self._cut = None

    def pair(self, ul, nl, us, ns, edges, lens, source, keys, nside,
             comoving, weighting, source_token):
        from scipy.spatial import cKDTree
        with self._lock:
            self._ensure_open()
            if self.source_token is not None and self.source_token != source_token:
                raise ValueError('A session is bound to one source preparation; create a new session.')
            cp = cupy()
            hp, pixrad, _ = _pair_geometry(nside)
            lens_xyz = _pixel_xyz(hp, ul)

            if self.tree is None:
                self.tree = cKDTree(_pixel_xyz(hp, us))
            self.source_token = source_token
            # Bound task results while allowing tasks to span small pixels, so
            # sparse pixels do not force many tiny GPU launches.
            source_counts = np.diff(np.r_[0, ns])
            costs = np.empty(len(lens['z']), dtype=np.float64)
            for p, (lo, hi) in enumerate(zip(np.r_[0, nl[:-1]], nl)):
                radius = np.sqrt(np.max(edges[lo:hi, -1])) + 2*pixrad
                candidates = self.tree.query_ball_point(lens_xyz[p], radius)
                costs[lo:hi] = max(1, int(source_counts[candidates].sum()))
            tasks = [(float(costs[lo:lo+self.lens_chunk].sum()), lo,
                      min(lo+self.lens_chunk, len(costs)))
                     for lo in range(0, len(costs), self.lens_chunk)]
            pending = Queue()
            for task in sorted(tasks, reverse=True):
                pending.put(task)
            output = {key: np.empty((len(lens['z']), edges.shape[1]-1),
                                    dtype=np.int64 if key == 'sum 1' else np.float64)
                      for key in keys}

            def run(entry):
                worker, stream = entry
                worker.tree = self.tree
                with cp.cuda.Device(worker.device), stream:
                    try:
                        while True:
                            try:
                                _, lo, hi = pending.get_nowait()
                            except Empty:
                                break
                            local = {key: value.reshape(len(lens['z']), -1)[lo:hi].ravel()
                                     if key == 'sigma_crit_eff' else value[lo:hi]
                                     for key, value in lens.items()}
                            first = np.searchsorted(nl, lo, side='right')
                            last = np.searchsorted(nl, hi-1, side='right')+1
                            local_nl = np.minimum(nl[first:last], hi)-lo
                            result = worker.pair(
                                ul[first:last], local_nl, us, ns, edges[lo:hi],
                                local, source, keys, nside, comoving, weighting, source_token)
                            self._gather(result, output, lo, hi, stream)
                            del result
                    finally:
                        stream.synchronize()

            futures = [executor.submit(run, entry)
                       for executor, entry in zip(self._executors, self._workers)]
            error = None
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    if error is None:
                        error = exc
            if error is not None:
                raise error
            return output

    @staticmethod
    def _gather(result, output, lo, hi, stream):
        cp = cupy()
        for key, value in result.items():
            output[key][lo:hi] = cp.asnumpy(value, stream=stream)


@_nvtx('dsigma.region_aggregation')
def aggregate(catalog, regions=None):
    """Weighted sufficient statistics, optionally all leave-one-region-out rows.

    Non-finite values are tracked separately so removing their region restores
    finite sums and zero-weight NaNs retain NumPy's multiplication semantics.
    Region membership is reduced in parallel; no stacking calls per region.
    """
    cp = cupy()
    catalog = DeviceCatalog.from_table(catalog)
    w = catalog['w_sys']
    if regions is None:
        return {**{k: (v if k == 'sum 1' else v*w[:, None]).sum(axis=0)
                   for k, v in catalog.columns.items() if k.startswith('sum ')},
                'w_sys': w.sum()}
    labels = catalog['field_jk']
    # Sort once on device, then segmented reductions for every statistic.
    order = cp.argsort(labels)
    sorted_labels = labels[order]
    if len(order) == 0:
        nb = catalog['sum 1'].shape[1]
        return {**{k: cp.zeros((len(regions), nb), dtype=v.dtype)
                   for k, v in catalog.columns.items() if k.startswith('sum ')},
                'w_sys': cp.zeros(len(regions))}
    starts = cp.r_[cp.array([0]), cp.nonzero(sorted_labels[1:] != sorted_labels[:-1])[0]+1]
    unique = sorted_labels[starts]
    positions = cp.searchsorted(unique, regions)
    safe = cp.minimum(positions, len(unique)-1)
    match = (positions < len(unique)) & (unique[safe] == regions)
    # One lane per region, statistic and radial bin; finite sums and exceptional
    # masks are accumulated together. This also handles non-finite w_sys.
    keys = [k for k in catalog.colnames if k.startswith('sum ') and k != 'sum 1']
    nb = catalog['sum 1'].shape[1]
    values = cp.stack([catalog[k]*w[:, None] for k in keys] +
                      [cp.broadcast_to(w[:, None], (len(w), nb))], axis=1)
    values = cp.ascontiguousarray(values[order], dtype=cp.float64)
    width = (len(keys)+1)*nb
    shape = (len(starts), len(keys)+1, nb)
    finite = cp.empty(shape, dtype=cp.float64)
    flags = cp.empty((4,) + shape, dtype=cp.int64)
    size = len(starts)*width
    _finite_kernel(cp.cuda.Device().id)(
        ((size+127)//128,), (128,),
        (values, starts, np.int64(len(w)), np.int64(len(starts)),
         np.int64(width), finite, flags))
    def exclude_grouped(grouped):
        removed = cp.where(match.reshape((-1,) + (1,)*(grouped.ndim-1)),
                           grouped[safe], 0)
        return grouped.sum(axis=0)[None] - removed
    retained = exclude_grouped(finite)
    nan, pos, neg, nonzero = [exclude_grouped(flags[i]) for i in range(4)]
    retained = cp.where(nonzero == 0, 0, retained)
    retained = cp.where((nan > 0) | ((pos > 0) & (neg > 0)), cp.nan,
                        cp.where(pos > 0, cp.inf, cp.where(neg > 0, -cp.inf, retained)))
    result = {key: retained[:, i, :] for i, key in enumerate(keys)}
    result['w_sys'] = retained[:, -1, 0]
    result['sum 1'] = exclude_grouped(_segmented_sum(catalog['sum 1'][order], starts))

    return result


@_nvtx('dsigma.batched_stacking')
def stack_sums(lens, random=None, shear=False, component='t', **options):
    """Broadcast stacking over bins and an optional leading sample dimension."""
    if component not in ('t', 'x'):
        raise ValueError("component must be 't' or 'x'.")
    signal = 'gt' if shear else 'ds'
    numerator = 'sum w_ls e_' + component + ('' if shear else ' sigma_crit')
    w = lens['sum w_ls']
    raw = lens[numerator] / w
    result = {'n_pairs': lens['sum 1'], signal + '_raw': raw,
              signal: raw.copy(), 'z_l': lens['sum w_ls z_l']/w,
              'z_s': lens['sum w_ls z_s']/w}
    if options.get('boost_correction'):
        if random is None:
            raise ValueError('Boost correction requires a random catalog.')
        lw, rw = lens['w_sys'], random['w_sys']
        if getattr(lw, 'ndim', 0):
            lw, rw = lw[:, None], rw[:, None]
        result['b'] = w/lw/random['sum w_ls']*rw
        result[signal] *= result['b']
    for flag, column, output, offset, scale in [
            ('scalar_shear_response_correction', 'sum w_ls m', '1+m', 1, 1),
            ('matrix_shear_response_correction', 'sum w_ls R_T', 'R_t', 0, 1),
            ('shear_responsivity_correction', 'sum w_ls (1 - e_rms^2)', '2R', 0, 2),
            ('selection_bias_correction', 'sum w_ls m_sel', '1+m_sel', 1, 1)]:
        if options.get(flag):
            result[output] = scale*(offset + lens[column]/w)
            result[signal] /= result[output]
    if options.get('photo_z_dilution_correction') and not shear:
        result['f_bias'] = lens[numerator + ' f_bias']/lens[numerator]
        result[signal] *= result['f_bias']
    if options.get('random_subtraction'):
        if random is None:
            raise ValueError('Random subtraction requires a random catalog.')
        ropts = dict(options, boost_correction=False, random_subtraction=False)
        result[signal + '_r'] = stack_sums(random, shear=shear, component=component, **ropts)[signal]
        result[signal] -= result[signal + '_r']
    return result


def stack(table_l, table_r=None, shear=False, return_table=False,
          return_device=False, component='t', **options):
    from astropy.table import Table
    cp = cupy()
    result = stack_sums(aggregate(table_l), None if table_r is None else aggregate(table_r),
                        shear=shear, component=component, **options)
    if return_table:
        if return_device:
            raise ValueError('return_table and return_device cannot both be True.')
        bins = table_l.meta['bins']
        table = Table({'rp_min': bins[:-1], 'rp_max': bins[1:]})
        table['n_pairs'] = cp.asnumpy(result.pop('n_pairs'))
        for k, v in result.items():
            table[k] = cp.asnumpy(v)
        if not shear:
            unit = _density_unit(table_l)
            for key in ('ds', 'ds_raw', 'ds_r'):
                if key in table.colnames and unit is not None:
                    table[key].unit = unit
        return table
    value = result['gt' if shear else 'ds']
    if return_device:
        return value
    value = cp.asnumpy(value)
    unit = _density_unit(table_l) if not shear else None
    return value if unit is None else value * unit


def jackknife(f, table_l, table_r=None, table_l_2=None, table_r_2=None,
              return_device=False, return_samples=False, batch_size=64, **options):
    from . import stacking
    cp = cupy()
    difference = f in (stacking.tangential_shear_difference, stacking.excess_surface_density_difference)
    shear = f in (stacking.tangential_shear, stacking.tangential_shear_difference)
    if f not in (stacking.tangential_shear, stacking.excess_surface_density,
                 stacking.tangential_shear_difference, stacking.excess_surface_density_difference):
        raise ValueError('GPU jackknife supports built-in shear and density statistics; use backend="cpu" for custom functions.')
    if difference and table_l_2 is None:
        raise ValueError('Profile differences require table_l_2.')
    if not difference and (table_l_2 is not None or table_r_2 is not None):
        raise ValueError('Use a built-in profile difference for two catalog pairs.')
    if options.pop('return_table', False):
        raise ValueError('Jackknife statistics must return arrays.')
    component = options.pop('component', 't')
    import inspect
    base = stacking.tangential_shear if shear else stacking.excess_surface_density
    inspect.signature(base).bind(table_l, **options)
    if batch_size < 2:
        raise ValueError('batch_size must contain at least two regions.')
    lens = DeviceCatalog.from_table(table_l)
    random = None if table_r is None else DeviceCatalog.from_table(table_r)
    lens2 = None if table_l_2 is None else DeviceCatalog.from_table(table_l_2)
    random2 = None if table_r_2 is None else DeviceCatalog.from_table(table_r_2)
    regions = cp.unique(lens['field_jk'])
    if len(regions) < 2:
        raise ValueError('Jackknife covariance requires at least two regions.')
    # Batches contain multiple region samples; never call stacking in a region loop.
    blocks = []
    n_batches = min((len(regions)+batch_size-1)//batch_size, len(regions)//2)
    size, remainder = divmod(len(regions), n_batches)
    for batch in range(n_batches):
        start = batch*size + min(batch, remainder)
        stop = start + size + (batch < remainder)
        labels = regions[start:stop]
        sums = aggregate(lens, labels)
        rsums = None if random is None else aggregate(random, labels)
        values = stack_sums(sums, rsums, shear=shear,
                            component=component, **options)
        value = values['gt' if shear else 'ds']
        if difference:
            second = stack_sums(aggregate(lens2, labels),
                                None if random2 is None else aggregate(random2, labels),
                                shear=shear, component=component, **options)
            value = value - second['gt' if shear else 'ds']
        blocks.append(value)
    samples = cp.concatenate(blocks, axis=0)
    cov = _covariance(samples)
    if samples.shape[1] == 1:
        cov = cov[0, 0]  # Match numpy.cov's historical one-output shape.
    if not return_device:
        cov, samples = cp.asnumpy(cov), cp.asnumpy(samples)
        unit = _density_unit(table_l) if not shear else None
        if unit is not None:
            cov = cov * unit**2
            samples = [row * unit for row in samples]
        else:
            samples = list(samples)
    return (cov, samples) if return_samples else cov


@_nvtx('dsigma.covariance')
def _covariance(samples):
    centered = samples - samples.mean(axis=0)
    return ((len(samples)-1)/len(samples)) * (centered.T @ centered)


def _density_unit(catalog):
    if isinstance(catalog, DeviceCatalog):
        return catalog.units.get('sum w_ls e_t sigma_crit')
    return getattr(catalog['sum w_ls e_t sigma_crit'], 'unit', None)
