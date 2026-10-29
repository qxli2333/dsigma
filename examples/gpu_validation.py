#!/usr/bin/env python
"""Reproducible Perlmutter scientific and timing acceptance against CPU.

Run from an installed checkout. All outputs go to --output (scratch by default).
The external measurement module is read as a source adapter, never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import threading
from datetime import datetime, timezone

import numpy as np
from astropy.table import Table, Column
from astropy.cosmology import Planck18

from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.helpers import dsigma_table
from dsigma.gpu import GPUSession, DeviceCatalog, aggregate, stack_sums, cupy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, default=Path('/global/u1/q/qinxunli/dev/sdss_bh_lensing_calibrated'))
    ap.add_argument('--survey', default='unions1.6.6', choices=['unions1.6.6', 'desy3', 'decade_ngc', 'decade_sgc'])
    ap.add_argument('--source-limit', type=int, help='Fixed-seed source subset; omitted for full catalog')
    ap.add_argument('--lens-limit', type=int)
    ap.add_argument('--random-limit', type=int, default=1000)
    ap.add_argument('--cpu-jobs', type=int, default=4)
    ap.add_argument('--output', type=Path, default=Path('/pscratch/sd/q/qinxunli/dsigma_gpu_validation/acceptance'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    cp = cupy()
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), survey=args.survey, source_limit=args.source_limit, seed=42,
                  cpu_jobs=args.cpu_jobs, timings={}, checks=[], versions={},
                  reference=str(args.reference))
    code_root = Path(__file__).resolve().parents[1]
    report['code_sha256'] = {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [code_root/'dsigma/gpu.py', code_root/'dsigma/gpu_pairs.cu',
                                       code_root/'dsigma/precompute.py', Path(__file__).resolve()]}
    peak = {'device_pool_bytes': 0, 'device_used_bytes': 0}
    def sample_memory():
        while True:
            free, total = cp.cuda.runtime.memGetInfo()
            peak['device_used_bytes'] = max(peak['device_used_bytes'], total-free)
            peak['device_pool_bytes'] = max(peak['device_pool_bytes'], cp.get_default_memory_pool().total_bytes())
            time.sleep(.1)
    threading.Thread(target=sample_memory, daemon=True).start()
    import dsigma, scipy, astropy
    report['versions'] = dict(dsigma=dsigma.__version__, numpy=np.__version__,
                              scipy=scipy.__version__, astropy=astropy.__version__, cupy=cp.__version__,
                              driver=cp.cuda.runtime.driverGetVersion(), runtime=cp.cuda.runtime.runtimeGetVersion())
    report['gpu'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    report['reference_sha256'] = hashlib.sha256((args.reference/'legacy/test_cal_unions_dsigma.py').read_bytes()).hexdigest()
    def save():
        report['wall_elapsed_seconds'] = time.perf_counter()-wall_start
        report['host_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['memory_peaks'] = dict(peak)
        report['device_pool_reserved_bytes'] = cp.get_default_memory_pool().total_bytes()
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    def timed(name, func):
        cp.cuda.get_current_stream().synchronize()
        cp.cuda.nvtx.RangePush(name)
        start = time.perf_counter()
        try:
            result = func()
            cp.cuda.get_current_stream().synchronize()
        finally:
            report['timings'][name] = time.perf_counter()-start
            cp.cuda.nvtx.RangePop()
            print(name, report['timings'][name], flush=True)
            save()
        return result
    def compare(name, a, b, rtol=1e-8, atol=1e-10):
        a, b = np.asarray(a), np.asarray(b)
        if rtol == 1e-10:
            finite = np.isfinite(a) & np.isfinite(b)
            # This is stricter than the requested absolute-contribution gate:
            # |sum(terms)| <= sum(|terms|). No automatic tolerance relaxation.
            errors = np.abs(a[finite]-b[finite])/(1e-10*np.maximum(1., np.abs(a[finite])))
            assert np.all(errors <= 1), (name, 'maximum normalized error', errors.max())
        else:
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        for f in (np.isnan, np.isposinf, np.isneginf): np.testing.assert_array_equal(f(a), f(b))
        report['checks'].append(name)
        save()
    sys.path.insert(0, str(args.reference/'legacy'))
    import test_cal_unions_dsigma as reference
    rng = np.random.default_rng(42)
    def prepare():
        # The FITS adapter reads only science columns. HDF5 regressions are
        # fixed regular subsets before conversion to avoid loading full surveys.
        if args.survey == 'unions1.6.6':
            source, nz, opts = reference.prepare_source(args.survey)
            if args.source_limit and args.source_limit < len(source):
                ix = np.sort(rng.choice(len(source), args.source_limit, replace=False))
                source = source[ix]
            return source, nz, opts
        if not args.source_limit:
            return reference.prepare_source(args.survey)
        import h5py
        base = Path('/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/lensingsurvey_catalogues')
        decade = args.survey.startswith('decade')
        path = base / (f'decade_zhiwei/{args.survey}_dsigma.hdf5' if decade else 'desy3/raw/des_y3.hdf5')
        with h5py.File(path, 'r') as f:
            dataset = f['galaxies' if decade else 'catalog']
            stride = max(1, len(dataset)//args.source_limit)
            raw = Table(dataset[::stride][:args.source_limit])
        if decade:
            source = dsigma_table(raw, 'source', weight='w', z_bin='z_bin', z='z', R_11='R_11', R_22='R_22', R_12='R_12', R_21='R_21', m='m')
            nz = Table.read(path, path='redshift')
            opts = dict(matrix_shear_response_correction=True, scalar_shear_response_correction=True)
        else:
            from dsigma.surveys import des
            source = dsigma_table(raw, 'source', survey='DES')
            source['m_sel'] = 0.
            for zbin in range(4):
                select = source['z_bin'] == zbin
                source['m_sel'][select] = .5*np.trace(des.selection_response(source[select]))
            source = source[(source['z_bin'] >= 0) & source['flags_select']]
            source['m'] = des.multiplicative_shear_bias(source['z_bin'], version='Y3')
            source['z'] = np.array([0., .358, .631, .872])[source['z_bin']]
            nz = Table.read(base/'desy3/raw/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits', hdu=6)
            nz.rename_column('Z_MID', 'z')
            nz['n'] = np.column_stack([nz[f'BIN{i+1}'] for i in range(4)])
            nz.keep_columns(['z', 'n'])
            opts = dict(matrix_shear_response_correction=True, scalar_shear_response_correction=True, selection_bias_correction=True)
        return source, nz, opts
    source, nz, options = timed('source_read_prepare', prepare)
    report['source_fingerprints'] = {k: hashlib.sha256(np.ascontiguousarray(source[k]).view(np.uint8)).hexdigest()
                                     for k in source.colnames if k in ('ra', 'dec', 'w', 'e_1', 'e_2', 'z', 'z_bin')}
    report['nz_fingerprints'] = {k: hashlib.sha256(np.ascontiguousarray(nz[k]).view(np.uint8)).hexdigest()
                                for k in ('z', 'n')}
    options.pop('return_table', None)
    cat = Table.read(args.reference/'outputs/catalogs/sdss_type1_agn_calibrated_ws22.fits')
    cat = cat[np.char.strip(np.asarray(cat['WL_SURVEY']).astype(str)) == args.survey]
    if args.lens_limit: cat = cat[:args.lens_limit]
    lenses = dsigma_table(cat, 'lens', ra='RA', dec='DEC', z='Z', w_sys=1., verbose=False)
    report.update(n_lens=len(lenses), n_source=len(source))
    report['lens_fingerprints'] = {k: hashlib.sha256(np.ascontiguousarray(cat[k]).view(np.uint8)).hexdigest()
                                  for k in ('RA', 'DEC', 'Z', 'LOGMBH')}
    bins = np.geomspace(.02, 20., 16)/.6766
    prep = dict(table_n=nz, cosmology=Planck18, lens_source_cut=None if args.survey=='unions1.6.6' else .2)
    report['bins_Mpc'] = bins.tolist()
    report['options'] = options
    session = GPUSession()
    timed('cuda_pair_compile', session.kernel.compile)
    gpu = timed('gpu_precompute', lambda: precompute(lenses.copy(), source, bins, backend='gpu', session=session, cross=True, return_device=True, **prep))
    report['session_budget_bytes'] = session.budget
    cpu = timed('cpu_single_precompute', lambda: precompute(lenses.copy(), source, bins, n_jobs=1, **prep))
    for key in cpu.colnames:
        if key.startswith('sum '):
            g = cp.asnumpy(gpu[key])
            if key == 'sum 1': np.testing.assert_array_equal(cpu[key], g)
            else: compare(key, cpu[key], g, rtol=1e-10, atol=1e-10)
    if args.cpu_jobs > 1:
        parallel = timed('cpu_parallel_precompute', lambda: precompute(lenses.copy(), source, bins, n_jobs=args.cpu_jobs, **prep))
        for key in cpu.colnames:
            if key.startswith('sum '): compare('parallel '+key, cpu[key], parallel[key], rtol=1e-10)
        del parallel
    # Cross comparison reuses the immutable source table for the GPU, and the
    # CPU reference rotates a separate source copy as production does.
    # Host validation must not duplicate the full 88-million-row source table.
    # Releasing the host preparation cache leaves session device buffers and
    # its spatial index available for the later random-catalog regression.
    session._prepared_source = None
    rotated = source.copy(copy_data=False)
    rotated.replace_column('e_1', Column(source['e_2'], name='e_1', copy=False))
    rotated.replace_column('e_2', Column(-np.asarray(source['e_1']), name='e_2', copy=False))
    cross = timed('cpu_cross_precompute', lambda: precompute(lenses.copy(), rotated, bins, n_jobs=1, **prep))
    for suffix in ('', ' sigma_crit'):
        compare('cross'+suffix, cross['sum w_ls e_t'+suffix], cp.asnumpy(gpu['sum w_ls e_x'+suffix]), rtol=1e-10)
    del rotated
    mass_edges = [5., 7.47, 7.84, 8.16, 8.5, 11.]
    for lo, hi in zip(mass_edges[:-1], mass_edges[1:]):
        label = f'mbh{lo:.2f}_{hi:.2f}'
        select = (cat['LOGMBH'] >= lo) & (cat['LOGMBH'] < hi) & (np.sum(cpu['sum 1'], axis=1)>0)
        c = cpu[select]
        if len(c) < 30:
            report.setdefault('skipped_bins', []).append(label); continue
        np.random.seed(42)
        timed(label+'_cpu_regions', lambda: compute_jackknife_fields(c, min(64, len(c)//10), weights=np.sum(c['sum 1'],axis=1), distance_threshold=10))
        g = gpu[cp.asarray(select)].with_regions(np.asarray(c['field_jk']))
        for component in ('t', 'x'):
            name = label+'_'+component
            cr = c if component == 't' else cross[select]
            cr['field_jk'] = c['field_jk']
            signal = timed(name+'_cpu_stack', lambda: excess_surface_density(cr, **options))
            gsignal = timed(name+'_gpu_stack', lambda: excess_surface_density(g, backend='gpu', component=component, **options))
            compare(name+'_signal', signal, gsignal)
            cov, samples = timed(name+'_cpu_jk', lambda: jackknife_resampling(excess_surface_density, cr, return_samples=True, **options))
            gcov, gsamples = timed(name+'_gpu_jk', lambda: jackknife_resampling(excess_surface_density, g, backend='gpu', component=component, return_samples=True, **options))
            compare(name+'_samples', samples, gsamples); compare(name+'_covariance', cov, gcov)
            timed(name+'_gpu_jk_warm', lambda: jackknife_resampling(excess_surface_density, g, backend='gpu', component=component, **options))
            np.savez(args.output/(name+'.npz'), signal=signal, gpu_signal=gsignal, covariance=cov, gpu_covariance=gcov, samples=samples, gpu_samples=gsamples, regions=np.asarray(c['field_jk']))
        # Stage-specific benchmarks use all regions in one device batch.
        regions = cp.unique(g['field_jk'])
        sums = timed(label+'_gpu_region_aggregation', lambda: aggregate(g, regions))
        values = timed(label+'_gpu_batched_stack', lambda: stack_sums(sums, **options)['ds'])
        def covariance():
            centered = values-values.mean(axis=0)
            return (len(regions)-1)/len(regions)*(centered.T@centered)
        timed(label+'_gpu_covariance', covariance)
    if args.random_limit:
        random_path = Path('/pscratch/sd/q/qinxunli/lensing_catalog/lens/sdss_agn/sdss_type1_agn_randoms.fits')
        raw = Table.read(random_path)
        raw = raw[np.char.strip(np.asarray(raw['WL_SURVEY']).astype(str)) == args.survey]
        raw = raw[:args.random_limit]
        rand = dsigma_table(raw, 'lens', ra='RA', dec='DEC', z='Z', w_sys=1., verbose=False)
        if len(rand):
            r = timed('cpu_random_precompute', lambda: precompute(rand.copy(), source, bins, n_jobs=args.cpu_jobs, **prep))
            gr = timed('gpu_random_precompute_cached', lambda: precompute(rand.copy(), source, bins, backend='gpu', session=session, return_device=True, **prep))
            # Use deterministic shared region labels for the random regression.
            cpu['field_jk'] = np.arange(len(cpu))%64
            r['field_jk'] = np.arange(len(r))%61
            g = gpu.with_regions(np.asarray(cpu['field_jk']))
            gr = gr.with_regions(np.asarray(r['field_jk']))
            opts = dict(options, random_subtraction=True, boost_correction=True)
            a, x = jackknife_resampling(excess_surface_density, cpu, table_r=r, return_samples=True, **opts)
            b, y = jackknife_resampling(excess_surface_density, g, table_r=gr, backend='gpu', return_samples=True, **opts)
            compare('random_samples', x, y); compare('random_covariance', a, b)
    timing = report['timings']
    common = timing['source_read_prepare'] + sum(v for k, v in timing.items() if k.endswith('_cpu_regions'))
    cpu_stack = sum(v for k, v in timing.items() if k.endswith('_cpu_stack') or k.endswith('_cpu_jk'))
    gpu_stack = sum(v for k, v in timing.items() if k.endswith('_gpu_stack') or k.endswith('_gpu_jk'))
    report['main_pipeline_seconds'] = dict(
        cpu_single=common+timing['cpu_single_precompute']+timing['cpu_cross_precompute']+cpu_stack,
        gpu_including_cold=common+timing['cuda_pair_compile']+timing['gpu_precompute']+gpu_stack)
    report['status'] = 'passed'
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
