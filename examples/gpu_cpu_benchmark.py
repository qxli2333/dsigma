#!/usr/bin/env python
"""Production CPU baseline on a Slurm CPU node (no CUDA dependency)."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
import threading
import time
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18
import dsigma
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--reference', type=Path, default=Path('/global/u1/q/qinxunli/dev/sdss_bh_lensing_calibrated'))
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--regions-from', type=Path, required=True)
ap.add_argument('--cpu-jobs', type=int, default=256)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
report = dict(cpu_jobs=args.cpu_jobs, timings={}, job_id=os.environ.get('SLURM_JOB_ID'),
              cpu_affinity=len(os.sched_getaffinity(0)), peak_family_pss_bytes=0,
              dsigma_path=dsigma.__file__, numpy_version=np.__version__)
try:
    import psutil
    process = psutil.Process()
    def sample():
        while True:
            try:
                pss = sum(p.memory_full_info().pss for p in [process]+process.children(recursive=True))
                report['peak_family_pss_bytes'] = max(report['peak_family_pss_bytes'], pss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(2)
    threading.Thread(target=sample, daemon=True).start()
except ImportError:
    report['peak_family_pss_bytes'] = None

def save():
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')

def timed(name, f):
    start = time.perf_counter()
    result = f()
    report['timings'][name] = time.perf_counter()-start
    print(name, report['timings'][name], flush=True)
    save()
    return result

sys.path.insert(0, str(args.reference/'legacy'))
import test_cal_unions_dsigma as reference
source, nz, opts = timed('source_read_prepare', lambda: reference.prepare_source('unions1.6.6'))
opts.pop('return_table', None)
report['source_fingerprints'] = {k: hashlib.sha256(np.ascontiguousarray(source[k]).view(np.uint8)).hexdigest()
                                 for k in source.colnames if k in ('ra', 'dec', 'w', 'e_1', 'e_2', 'z', 'z_bin')}
cat = Table.read(args.reference/'outputs/catalogs/sdss_type1_agn_calibrated_ws22.fits')
cat = cat[np.char.strip(np.asarray(cat['WL_SURVEY']).astype(str)) == 'unions1.6.6']
lens = dsigma_table(cat, 'lens', ra='RA', dec='DEC', z='Z', w_sys=1., verbose=False)
bins = np.geomspace(.02, 20., 16)/.6766
prep = dict(table_n=nz, lens_source_cut=None, cosmology=Planck18, n_jobs=args.cpu_jobs)
report.update(n_source=len(source), n_lens=len(lens), bins_Mpc=bins.tolist())
lens = timed('cpu_production_precompute', lambda: precompute(lens, source, bins, **prep))
np.savez(args.output/'precompute.npz', **{k: np.asarray(lens[k]) for k in lens.colnames if k.startswith('sum ')})
source['e_1'], source['e_2'] = source['e_2'].copy(), -source['e_1'].copy()
cross = timed('cpu_production_cross_precompute', lambda: precompute(lens.copy(), source, bins, **prep))
edges = [5., 7.47, 7.84, 8.16, 8.5, 11.]
for lo, hi in zip(edges[:-1], edges[1:]):
    label = f'mbh{lo:.2f}_{hi:.2f}'
    select = (cat['LOGMBH']>=lo) & (cat['LOGMBH']<hi) & (np.sum(lens['sum 1'],axis=1)>0)
    c = lens[select]
    path = args.regions_from/(label+'_t.npz')
    if path.exists():
        c['field_jk'] = np.load(path)['regions']
    else:
        np.random.seed(42)
        timed(label+'_cpu_regions', lambda: compute_jackknife_fields(c, min(64,len(c)//10), weights=np.sum(c['sum 1'],axis=1), distance_threshold=10))
    for component in ('t','x'):
        cr = c if component=='t' else cross[select]
        cr['field_jk'] = c['field_jk']
        signal = timed(label+'_'+component+'_stack', lambda: excess_surface_density(cr, **opts))
        cov, samples = timed(label+'_'+component+'_jk', lambda: jackknife_resampling(excess_surface_density, cr, return_samples=True, **opts))
        np.savez(args.output/(label+'_'+component+'.npz'), signal=signal, covariance=cov, samples=samples, regions=np.asarray(c['field_jk']))
report['main_pipeline_seconds'] = sum(v for k,v in report['timings'].items())
report['status']='passed'
save()
