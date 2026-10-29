"""Profile 64 jackknife samples in shared GPU launches (no pair computation)."""
import time
import numpy as np
from astropy.table import Table
from astropy import units as u
from dsigma.gpu import DeviceCatalog, cupy
from dsigma.jackknife import jackknife_resampling
from dsigma.stacking import excess_surface_density

cp = cupy()
rng = np.random.default_rng(42)
n, nb, njk = 6400, 15, 64
catalog = Table(dict(w_sys=rng.uniform(.5, 2, n), field_jk=np.arange(n)%njk))
catalog['sum 1'] = rng.integers(1, 1000, (n, nb), dtype=np.int64)
w = rng.uniform(.01, 1, (n, nb))
catalog['sum w_ls'] = w
catalog['sum w_ls e_t'] = rng.normal(0, .1, (n, nb))*w
catalog['sum w_ls e_t sigma_crit'] = catalog['sum w_ls e_t']*3000
catalog['sum w_ls z_l'] = .2*w
catalog['sum w_ls z_s'] = .7*w
catalog['sum w_ls sigma_crit'] = 3000*w
catalog.meta['bins'] = np.geomspace(.1, 20, nb+1)*u.Mpc
device = DeviceCatalog.from_table(catalog)
for label in ('cold', 'warm'):
    cp.cuda.nvtx.RangePush(label)
    start = time.perf_counter()
    cov, samples = jackknife_resampling(excess_surface_density, device, backend='gpu',
                                        return_device=True, return_samples=True)
    cp.cuda.get_current_stream().synchronize()
    assert samples.shape == (njk, nb)
    assert cov.shape == (nb, nb)
    print(label, time.perf_counter()-start, 'seconds', samples.shape, flush=True)
    cp.cuda.nvtx.RangePop()
