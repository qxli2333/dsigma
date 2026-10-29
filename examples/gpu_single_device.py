"""Minimal synthetic example: python examples/gpu_single_device.py."""
import numpy as np
from astropy.table import Table
from astropy import units as u
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density
from dsigma.jackknife import jackknife_resampling
from dsigma.gpu import GPUSession

rng = np.random.default_rng(42)
lenses = Table(dict(ra=rng.uniform(0, 1, 200), dec=rng.uniform(0, 1, 200),
                    z=rng.uniform(.1, .3, 200), w_sys=np.ones(200),
                    field_jk=np.arange(200)%8))
sources = Table(dict(ra=rng.uniform(-1, 2, 2000), dec=rng.uniform(-1, 2, 2000),
                     z=rng.uniform(.5, 1, 2000), w=np.ones(2000),
                     e_1=rng.normal(0, .2, 2000), e_2=rng.normal(0, .2, 2000)))
with GPUSession() as session:
    device = precompute(lenses, sources, np.geomspace(.1, 5, 8)*u.Mpc,
                        backend='gpu', session=session, return_device=True, cross=True)
    print(excess_surface_density(device, backend='gpu'))
    covariance, samples = jackknife_resampling(
        excess_surface_density, device, backend='gpu', return_samples=True)
    print('jackknife samples:', samples.shape, 'covariance:', covariance.shape)
    print('cross:', excess_surface_density(device, backend='gpu', component='x'))
