"""GPU scientific parity checks; skipped when the optional device is absent."""
import itertools
import numpy as np
import pytest
from astropy.table import Table
from astropy import units as u
from dsigma.precompute import precompute
from dsigma import stacking
from dsigma.jackknife import jackknife_resampling
from dsigma.gpu import DeviceCatalog, GPUSession


@pytest.fixture(scope='module')
def cp():
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('No CUDA driver')
    return cp


def catalogs():
    rng = np.random.default_rng(3942)
    l = Table(dict(ra=[359.9, 0., 1., 2., 90., 270.],
                   dec=[0., .1, -.2, .3, 89.8, -89.8],
                   z=np.linspace(.1, .4, 6), w_sys=[0., 1., 2., .5, 1., 3.],
                   field_jk=[2, 9, 2, 14, 9, 14]))
    n = 300
    s = Table(dict(ra=rng.uniform(-3, 3, n)%360, dec=rng.uniform(-2, 2, n),
                   z=rng.uniform(.05, 1., n), w=rng.uniform(0, 2, n),
                   e_1=rng.normal(0, .2, n), e_2=rng.normal(0, .2, n)))
    s['w'][::7] = 0
    s['ra'][-2:] = [91, 271]
    s['dec'][-2:] = [89.9, -89.9]
    for k, value in dict(m=.02, m_sel=.01, e_rms=.2, R_11=.8, R_22=.9,
                         R_12=.03, R_21=.04).items():
        s[k] = value + rng.uniform(-.001, .001, n)
    return l, s


def close(a, b, rtol=1e-8, atol=1e-10):
    if isinstance(a, (list, tuple)) and len(a) and isinstance(a[0], u.Quantity):
        a = u.Quantity(a).value
    if isinstance(b, (list, tuple)) and len(b) and isinstance(b[0], u.Quantity):
        b = u.Quantity(b).value
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(np.isposinf(a), np.isposinf(b))
    np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))


@pytest.mark.parametrize('weighting,cut,comoving,angular', [
    (-2, 0, True, True), (-1.3, .2, False, False),
    (0, None, True, True), (1, 0, False, True)])
def test_pairs(cp, weighting, cut, comoving, angular):
    l, s = catalogs()
    bins = np.array([0, .01, .3, 1., 4., 10.]) * (u.deg if angular else u.Mpc)
    opts = dict(weighting=weighting, lens_source_cut=cut, comoving=comoving, nside=8)
    cpu = precompute(l.copy(), s, bins, **opts)
    gpu = precompute(l.copy(), s, bins, backend='gpu', cross=True, **opts)
    for k in cpu.colnames:
        if k.startswith('sum '):
            if k == 'sum 1':
                np.testing.assert_array_equal(cpu[k], gpu[k])
            else:
                # Conservative absolute contribution bound for this small catalog.
                close(cpu[k], gpu[k], rtol=1e-10, atol=1e-10)
    rotated = s.copy()
    rotated['e_1'], rotated['e_2'] = s['e_2'], -s['e_1']
    x = precompute(l.copy(), rotated, bins, **opts)
    close(x['sum w_ls e_t'], gpu['sum w_ls e_x'])
    close(x['sum w_ls e_t sigma_crit'], gpu['sum w_ls e_x sigma_crit'])
    # An independent geometric pair count, using the documented open lower edge.
    if weighting == -2:
        from astropy.coordinates import SkyCoord
        angles = SkyCoord(l['ra']*u.deg, l['dec']*u.deg)[:, None].separation(
            SkyCoord(s['ra']*u.deg, s['dec']*u.deg)[None, :]).deg
        valid = (np.asarray(s['z'])[None, :] >= np.asarray(l['z'])[:, None]) & (np.asarray(s['w'])[None, :] != 0)
        expected = np.stack([np.sum(valid & (angles > lo) & (angles <= hi), axis=1)
                             for lo, hi in zip(bins.value[:-1], bins.value[1:])], axis=1)
        np.testing.assert_array_equal(cpu['sum 1'], expected)


def test_calibration_chunks_session(cp):
    l, s = catalogs()
    bins = np.array([0, .1, 1, 5])*u.deg
    cal = s.copy(); cal['z_true'] = cal['z']+.02; cal['w_sys'] = 1.
    cpu = precompute(l.copy(), s, bins, table_c=cal, nside=8)
    outputs = []
    for sc, lc in [(7, 1), (64, 4)]:
        with GPUSession(source_chunk=sc, lens_chunk=lc) as session:
            gpu = precompute(l.copy(), s, bins, table_c=cal, nside=8,
                             backend='gpu', cross=True, session=session, return_device=True)
            again = precompute(l.copy(), s, bins, table_c=cal, nside=8,
                               backend='gpu', session=session, return_device=True)
            for key in cpu.colnames:
                if key.startswith('sum '):
                    close(cpu[key], cp.asnumpy(gpu[key]))
                    close(cpu[key], cp.asnumpy(again[key]))
            outputs.append(gpu.to_table())
            selected = gpu[cp.asarray([True, False, True, False, True, False])].with_regions([1, 2, 3])
            assert len(selected.to_table()) == 3
    for key in cpu.colnames:
        if key.startswith('sum '):
            close(outputs[0][key], outputs[1][key])


def test_nz(cp):
    l, s = catalogs()
    s['z_bin'] = np.arange(len(s))%2
    z = np.linspace(.01, 2, 101)
    nz = Table(dict(z=z, n=np.stack([np.exp(-((z-.6)/.2)**2), np.exp(-((z-.9)/.3)**2)], axis=1)))
    bins = [0., .1, 1., 5.]*u.deg
    opts = dict(table_n=nz, lens_source_cut=None, nside=8)
    a = precompute(l.copy(), s, bins, **opts)
    b = precompute(l.copy(), s, bins, backend='gpu', **opts)
    for key in a.colnames:
        if key.startswith('sum '): close(a[key], b[key])


@pytest.mark.parametrize('stat', [stacking.tangential_shear, stacking.excess_surface_density])
def test_stack_jackknife(cp, stat):
    l, s = catalogs()
    cal = s.copy(); cal['z_true'] = cal['z']+.02; cal['w_sys'] = 1.
    l = precompute(l, s, [0., .01, .3, 1., 5.]*u.deg, table_c=cal, nside=8)
    r = l[[0, 1, 2, 4]].copy()  # region 14 absent in randoms
    r['w_sys'] = [1., .5, 2., .7]
    for scalar, matrix, resp, selection, boost, random, photo in itertools.product([False, True], repeat=7):
        opts = dict(scalar_shear_response_correction=scalar, matrix_shear_response_correction=matrix,
                    shear_responsivity_correction=resp, selection_bias_correction=selection,
                    boost_correction=boost, random_subtraction=random)
        if stat is stacking.excess_surface_density: opts['photo_z_dilution_correction'] = photo
        close(stat(l, table_r=r, **opts), stat(l, table_r=r, backend='gpu', **opts))
        cov, samples = jackknife_resampling(stat, l, table_r=r, return_samples=True, compress=False, **opts)
        gcov, gsamples = jackknife_resampling(stat, l, table_r=r, backend='gpu', return_samples=True, batch_size=2, **opts)
        close(samples, gsamples); close(cov, gcov)
    table = stat(l, table_r=r, backend='gpu', return_table=True, **opts)
    original = stat(l, table_r=r, return_table=True, **opts)
    assert table.colnames == original.colnames
    for key in original.colnames: close(table[key], original[key])


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite(cp, value):
    l, s = catalogs()
    l = precompute(l, s, [0., .1, 1., 5.]*u.deg, nside=8)
    l['sum w_ls e_t'][1, 1] = value
    cov, samples = jackknife_resampling(stacking.tangential_shear, l, return_samples=True, compress=False)
    gcov, gsamples = jackknife_resampling(stacking.tangential_shear, l, backend='gpu', return_samples=True, compress=False)
    close(samples, gsamples); close(cov, gcov)


def test_invalid_backend():
    with pytest.raises(ValueError): stacking.tangential_shear(None, backend='typo')
    with pytest.raises(ValueError): precompute(None, None, None, backend='typo')


def test_second_catalog_pair(cp):
    l, s = catalogs()
    l = precompute(l, s, [.1, 1., 5.]*u.deg, nside=8)
    l2 = l[[1, 2, 3, 4]].copy()
    r = l[[0, 1, 4]].copy()
    for f in (stacking.tangential_shear_difference, stacking.excess_surface_density_difference):
        opts = dict(table_l_2=l2, table_r=r, table_r_2=r,
                    random_subtraction=True, boost_correction=True, return_samples=True, compress=False)
        c, x = jackknife_resampling(f, l, **opts)
        gc, gx = jackknife_resampling(f, l, backend='gpu', **opts)
        close(x, gx); close(c, gc)


def test_streaming_and_edges(cp):
    l = Table(dict(ra=[0., 0.], dec=[0., 0.], z=[.2, .3], w_sys=[1., 1.]))
    s = Table(dict(ra=[0., 1., 2., 3.], dec=[0.]*4, z=[.7]*4,
                   w=[1.]*4, e_1=[.2]*4, e_2=[.1]*4))
    bins = [0., 1., 2., 3.]*u.deg
    a = precompute(l.copy(), s, bins, nside=8)
    session = GPUSession(source_chunk=1, lens_chunk=1)
    session.budget = 10000  # force streaming with a separate larger source
    s = Table(np.repeat(s.as_array(), 100))
    a = precompute(l.copy(), s, bins, nside=8)
    # Enough for outputs and one staged source range, below source residency.
    session.budget = 18000
    b = precompute(l.copy(), s, bins, nside=8, backend='gpu', session=session)
    for key in a.colnames:
        if key.startswith('sum '): close(a[key], b[key])
    assert session.source is None


def test_independent_pair_statistics(cp):
    """Astropy great-circle bearings, independent brute-force weighted sums."""
    from astropy.coordinates import SkyCoord
    from dsigma import default_cosmology
    from dsigma.helpers import interpolate_over_redshift
    from dsigma.physics import critical_surface_density
    l, s = catalogs()
    bins = np.array([0., .1, .3, 1., 4., 10.])*u.deg
    cosmology = default_cosmology
    def distances(z):
        return interpolate_over_redshift(cosmology.comoving_transverse_distance, z)
    dl, ds = distances(l['z']), distances(s['z'])
    gpu = precompute(l.copy(), s, bins, backend='gpu', nside=8)
    cpu = precompute(l.copy(), s, bins, nside=8)
    sc = SkyCoord(s['ra']*u.deg, s['dec']*u.deg)
    for i, lens in enumerate(l):
        lc = SkyCoord(lens['ra']*u.deg, lens['dec']*u.deg)
        theta = sc.separation(lc).deg
        pa = sc.position_angle(lc).rad
        co, si = np.cos(2*pa), np.sin(2*pa)
        et = s['e_1']*co+s['e_2']*si
        sigma = critical_surface_density(lens['z'], s['z'], d_l=dl[i], d_s=ds, comoving=True).value
        weight = np.asarray(s['w'])/sigma**2
        valid = (np.asarray(s['z']) >= lens['z']) & (weight != 0)
        values = {'sum w_ls': weight, 'sum w_ls e_t': weight*et,
                  'sum w_ls e_t sigma_crit': (weight*et)*sigma,
                  'sum w_ls sigma_crit': weight*sigma,
                  'sum w_ls z_l': weight*lens['z'], 'sum w_ls z_s': weight*s['z'],
                  'sum w_ls m': weight*s['m'], 'sum w_ls m_sel': weight*s['m_sel'],
                  'sum w_ls (1 - e_rms^2)': weight*(1-s['e_rms']**2),
                  'sum w_ls R_T': weight*(s['R_11']*co**2+s['R_22']*si**2-(s['R_12']+s['R_21'])*si*co)}
        for b, (low, high) in enumerate(zip(bins.value[:-1], bins.value[1:])):
            mask = valid & (theta > low) & (theta <= high)
            for key, terms in values.items():
                expected = np.sum(terms[mask])
                tolerance = 1e-10*max(1., np.sum(np.abs(terms[mask])))
                assert abs(cpu[key][i, b]-expected) <= tolerance
                assert abs(gpu[key][i, b]-expected) <= tolerance


@pytest.mark.parametrize('weight_value', [0., np.nan, np.inf, -np.inf])
def test_large_region_set_single_bin(cp, weight_value):
    n = 103
    rng = np.random.default_rng(12)
    l = Table(dict(w_sys=rng.uniform(.1, 2, n), field_jk=np.arange(n)*7-400))
    l['w_sys'][0] = weight_value
    l['sum 1'] = np.ones((n, 1), dtype=np.int64)
    l['sum 1'][2, 0] = 2**53+7
    l['sum w_ls'] = rng.uniform(.1, 2, (n, 1))
    l['sum w_ls e_t'] = rng.normal(0, .2, (n, 1))
    l['sum w_ls z_l'] = np.asarray(l['sum w_ls'])*.2
    l['sum w_ls z_s'] = np.asarray(l['sum w_ls'])*.7
    l.meta['bins'] = np.array([.1, 1.])*u.Mpc
    c, x = jackknife_resampling(stacking.tangential_shear, l, return_samples=True, compress=False)
    gc, gx = jackknife_resampling(stacking.tangential_shear, l, backend='gpu', return_samples=True, return_device=True, compress=False)
    close(x, cp.asnumpy(gx)); close(c, cp.asnumpy(gc))
    original = stacking.tangential_shear(l, return_table=True)
    gpu = stacking.tangential_shear(l, return_table=True, backend='gpu')
    np.testing.assert_array_equal(original['n_pairs'], gpu['n_pairs'])
    assert gpu['n_pairs'].dtype == np.int64
    assert np.shape(gc) == np.shape(c)


def test_retained_empty_bin_and_custom_cpu(cp):
    l, s = catalogs()
    l = precompute(l, s, [0., 1., 5.]*u.deg, nside=8)
    # Only the removed region contributes to the first bin.
    for key in l.colnames:
        if key.startswith('sum '): l[key][l['field_jk'] != 2, 0] = 0
    c, x = jackknife_resampling(stacking.tangential_shear, l, return_samples=True, compress=False)
    gc, gx = jackknife_resampling(stacking.tangential_shear, l, backend='gpu', return_samples=True, compress=False)
    close(x, gx); close(c, gc)
    def custom(table, table_l_2=None):
        return stacking.tangential_shear(table)-stacking.tangential_shear(table_l_2)
    result = jackknife_resampling(custom, l, table_l_2=l, compress=False)
    close(result, np.array([[np.nan, np.nan], [np.nan, 0.]]))
    with pytest.raises(ValueError, match='custom'):
        jackknife_resampling(custom, l, table_l_2=l, backend='gpu')


def test_fits_endianness_and_session_binding(cp):
    l, s = catalogs()
    l = precompute(l, s, [.1, 1., 5.]*u.deg, nside=8)
    for key in l.colnames:
        if key.startswith('sum ') or key in ('w_sys', 'field_jk'):
            l[key] = np.asarray(l[key]).astype(l[key].dtype.newbyteorder('>'))
    close(stacking.tangential_shear(l), stacking.tangential_shear(l, backend='gpu'))
    a, x = jackknife_resampling(stacking.tangential_shear, l, return_samples=True, compress=False)
    b, y = jackknife_resampling(stacking.tangential_shear, l, backend='gpu', return_samples=True, compress=False)
    close(a, b); close(x, y)
    with GPUSession() as session:
        precompute(l.copy(), s, [.1, 1., 5.]*u.deg, nside=8, backend='gpu', session=session)
        with pytest.raises(ValueError, match='bound'):
            precompute(l.copy(), s.copy(), [.1, 1., 5.]*u.deg, nside=8, backend='gpu', session=session)


@pytest.mark.parametrize('tomographic', [False, True])
def test_upstream_units_and_source_cut(cp, tomographic):
    """Upstream 1.2 units, radians, column cuts and n(z) without source z."""
    l, s = catalogs()
    l['ra'] = np.deg2rad(l['ra']) * u.rad
    l['dec'] = np.deg2rad(l['dec']) * u.rad
    s['ra'] = s['ra'] * u.deg
    s['dec'] = s['dec'] * u.deg
    s['z_l_max'] = np.asarray(s['z']) - .1
    options = dict(nside=8)
    if tomographic:
        s['z_bin'] = np.arange(len(s)) % 2
        s.remove_column('z')
        z = np.linspace(.01, 2, 101)
        options['table_n'] = Table(dict(z=z, n=np.stack([
            np.exp(-((z-.6)/.2)**2), np.exp(-((z-.9)/.3)**2)], axis=1)))
    bins = np.array([0., .03, .1, 1., 600.]) * u.arcmin
    cpu = precompute(l.copy(), s, bins, **options)
    device = precompute(l.copy(), s, bins, backend='gpu', return_device=True, **options)
    host = device.to_table()
    for key in cpu.colnames:
        if key.startswith('sum '):
            close(cpu[key], host[key])
            assert cpu[key].unit == host[key].unit
    a = stacking.excess_surface_density(cpu)
    b = stacking.excess_surface_density(device, backend='gpu')
    assert a.unit == b.unit
    close(a, b)
    c, x = jackknife_resampling(stacking.excess_surface_density, cpu, return_samples=True)
    gc, gx = jackknife_resampling(stacking.excess_surface_density, device,
                                  backend='gpu', return_samples=True)
    assert c.unit == gc.unit
    assert x[0].unit == gx[0].unit
    close(c, gc)
    close(x, gx)


def test_upstream_pairs_below_pixel_size(cp):
    from astropy.coordinates import SkyCoord
    l = Table(dict(ra=[10., 18.], dec=[-8., 8.], z=[.2, .2], w_sys=[1., 1.]))
    position = SkyCoord(l['ra']*u.deg, l['dec']*u.deg).directional_offset_by(
        0*u.deg, 30*u.arcsec)
    s = Table(dict(ra=position.ra, dec=position.dec, z=[.8, .8],
                   w=[1., 1.], e_1=[0., 0.], e_2=[0., 0.]))
    bins = [15, 60]*u.arcsec
    cpu = precompute(l.copy(), s, bins)
    gpu = precompute(l.copy(), s, bins, backend='gpu')
    np.testing.assert_array_equal(cpu['sum 1'], [[1], [1]])
    np.testing.assert_array_equal(gpu['sum 1'], cpu['sum 1'])
