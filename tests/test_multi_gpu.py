"""Multi-device parity and lifecycle; one-device mode also runs on a single GPU."""
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from dsigma.gpu import MultiGPUSession, GPUSession
from dsigma.precompute import precompute
from dsigma import stacking
from dsigma.jackknife import jackknife_resampling
from test_gpu import catalogs, close

pytest_plugins = ["test_gpu"]


@pytest.mark.parametrize('devices', [[], [0, 0], [-1.2], [True]])
def test_invalid_devices(devices):
    with pytest.raises(ValueError):
        MultiGPUSession(devices)


def test_invalid_visible_device(cp):
    with pytest.raises(ValueError, match='visible'):
        MultiGPUSession([cp.cuda.runtime.getDeviceCount()])
    with pytest.raises(ValueError, match='visible'):
        MultiGPUSession([-1])


@pytest.mark.parametrize('device_count', [1, 2])
@pytest.mark.parametrize('mode', ['calibration', 'tomographic', 'streaming'])
def test_multi_pair_and_profiles(cp, device_count, mode):
    if cp.cuda.runtime.getDeviceCount() < device_count:
        pytest.skip('Requires two visible CUDA devices')
    lenses, sources = catalogs()
    # Many lenses in the same pixel; chunks must split it and restore row order.
    lenses = lenses[np.tile(np.arange(len(lenses)), 3)]
    bins = np.array([0, .01, .3, 1, 4, 10])*u.deg
    options = dict(nside=8, cross=True)
    if mode == 'calibration':
        calibration = sources.copy()
        calibration['z_true'] = calibration['z']+.02
        calibration['w_sys'] = 1.
        options['table_c'] = calibration
    elif mode == 'tomographic':
        sources['z_bin'] = np.arange(len(sources)) % 2
        z = np.linspace(.01, 2, 101)
        options['table_n'] = Table(dict(
            z=z, n=np.stack([np.exp(-((z-.6)/.2)**2),
                            np.exp(-((z-.9)/.3)**2)], axis=1)))
    cpu = precompute(lenses.copy(), sources, bins,
                     **{k: v for k, v in options.items() if k != 'cross'})
    single = precompute(lenses.copy(), sources, bins, backend='gpu', **options)
    original_device = cp.cuda.Device().id
    with MultiGPUSession(list(reversed(range(device_count))), source_chunk=7,
                         lens_chunk=2) as session:
        if mode == 'streaming':
            for worker, _ in session._workers:
                worker.budget = 40000  # source exceeds half the budget
        result = precompute(lenses.copy(), sources, bins, backend='gpu',
                            session=session, **options)
        for reference in (cpu, single):
            for key in reference.colnames:
                if key.startswith('sum '):
                    if key == 'sum 1':
                        np.testing.assert_array_equal(result[key], reference[key])
                    else:
                        close(result[key], reference[key], rtol=1e-10)
                    assert result[key].unit == reference[key].unit
        assert result.meta.keys() == single.meta.keys()
        cached = session._prepared_source
        device = precompute(lenses.copy(), sources, bins, backend='gpu',
                            session=session, return_device=True, **options)
        assert all(a is b for a, b in zip(session._prepared_source, cached))
        assert cp.cuda.Device().id == original_device
        with cp.cuda.Device(session.device):
            for key in device.colnames:
                assert device[key].device.id == session.device
            for stat in (stacking.tangential_shear, stacking.excess_surface_density):
                for component in ('t', 'x'):
                    opts = dict(component=component)
                    close(stat(device, backend='gpu', **opts),
                          stat(single, backend='gpu', **opts))
                    cov, samples = jackknife_resampling(
                        stat, device, backend='gpu', return_samples=True, **opts)
                    expected_cov, expected_samples = jackknife_resampling(
                        stat, single, backend='gpu', return_samples=True, **opts)
                    close(samples, expected_samples)
                    close(cov, expected_cov)
        # Different lens count, bins and a different pixel distribution reuse sources.
        tiny = precompute(lenses[:1].copy(), sources, bins[::2], backend='gpu',
                          session=session, **options)
        expected = precompute(lenses[:1].copy(), sources, bins[::2], backend='gpu', **options)
        close(tiny['sum w_ls'], expected['sum w_ls'])
        assert all(worker.tree is session.tree for worker, _ in session._workers)
    session.close()
    with pytest.raises(RuntimeError, match='closed'):
        precompute(lenses.copy(), sources, bins, backend='gpu', session=session)


def test_worker_failure_and_source_binding(cp, monkeypatch):
    lenses, sources = catalogs()
    bins = np.array([0, .1, 1, 5])*u.deg
    with MultiGPUSession([0], lens_chunk=1) as session:
        worker = session._workers[0][0]
        real_pair = worker.pair

        def fail(*args, **kwargs):
            raise RuntimeError('injected worker failure')

        monkeypatch.setattr(worker, 'pair', fail)
        with pytest.raises(RuntimeError, match='injected'):
            precompute(lenses, sources, bins, backend='gpu', session=session)
        assert not any(key.startswith('sum ') for key in lenses.colnames)
        monkeypatch.setattr(worker, 'pair', real_pair)
        precompute(lenses.copy(), sources, bins, backend='gpu', session=session)
        with pytest.raises(ValueError, match='bound'):
            precompute(lenses.copy(), sources.copy(), bins, backend='gpu', session=session)


def test_initialization_failure_cleans_workers(cp, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('injected initialization failure')

    monkeypatch.setattr(GPUSession, '__init__', fail)
    with pytest.raises(RuntimeError, match='initialization'):
        MultiGPUSession([0])


def test_vectorized_flush_streaming(cp):
    """A lens spanning several launches must retain all source ranges."""
    lenses, sources = catalogs()
    bins = np.array([0, .01, .3, 1, 4, 10])*u.deg
    options = dict(nside=8, cross=True, backend='gpu')
    expected = precompute(lenses.copy(), sources, bins, **options)
    with GPUSession(source_chunk=1, lens_chunk=3) as session:
        session.budget = 10000
        launches = []
        real_kernel = session.kernel

        def record(grid, block, arguments):
            launches.append(cp.asnumpy(arguments[5]))
            return real_kernel(grid, block, arguments)

        session.kernel = record
        actual = precompute(lenses.copy(), sources, bins, session=session, **options)
        assert session.source is None
        assert len(launches) > len(lenses)
        # The same lens must occur on both sides of at least one flush.
        assert any(a[-1, 0] == b[0, 0] for a, b in zip(launches, launches[1:]))
    for key in expected.colnames:
        if key.startswith('sum '):
            if key == 'sum 1':
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                close(actual[key], expected[key], rtol=1e-10)
