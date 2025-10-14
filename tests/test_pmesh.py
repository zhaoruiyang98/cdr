"""
Make sure I understand how pmesh works.
"""

import pytest
import numpy as np
from pmesh.pm import ParticleMesh

seed = 114514


def cic_kernel(s):
    """
    Parameters
    ----------
    s : float
        relative shift in grid unit
    """
    s = np.abs(s)
    if s < 1.0:
        return 1.0 - np.abs(s)
    return 0.0


def test_paint():
    pm = ParticleMesh([3, 3, 3], BoxSize=9.0, resampler="cic")

    # normal paint
    x, y, z = 3.3, 3.0, 3.0
    expected = np.zeros((3, 3, 3))
    expected[1, 1, 1] = cic_kernel(x / 3.0 - 1)
    expected[2, 1, 1] = cic_kernel(x / 3.0 - 2)
    field = pm.paint([[x, y, z]])
    np.testing.assert_array_almost_equal_nulp(field, expected, nulp=100)

    # period
    x, y, z = 6.3, 3.0, 3.0
    expected = np.zeros((3, 3, 3))
    expected[0, 1, 1] = cic_kernel(x / 3 - 3)
    expected[2, 1, 1] = cic_kernel(x / 3 - 2)
    field = pm.paint([[x, y, z]])
    np.testing.assert_array_almost_equal_nulp(field, expected, nulp=100)


@pytest.mark.parallel(4)
def test_mpi_paint():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm_self = MPI.COMM_SELF

    rng = np.random.default_rng(seed=seed)
    pos = rng.random((10000, 3))

    pm = ParticleMesh([10, 10, 10], comm=comm_self, resampler="tsc")
    expected_local = pm.paint(pos)
    expected = expected_local.value.copy()
    comm.Allreduce(MPI.IN_PLACE, expected)

    pm = ParticleMesh([10, 10, 10], comm=comm, resampler="tsc")
    layout = pm.decompose(pos)
    rfield = pm.paint(pos, layout=layout)
    rfield = rfield.preview()

    np.testing.assert_allclose(rfield, expected, rtol=1e-12, atol=1e-14)


def test_paint_with_boxcenter():
    boxcenter = 10.0
    boxsize = 9.0
    pm = ParticleMesh([3, 3, 3], BoxSize=boxsize, resampler="cic")
    x, y, z = np.array([8.8, 8.5, 8.5])
    expected = np.zeros((3, 3, 3))
    expected[1, 1, 1] = cic_kernel(3.3 / 3.0 - 1)
    expected[2, 1, 1] = cic_kernel(3.3 / 3.0 - 2)

    offset = (boxsize / 2 - boxcenter) * pm.affine.scale
    transform = pm.affine.shift(offset)
    field = pm.paint([[x, y, z]], transform=transform)
    np.testing.assert_allclose(field, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parallel(4)
def test_mpi_paint_with_boxcenter():
    from mpi4py import MPI
    from pmesh.pm import FindResampler

    comm = MPI.COMM_WORLD
    comm_self = MPI.COMM_SELF

    rng = np.random.default_rng(seed=seed)
    boxcenter = 2.3
    boxsize = 1.0
    pos = rng.random((10000, 3)) + (boxcenter - boxsize / 2)

    pm = ParticleMesh([10, 10, 10], BoxSize=boxsize, comm=comm_self, resampler="tsc")
    expected_local = pm.paint(pos - (boxcenter - boxsize / 2))
    expected = expected_local.value.copy()
    comm.Allreduce(MPI.IN_PLACE, expected)

    pm = ParticleMesh([10, 10, 10], BoxSize=boxsize, comm=comm, resampler="tsc")
    offset = (boxsize / 2 - boxcenter) * pm.affine.scale
    layout = pm.domain.decompose(
        pos, smoothing=FindResampler(pm.resampler).support * 0.5, transform=lambda x: pm.affine.scale * x + offset
    )
    rfield = pm.paint(pos, layout=layout, transform=pm.affine.shift(offset))
    rfield = rfield.preview()

    np.testing.assert_allclose(rfield, expected, rtol=1e-12, atol=1e-14)


def test_r2c():
    rng = np.random.default_rng(seed=seed)
    pm = ParticleMesh([10, 10, 10], BoxSize=100.0)
    rfield = pm.create("real")
    rfield[...] = rng.random(rfield.shape) * pm.BoxSize[0]
    cfield = rfield.r2c()  # type: ignore
    expected = np.fft.rfftn(rfield.value, norm="forward")

    np.testing.assert_allclose(cfield, expected, rtol=1e-12, atol=1e-14)


def test_c2r():
    rng = np.random.default_rng(seed=seed)
    pm = ParticleMesh([10, 10, 10], BoxSize=100.0)
    cfield = pm.create("complex")
    cfield[...] = (rng.random(cfield.shape) + 1j * rng.random(cfield.shape)) * pm.BoxSize[0]
    rfield = cfield.c2r()  # type: ignore
    expected = np.fft.irfftn(cfield.value, norm="forward")

    np.testing.assert_allclose(rfield, expected, rtol=1e-12, atol=1e-14)
