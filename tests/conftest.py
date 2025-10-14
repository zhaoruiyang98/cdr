import pytest


@pytest.fixture
def comm():
    from mpi4py import MPI

    yield MPI.COMM_WORLD
