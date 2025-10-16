import numpy as np
import pytest
from cdr.io import mpi_load_fits_table, MPITable
from pytest_mpi import parallel_assert

seed = 114514


def mk_data(size, boxsize=1.0):
    rng = np.random.default_rng(seed=seed)
    boxsize = np.full(3, boxsize)
    return {"position": rng.random((size, 3)) * boxsize, "weight": rng.random(size)}


def test_init_table():
    data = mk_data(10)
    table = MPITable(data)
    np.testing.assert_array_almost_equal_nulp(data["position"], table["position"])
    np.testing.assert_array_almost_equal_nulp(data["weight"], table["weight"])


def test_init_empty_table():
    table = MPITable({"position": [], "weight": None})
    assert table.size == 0
    assert table["position"].shape == (0,)
    assert table["weight"].shape == (0,)


def test_init_broadcasting():
    table = MPITable({"position": np.ones((10, 3)), "weight": 1.0})
    np.testing.assert_array_almost_equal_nulp(table["weight"], np.ones(10))


def test_columns():
    table = MPITable(mk_data(10) | {"nbar": 1.0})
    assert table.columns == ("position", "weight", "nbar")


def test_raise_if_column_size_not_match():
    with pytest.raises(ValueError, match="column sizes do not match"):
        MPITable({"position": np.ones((10, 3)), "weight": np.ones(3)})


@pytest.mark.parallel(2)
def test_raise_if_column_names_not_match(comm):
    local_data = mk_data(10)
    if comm.rank == 0:
        local_data = local_data | {"nbar": np.ones(10)}
    else:
        local_data = local_data | {"z": np.ones(10)}
    with pytest.raises(RuntimeError, match="column names do not match"):
        MPITable(local_data)


@pytest.mark.parallel([1, 2, 3, 4])
def test_table_size(comm):
    local_size = 10 + comm.rank
    local_data = mk_data(local_size)
    table = MPITable(local_data, comm=comm)
    parallel_assert(table.size == local_size)
    parallel_assert(table.csize == comm.allreduce(local_size))


def test_getitem():
    data = mk_data(10) | {"nbar": 1.0, "z": 2.0}
    table = MPITable(data)
    for k, v in data.items():
        np.testing.assert_array_almost_equal_nulp(table[k], v)
    with pytest.raises(KeyError):
        table["not-existed"]


def test_bool_getitem():
    data = mk_data(10)
    mask = data["position"][:, -1] < 0.5
    table = MPITable(data)[mask]
    for k, v in data.items():
        np.testing.assert_array_almost_equal_nulp(table[k], v[mask])


def test_strings_getitem():
    data = mk_data(10) | dict(nbar=1, z=2)
    table = MPITable(data)[["position", "weight"]]
    assert table.columns == ("position", "weight")


def test_setitem():
    table = MPITable(mk_data(10))
    table["nbar"] = 2.0
    assert table.columns == ("position", "weight", "nbar")
    with pytest.raises(ValueError):
        table["nbar"] = np.zeros(9)


def test_self_setitem():
    table = MPITable(mk_data(10))
    table["Position"] = table["position"]
    assert table["Position"] is not table["position"]


@pytest.mark.parallel(2)
def test_mpi_setitem(comm):
    local_size = 10 + comm.rank
    local_data = mk_data(local_size)
    table = MPITable(local_data, comm=comm)
    table["nbar"] = np.zeros(local_size)
    with pytest.raises(KeyError):
        if comm.rank == 0:
            table["z0"] = 1.0
        else:
            table["z1"] = 2.0
    with pytest.raises(ValueError):
        table["id"] = np.zeros(local_size - 1)


def test_delitem():
    table = MPITable(mk_data(10) | dict(nbar=1.0))
    del table["nbar"]
    assert table.columns == ("position", "weight")


@pytest.mark.parallel(2)
def test_mpi_delitem(comm):
    local_size = 10 + comm.rank
    local_data = mk_data(local_size) | dict(z0=1, z1=1)
    table = MPITable(local_data, comm=comm)
    with pytest.raises(KeyError):
        if comm.rank == 0:
            del table["z0"]
        else:
            del table["z1"]


@pytest.mark.parallel(4)
def test_mpi_load_fits(comm, tmp_path):
    import fitsio

    if comm.rank == 0:
        data = np.zeros(10, dtype=[("x", "i8")])
        data["x"] = np.arange(10)
        file = tmp_path / "data.fits"
        fitsio.write(file, data)
    else:
        file = None
    file = comm.bcast(file)

    cat = mpi_load_fits_table(file)
    sizelist = comm.allgather(cat.size)
    data = np.hstack(comm.allgather(cat["x"]))
    parallel_assert(sizelist == [3, 3, 2, 2])
    np.testing.assert_array_almost_equal_nulp(data, np.arange(10))
