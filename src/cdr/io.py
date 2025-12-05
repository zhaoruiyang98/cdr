"""
A standalone module to handle file loading and saving for MPI use cases.
"""

from __future__ import annotations
import numpy as np
from collections.abc import MutableMapping
from mpi4py import MPI
from typing import Any


def mpi_load_fits_table(path, columns: list[str] | None = None, ext=None, comm: MPI.Intracomm | None = None):
    import fitsio

    comm = MPI.COMM_WORLD if comm is None else comm
    # try to find the first Table HDU to read if not specified
    with fitsio.FITS(path) as ff:
        if ext is None:
            for i, hdu in enumerate(ff):
                if hdu.has_data():
                    ext = i
                    break
            if ext is None:
                raise ValueError(f"input fits file {path!r} has not binary table to read")
        else:
            if isinstance(ext, str):
                if ext not in ff:
                    raise ValueError(f"FITS file does not contain extension with name {ext!r}")
            elif ext >= len(ff):
                raise ValueError(f"FITS extension {ext} is not valid")

        # make sure we crash if data is wrong or missing
        if not ff[ext].has_data() or ff[ext].get_exttype() == "IMAGE_HDU":
            raise ValueError(f"FITS extension {ext} is not a readable binary table")

    with fitsio.FITS(path) as ff:
        size = ff[ext].get_nrows()
    q, r = divmod(size, comm.size)
    lstart = q * comm.rank + (comm.rank if comm.rank < r else r)
    lend = lstart + q + (comm.rank < r)
    data = fitsio.read(path, ext=ext, columns=columns, rows=range(lstart, lend))
    data = {k: data[k] for k in data.dtype.names}
    return MPITable(data, comm=comm)


def load_quijote_snapshot_header(filename):
    import h5py
    import hdf5plugin  # noqa: F401

    header: dict[str, Any] = {}
    with h5py.File(filename, "r") as f:
        f: Any
        header["time"] = f["Header"].attrs["Time"]
        header["redshift"] = f["Header"].attrs["Redshift"]
        header["npart"] = (f["Header"].attrs["NumPart_ThisFile"]).astype(np.int64)
        header["nall"] = (f["Header"].attrs["NumPart_Total"]).astype(np.int64)  # total number of particles
        try:
            header["nhigh"] = (f["Header"].attrs["NumPart_Total_HighWord"]).astype(np.int64)
        except KeyError:
            pass
        else:
            header["nall"] += header["nhigh"] << np.int64(32)
        header["filenum"] = int(f["Header"].attrs["NumFilesPerSnapshot"])
        header["massarr"] = f["Header"].attrs["MassTable"] * 1e10  # Masses of the particles in Msun/h
        header["boxsize"] = f["Header"].attrs["BoxSize"] / 1e3  # Mpc/h
        # check if it is a SWIFT snapshot
        if "/Cosmology" in f.keys():
            header["Omega0_m"] = f["Cosmology"].attrs["Omega_m"]
            header["Omega0_Lambda"] = f["Cosmology"].attrs["Omega_lambda"]
            header["h"] = f["Cosmology"].attrs["h"]
        # check if it is a Gadget-4 snapshot
        elif "/Parameters" in f.keys():
            header["Omega0_m"] = f["Parameters"].attrs["Omega0"]
            header["Omega0_Lambda"] = f["Parameters"].attrs["OmegaLambda"]
            header["h"] = f["Parameters"].attrs["HubbleParam"]
            # header['cooling']  = f['Parameters'].attrs[u'Flag_Cooling']
        # if it is a traditional Gadget-1/2/3 snapshot
        else:
            header["Omega0_m"] = f["Header"].attrs["Omega0"]
            header["Omega0_Lambda"] = f["Header"].attrs["OmegaLambda"]
            header["h"] = f["Header"].attrs["HubbleParam"]
            # header['cooling']  = f['Header'].attrs[u'Flag_Cooling']
    return header


def mpi_slice(n, comm: MPI.Intracomm | None = None):
    comm = MPI.COMM_WORLD if comm is None else comm
    rank = comm.rank
    size = comm.size
    q, r = divmod(n, size)
    nlocal = q + (rank < r)
    istart = q * rank + (rank if rank < r else r)
    return slice(istart, istart + nlocal)


def mpi_slice_size(n, comm: MPI.Intracomm | None = None):
    sl = mpi_slice(n, comm)
    return sl.stop - sl.start


def mpi_read_quijote_field(filename, block: str, ptype: int, comm: MPI.Intracomm | None = None):
    import h5py
    import hdf5plugin  # noqa: F401

    header = load_quijote_snapshot_header(filename)
    mass = header["massarr"][ptype]
    npart = header["npart"][ptype]
    sl = mpi_slice(npart, comm)

    prefix = f"PartType{ptype}/"
    with h5py.File(filename, "r") as f:
        f: Any
        if block == "POS":
            suffix = "Coordinates"
        elif block == "MASS":
            suffix = "Masses"
        elif block == "ID":
            suffix = "ParticleIDs"
        elif block == "VEL":
            suffix = "Velocities"
        else:
            raise NotImplementedError(f"block {block} not implemented!")

        if f"{prefix}{suffix}" not in f.keys():
            if mass != 0.0:
                array = np.ones(sl.stop - sl.start, np.float32) * mass
            else:
                raise RuntimeError(f"Problem reading the block {block}")
        else:
            array = f[prefix + suffix][sl]
    if block == "VEL":
        array *= np.sqrt(header["time"])
    if block == "POS" and array.dtype == np.float64:
        array = array.astype(np.float32)
    # convert POS to Mpc/h unit, following https://quijote-simulations.readthedocs.io/en/latest/snapshots.html
    if block == "POS":
        array = array / 1e3
    # IDs starting from zero
    if block == "ID":
        array = array - 1
    return array


def mpi_load_quijote_snapshot_table(path, columns=["POS", "VEL"], ptypes=["cdm"], comm: MPI.Intracomm | None = None):
    # adapted from pylians3: https://github.com/franciscovillaescusa/Pylians3
    from pathlib import Path

    path = Path(path)
    ptypes = [ptype if isinstance(ptype, int) else {"cdm": 1, "neutrino": 2}[ptype.lower()] for ptype in ptypes]
    comm = MPI.COMM_WORLD if comm is None else comm

    # read header from the first file
    if path.with_suffix(".hdf5").exists():
        firstfile = path.with_suffix(".hdf5")
    elif path.with_suffix(".0.hdf5").exists():
        firstfile = path.with_suffix(".0.hdf5")
    else:
        raise FileNotFoundError(f"Could not find Quijote snapshot file at {path} with .hdf5 or .0.hdf5 suffix")
    attrs = load_quijote_snapshot_header(firstfile)
    attrs.pop("npart")
    single_file = attrs["filenum"] == 1

    ntotal = 0
    for i in range(attrs["filenum"]):
        if single_file:
            filename = path.with_suffix(".hdf5")
        else:
            filename = path.with_suffix(f".{i}.hdf5")
        ntotal += sum(mpi_slice_size(load_quijote_snapshot_header(filename)["npart"][ptype]) for ptype in ptypes)

    data = {}
    for column in columns:
        if column == "POS":
            dtype = np.dtype((np.float32, 3))
        elif column == "VEL":
            dtype = np.dtype((np.float32, 3))
        elif column == "MASS":
            dtype = np.float32
        elif column == "ID":
            dtype = mpi_read_quijote_field(firstfile, column, ptypes[0], comm=comm).dtype
        else:
            raise NotImplementedError(f"column {column} not found")
        buffer = np.empty(ntotal, dtype=dtype)
        offset = 0
        for ptype in ptypes:
            for i in range(attrs["filenum"]):
                if single_file:
                    filename = path.with_suffix(".hdf5")
                else:
                    filename = path.with_suffix(f".{i}.hdf5")
                field = mpi_read_quijote_field(filename, column, ptype, comm=comm)
                npart = field.shape[0]
                buffer[offset : offset + npart] = field
                offset += npart
        data[column] = buffer

    return MPITable(data, attrs=attrs, comm=comm)


def _is_bool_mask(x):
    return isinstance(x, np.ndarray) and x.dtype == bool and x.ndim == 1


class MPITable(MutableMapping):
    def __init__(self, data: dict[str, Any], attrs: dict[str, Any] | None = None, comm: MPI.Intracomm | None = None):
        self.comm = MPI.COMM_WORLD if comm is None else comm

        self.attrs = {} if attrs is None else attrs
        self._data = {k: np.empty((0,)) if v is None else np.atleast_1d(v) for k, v in data.items()}
        colsize = max(v.shape[0] for v in self.values())
        if colsize > 1:
            # broadcast
            self._data = {k: np.full((colsize,) + v.shape[1:], v[0]) if v.shape[0] == 1 else v for k, v in self.items()}
        if any(v.shape[0] != colsize for v in self.values()):
            msg = str({k: v.shape[0] for k, v in self.items()})
            raise ValueError(f"column sizes do not match: {msg}")

        self._size = colsize
        self._csize = self.comm.allreduce(self._size)
        if len(set(self.comm.allgather(self.columns))) != 1:
            msg = f"rank-{self.comm.rank}: {self.columns}"
            msg = self.comm.allgather(msg)
            msg = ", ".join(msg)
            raise RuntimeError(f"column names do not match: {msg}")

    @property
    def size(self):
        return self._size

    @property
    def csize(self):
        return self._csize

    @property
    def columns(self):
        return tuple(self.keys())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(columns={self.columns}, csize={self.csize})"

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._data[key]
        if _is_bool_mask(key):
            data = {k: v[key] for k, v in self.items()}
            return MPITable(data, comm=self.comm)
        if isinstance(key, list):
            data = {k: v for k, v in self.items() if k in key}
            return MPITable(data, comm=self.comm)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            all_equal = len(set(self.comm.allgather(key))) == 1
            if not all_equal:
                raise KeyError(f"{type(self).__name__}.__setitem__ requires the same key among all mpi processors")
            value = np.atleast_1d(value)
            if value.shape[0] == 1:
                value = np.full((self.size,) + value.shape[1:], value[0])
            elif value.shape[0] != self.size:
                raise ValueError(f"expected size={self.size}, got {value.shape[0]}")
            else:
                # deal with self assignment in simple cases: self['X'] = self['x']
                if id(value) in {id(_) for _ in self.values()}:
                    value = value.copy()
            self._data[key] = value
            return
        raise KeyError(key)

    def __delitem__(self, key) -> None:
        if not isinstance(key, str):
            raise KeyError(key)
        all_equal = len(set(self.comm.allgather(key))) == 1
        if not all_equal:
            raise KeyError(f"{type(self).__name__}.__delitem__ requires the same key among all mpi processors")
        del self._data[key]
