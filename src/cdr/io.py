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


def is_bool_mask(x):
    return isinstance(x, np.ndarray) and x.dtype == bool and x.ndim == 1


class MPITable(MutableMapping):
    def __init__(self, data: dict[str, Any], comm: MPI.Intracomm | None = None):
        self.comm = MPI.COMM_WORLD if comm is None else comm

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
        if is_bool_mask(key):
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
