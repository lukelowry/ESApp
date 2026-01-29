"""
Binary 3D (B3D) file format handler for electric field data.

The B3D format stores time-varying electric field data (Ex, Ey) at
geographic locations. It is used by PowerWorld for GIC electric field
input. This module supports version 4 with variable location points
and variable time points.
"""

from __future__ import annotations

import numpy as np
from numpy import array, zeros, frombuffer, stack, meshgrid, linspace
from numpy import single, uint32, double

__all__ = ['B3D']

_B3D_CODE = 34280
_B3D_VERSION = 4


class B3D:
    """
    Handler for the B3D (Binary 3D) electric field file format.

    Supports reading, writing, and constructing B3D files containing
    time-varying electric field vectors at geographic locations.

    Parameters
    ----------
    fname : str, optional
        Path to a B3D file to load on initialization.

    Attributes
    ----------
    comment : str
        Metadata comment string.
    time_0 : int
        Reference time origin.
    time_units : int
        Time unit code (0 = milliseconds).
    lat, lon : np.ndarray
        1D arrays of geographic coordinates (float64).
    grid_dim : list of int
        Grid dimensions [nx, ny].
    time : np.ndarray
        1D array of time points (uint32).
    ex, ey : np.ndarray
        2D arrays of electric field components, shape (nt, n), dtype float32.
    """

    def __init__(self, fname: str | None = None) -> None:
        self.comment = "Default 2x2 grid with 3 time points"
        self.time_0 = 0
        self.time_units = 0

        self.lat = array([30.5, 30.5, 31.0, 31.0])
        self.lon = array([-84.5, -85.0, -84.5, -85.0])
        self.grid_dim = [2, 2]

        self.time = array([0, 1000, 2000], dtype=uint32)

        self.ex = zeros([3, 4], dtype=single)
        self.ey = zeros([3, 4], dtype=single)

        if fname is not None:
            self.load_b3d_file(fname)

    @classmethod
    def from_mesh(
        cls,
        long: np.ndarray,
        lat: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
        times: np.ndarray | None = None,
        comment: str = "GWB Electric Field Data",
    ) -> B3D:
        """
        Construct a B3D from mesh-grid style electric field data.

        Currently supports static (single time step) fields only.

        Parameters
        ----------
        long : np.ndarray
            Array of longitudes, shape (n,).
        lat : np.ndarray
            Array of latitudes, shape (m,).
        ex : np.ndarray
            X-component electric field, shape (n, m).
        ey : np.ndarray
            Y-component electric field, shape (n, m).
        times : np.ndarray, optional
            Time points. Currently unused.
        comment : str, default "GWB Electric Field Data"
            Metadata comment string.

        Returns
        -------
        B3D
            Initialized B3D object.
        """
        b3d = cls()
        b3d.comment = comment

        n = len(long)
        m = len(lat)
        nt = n * m
        X, Y = meshgrid(long, lat)
        b3d.lon = X.reshape(nt, order='F')
        b3d.lat = Y.reshape(nt, order='F')
        b3d.grid_dim = [n, m]

        b3d.time = linspace(0, 10, 1, dtype=uint32)

        eshape = (1, nt)
        b3d.ex = ex.reshape(eshape, order='F').astype(single)
        b3d.ey = ey.reshape(eshape, order='F').astype(single)

        return b3d

    def write_b3d_file(self, fname: str) -> None:
        """
        Write the B3D data to a binary file.

        Parameters
        ----------
        fname : str
            Output file path.

        Raises
        ------
        ValueError
            If data arrays have inconsistent shapes or incorrect dtypes.
        """
        with open(fname, "wb") as f:
            n = self.lat.shape[0]
            nt = self.time.shape[0]

            if self.lon.shape[0] != n:
                raise ValueError("lat and lon must have the same length")
            if self.lat.dtype != double:
                raise ValueError("lat must be a float64 (double) array")
            if self.lon.dtype != double:
                raise ValueError("lon must be a float64 (double) array")
            if self.time.dtype != uint32:
                raise ValueError("time must be a uint32 array")
            if self.ex.dtype != single:
                raise ValueError("ex must be a float32 (single) array")
            if self.ey.dtype != single:
                raise ValueError("ey must be a float32 (single) array")
            if self.ex.shape[1] != n:
                raise ValueError(f"ex columns ({self.ex.shape[1]}) must match location count ({n})")
            if self.ey.shape[1] != n:
                raise ValueError(f"ey columns ({self.ey.shape[1]}) must match location count ({n})")
            if self.ex.shape[0] != nt:
                raise ValueError(f"ex rows ({self.ex.shape[0]}) must match time count ({nt})")
            if self.ey.shape[0] != nt:
                raise ValueError(f"ey rows ({self.ey.shape[0]}) must match time count ({nt})")

            def _write_int(val: int) -> None:
                f.write(val.to_bytes(4, byteorder="little"))

            _write_int(_B3D_CODE)
            _write_int(_B3D_VERSION)
            _write_int(2)  # Two meta strings
            meta = self.comment + "\0" + str(self.grid_dim) + "\0"
            f.write(meta.encode('ascii'))
            _write_int(2)  # 2 float channels (ex, ey)
            _write_int(0)  # 0 byte channels
            _write_int(1)  # Variable location format
            _write_int(n)

            loc0 = zeros(n, dtype=double)
            loc_data = stack([self.lon, self.lat, loc0]).transpose().reshape(1, n * 3).tobytes()
            f.write(loc_data)

            _write_int(self.time_0)
            _write_int(self.time_units)
            _write_int(0)   # Time offset (not supported)
            _write_int(0)   # Time step (variable)
            _write_int(nt)
            f.write(self.time.tobytes())

            exd = self.ex.reshape(n * nt)
            eyd = self.ey.reshape(n * nt)
            f.write(stack([exd, eyd]).transpose().reshape(n * nt * 2).tobytes())

    def load_b3d_file(self, fname: str) -> None:
        """
        Load a B3D binary file into this object.

        Parameters
        ----------
        fname : str
            Path to the B3D file.

        Raises
        ------
        IOError
            If the file is not a valid B3D file or uses an unsupported format.
        """
        with open(fname, "rb") as f:
            b = f.read()

        code = int.from_bytes(b[0:4], "little")
        if code != _B3D_CODE:
            raise IOError(f"Invalid B3D file (code {code}, expected {_B3D_CODE})")

        version = int.from_bytes(b[4:8], "little")
        if version != _B3D_VERSION:
            raise IOError(f"Unsupported B3D version {version} (expected {_B3D_VERSION})")

        nmeta = int.from_bytes(b[8:12], "little")
        self.grid_dim = [0, 0]
        x1 = x2 = 12
        meta_strings = []
        for _ in range(nmeta):
            while b[x2] != 0:
                x2 += 1
            meta_strings.append(b[x1:x2].decode("ascii"))
            x2 += 1
            x1 = x2

        if nmeta <= 0:
            self.comment = "No comment"
        else:
            self.comment = meta_strings[0]
            if nmeta >= 2:
                try:
                    dim_text = meta_strings[1].strip("[]")
                    if "," in dim_text:
                        self.grid_dim = [int(x) for x in dim_text.split(',')]
                    else:
                        self.grid_dim = [int(x) for x in dim_text.split()]
                    if len(self.grid_dim) != 2:
                        raise ValueError("grid_dim must have exactly 2 elements")
                except (ValueError, IndexError):
                    self.grid_dim = [0, 0]

        float_channels = int.from_bytes(b[x2:x2+4], "little")
        byte_channels = int.from_bytes(b[x2+4:x2+8], "little")
        loc_format = int.from_bytes(b[x2+8:x2+12], "little")

        if float_channels < 2:
            raise IOError("Only B3D files with at least 2 float channels are supported")
        if loc_format != 1:
            raise IOError(f"Only location format 1 is supported (got {loc_format})")

        n = int.from_bytes(b[x2+12:x2+16], "little")
        if self.grid_dim[0] * self.grid_dim[1] != n:
            self.grid_dim = [n, 1]

        x3 = x2 + 16 + 3 * 8 * n
        loc_data = frombuffer(b[x2+16:x3], dtype=double).reshape([n, 3]).copy()
        self.lon = loc_data[:, 0]
        self.lat = loc_data[:, 1]

        self.time_0 = int.from_bytes(b[x3:x3+4], "little")
        self.time_units = int.from_bytes(b[x3+4:x3+8], "little")
        self.time_offset = int.from_bytes(b[x3+8:x3+12], "little")
        time_step = int.from_bytes(b[x3+12:x3+16], "little")
        nt = int.from_bytes(b[x3+16:x3+20], "little")

        if time_step != 0:
            raise IOError("Only B3D files with variable time points are supported")

        x4 = x3 + 20 + 4 * nt
        self.time = frombuffer(b[x3+20:x4], dtype=uint32).copy()
        npts = n * nt

        if float_channels == 2 and byte_channels == 0:
            x5 = x4 + 4 * 2 * n * nt
            raw_exy = frombuffer(b[x4:x5], dtype=single)
        else:
            bxy = bytearray(npts * 8)
            for i in range(npts):
                x5 = x4 + i * (float_channels * 4 + byte_channels)
                bxy[i * 8:(i + 1) * 8] = b[x5:x5 + 8]
            raw_exy = frombuffer(bxy, dtype=single)

        edata = raw_exy.reshape([nt, n, 2]).copy()
        self.ex = edata[:, :, 0]
        self.ey = edata[:, :, 1]
