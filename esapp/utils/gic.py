"""
GIC (Geomagnetically Induced Currents) Analysis Module
======================================================

This module provides tools for analyzing geomagnetically induced currents
in power systems, including matrix generation, sensitivity analysis, and
integration with PowerWorld's GIC calculation engine.

Classes
-------
GIC
    Main application class providing GIC analysis methods, model generation,
    and PowerWorld integration via the Indexable interface.

Key Features
------------
- Direct G-matrix retrieval from PowerWorld via SAW
- Linear GIC model generation from network topology
- Interface sensitivity analysis (dBound/dI)
- E-field to GIC Jacobian computation
- Time-varying GIC series support

Example
-------
Basic GIC model generation::

    >>> from esapp import GridWorkBench
    >>> wb = GridWorkBench("case.pwb")
    >>> wb.gic.model()
    >>> H = wb.gic.H  # Linear GIC function matrix
    >>> G = wb.gic.G  # Conductance Laplacian

Using PowerWorld's G-matrix directly::

    >>> G_pw = wb.gic.gmatrix_from_powerworld()

See Also
--------
esapp.saw.gic : Low-level GIC SAW functions.
esapp.saw.matrices : Matrix retrieval functions including get_gmatrix().
"""

from typing import Union, Optional

import numpy as np
from pandas import DataFrame, read_csv
from scipy.sparse import csr_matrix, eye as speye, hstack, vstack, diags
from scipy.sparse.linalg import inv as sinv

from esapp.saw._enums import YesNo

from ..components import GIC_Options_Value, GICInputVoltObject
from ..components import Branch, Substation, Bus, Gen, GICXFormer
from ..indexable import Indexable

__all__ = ['GIC', 'jac_decomp']


def jac_decomp(jac):
    """
    Decompose a power flow Jacobian into sub-matrices.

    Parameters
    ----------
    jac : np.ndarray
        Full Jacobian matrix of shape (2n, 2n).

    Yields
    ------
    np.ndarray
        Sub-matrices in order: dP/dTheta, dP/dV, dQ/dTheta, dQ/dV.
    """
    dim = jac.shape[0]
    nbus = int(dim / 2)

    yield jac[:nbus, :nbus]   # dP/dTheta
    yield jac[:nbus, nbus:]   # dP/dV
    yield jac[nbus:, :nbus]   # dQ/dTheta
    yield jac[nbus:, nbus:]   # dQ/dV


class GIC(Indexable):
    """
    GIC analysis application for PowerWorld integration.

    Provides methods for GIC calculations, sensitivity analysis, and
    model generation using PowerWorld case data via the Indexable interface.

    This class is accessed via ``GridWorkBench.gic``.

    GIC Options
    -----------
    GIC analysis requires certain options to be enabled for full functionality.
    The most important is ``set_pf_include(True)`` which must be called before
    retrieving GIC data like transformer coil resistances (GICCoilR fields).
    Methods like ``model()`` and ``gmatrix_from_powerworld()`` automatically
    enable this option. Use ``configure()`` to set multiple options at once.

    Example
    -------
    >>> wb = GridWorkBench("case.pwb")
    >>> wb.gic.configure()  # Enable GIC with default options
    >>> wb.gic.storm(100, 90)  # 100 V/km, 90 degrees
    >>> wb.gic.model()
    >>> G = wb.gic.gmatrix_from_powerworld()

    See Also
    --------
    configure : Set GIC options with sensible defaults.
    settings : View or modify all GIC settings.
    """

    # --- GIC Options Configuration ---

    def _set_gic_option(self, option: str, value) -> None:
        """
        Internal helper to set a GIC option.

        Parameters
        ----------
        option : str
            The GIC option name (e.g., 'IncludeInPowerFlow', 'CalcMode').
        value : str or bool
            The value to set. Booleans are converted to 'YES'/'NO'.
        """
        if isinstance(value, bool):
            value = YesNo.from_bool(value)

        self.esa.EnterMode("EDIT")
        self.esa.SetData(
            'GIC_Options_Value',
            ['VariableName', 'ValueField'],
            [option, value]
        )
        self.esa.EnterMode("RUN")

    def get_gic_option(self, option: str) -> str:
        """
        Get the current value of a GIC option.

        Parameters
        ----------
        option : str
            The GIC option name (e.g., 'IncludeInPowerFlow', 'CalcMode').

        Returns
        -------
        str
            The current value of the option.
        """
        settings = self.settings()
        if settings is None:
            return None
        row = settings[settings['VariableName'] == option]
        if row.empty:
            return None
        return row['ValueField'].iloc[0]

    def configure(
        self,
        pf_include: bool = True,
        ts_include: bool = False,
        calc_mode: str = 'SnapShot'
    ) -> None:
        """
        Configure GIC options with sensible defaults.

        This is the recommended way to initialize GIC analysis. It ensures
        all necessary options are set for typical GIC workflows.

        Parameters
        ----------
        pf_include : bool, default True
            Include GIC effects in power flow calculations. Required for
            accessing GIC-related data like transformer coil resistances.
        ts_include : bool, default False
            Include GIC effects in transient stability simulations.
        calc_mode : str, default 'SnapShot'
            GIC calculation mode. Options:
            - 'SnapShot': Single time point calculation
            - 'TimeVarying': Time series from uniform field
            - 'NonUniformTimeVarying': Time series with spatial variation
            - 'SpatiallyUniformTimeVarying': Spatially uniform time series

        Example
        -------
        >>> wb.gic.configure()  # Use defaults (pf_include=True)
        >>> wb.gic.configure(ts_include=True)  # Enable for transient stability
        >>> wb.gic.configure(calc_mode='TimeVarying')  # For time series analysis
        """
        self.set_pf_include(pf_include)
        self.set_ts_include(ts_include)
        self.set_calc_mode(calc_mode)

    def set_pf_include(self, enable: bool = True) -> None:
        """
        Enable or disable GIC effects in power flow calculations.

        This option MUST be enabled (True) before accessing GIC-related
        data from PowerWorld, including transformer coil resistances
        (GICCoilRFrom, GICCoilRTo) and other GIC fields.

        Parameters
        ----------
        enable : bool, default True
            Whether to include GIC effects in power flow.

        Note
        ----
        Methods like ``model()`` and ``gmatrix_from_powerworld()``
        automatically call this with ``enable=True``.
        """
        self._set_gic_option('IncludeInPowerFlow', enable)

    def set_ts_include(self, enable: bool = True) -> None:
        """
        Enable or disable GIC effects in transient stability simulations.

        Parameters
        ----------
        enable : bool, default True
            Whether to include GIC effects in transient stability.
        """
        self._set_gic_option('IncludeTimeDomain', enable)

    def set_calc_mode(self, mode: str) -> None:
        """
        Set the GIC calculation mode.

        Parameters
        ----------
        mode : str
            One of:
            - 'SnapShot': Single time point calculation
            - 'TimeVarying': Time series from uniform field
            - 'NonUniformTimeVarying': Time series with spatial variation
            - 'SpatiallyUniformTimeVarying': Spatially uniform time series
        """
        self._set_gic_option('CalcMode', mode)

    # --- G-Matrix Retrieval ---

    def gmatrix(self, sparse: bool = True) -> Union[csr_matrix, np.ndarray]:
        """
        Retrieve the G-matrix directly from PowerWorld.

        This is the recommended approach when working with PowerWorld cases,
        as it uses the simulator's internal GIC calculation engine and
        ensures consistency with PowerWorld's results.

        This method automatically enables GIC in power flow (pf_include=True)
        before retrieving the matrix, ensuring GIC data is available.

        Parameters
        ----------
        sparse : bool, default True
            If True, returns scipy sparse CSR matrix.
            If False, returns dense numpy array.

        Returns
        -------
        scipy.sparse.csr_matrix or np.ndarray
            The GIC conductance matrix (G-matrix) from PowerWorld.

        See Also
        --------
        model : Generate full GIC model with H-matrix and per-unit model.
        configure : Set GIC options manually.
        """
        # Ensure GIC is included in power flow before retrieving matrix
        self.set_pf_include(True)
        return self.esa.get_gmatrix(full=not sparse)

    def storm(self, maxfield: float, direction: float, solvepf: bool = True) -> None:
        """
        Configure synthetic storm with uniform electric field.

        Parameters
        ----------
        maxfield : float
            Maximum electric field magnitude (V/km).
        direction : float
            Storm direction in degrees (0-360, 0=North).
        solvepf : bool, default True
            Whether to include GIC results in power flow solution.
        """
        self.esa.GICCalculate(maxfield, direction, solvepf)

    def cleargic(self) -> None:
        """Clear all GIC calculation results from the case."""
        self.esa.RunScriptCommand("GICClear;")

    def loadb3d(self, ftype: str, fname: str, setuponload: bool = True) -> None:
        """
        Load B3D file containing electric field data.

        Parameters
        ----------
        ftype : str
            File type identifier.
        fname : str
            Path to the B3D file.
        setuponload : bool, default True
            Whether to set up time-varying series on load.
        """
        self.esa.GICLoad3DEfield(ftype, fname, setuponload)

    def settings(self, value: Optional[DataFrame] = None) -> Optional[DataFrame]:
        """
        View or modify GIC calculation settings.

        Parameters
        ----------
        value : DataFrame, optional
            If provided, updates settings. If None, returns current settings.

        Returns
        -------
        DataFrame or None
            Current settings if value is None.
        """
        return self.esa.GetParametersMultipleElement(
                GIC_Options_Value.TYPE,
                GIC_Options_Value.fields
        )[['VariableName', 'ValueField']]

    def timevary_csv(self, fpath: str) -> None:
        """
        Upload time-varying series voltage inputs from CSV file.

        Parameters
        ----------
        fpath : str
            Path to CSV file with format::

                Time In Seconds, 1, 2, 3
                Branch '1' '2' '1', 0.1, 0.11, 0.14
                Branch '1' '2' '2', 0.1, 0.11, 0.14
        """
        csv = read_csv(fpath, header=None)
        obj = GICInputVoltObject.TYPE
        fields = ['WhoAmI'] + [f'GICObjectInputDCVolt:{i+1}' for i in range(csv.columns.size - 1)]

        for row in csv.to_records(False):
            values = list(row)
            # Quote the WhoAmI identifier (contains spaces) for PowerWorld
            values[0] = f'"{values[0]}"'
            self.esa.SetData(obj, fields, values)

    # --- Model ---

    def model(self) -> 'GIC':
        """
        Generate GIC model from current PowerWorld case data.

        Extracts substation, bus, line, transformer, and generator data
        from PowerWorld and computes all GIC matrices (incidence, G-matrix,
        H-matrix, per-unit linear model). Results are stored as properties
        on this instance.

        Transformer data is sourced from the ``GICXFormer`` object type,
        which provides the authoritative per-winding configuration, substation
        assignments, and auto-transformer status used by PowerWorld's GIC
        calculation engine.

        This method automatically enables GIC in power flow (pf_include=True)
        before retrieving data.

        Returns
        -------
        GIC
            Self, with computed model matrices accessible via properties
            (``G``, ``H``, ``A``, ``zeta``, ``Px``, ``eff``).

        See Also
        --------
        gmatrix_from_powerworld : Get just the G-matrix from PowerWorld.
        configure : Set GIC options manually.
        """
        self.set_pf_include(True)
        MOHM = 1e6

        # ---- Data from PowerWorld ----
        subs  = self[Substation, ["SubNum", "GICUsedSubGroundOhms", "Longitude", "Latitude"]]
        buses = self[Bus, ["BusNum", "BusNomVolt", "SubNum"]]
        lines = self[Branch, ["BusNum", "BusNum:1", "GICConductance", "BranchDeviceType"]]
        lines = lines.loc[lines['BranchDeviceType'] != 'Transformer',["BusNum", "BusNum:1", "GICConductance"]]
        xf = self[GICXFormer, [
            "BusNum3W", "BusNum3W:1", "SubNum", "SubNum:1",
            "GICXFCoilR1", "GICXFCoilR1:1", "GICXFConfigUsed",
            "GICBlockDevice", "GICAutoXFUsed", "GICXF3Type",
            "GICXFMVABase", "GICModelKUsed",
        ]]
        xf = xf[xf['GICXF3Type'].astype(str).str.upper() != 'YES'].copy()
        gens = (self[Gen, ["BusNum", "GICConductance", "GICGenIncludeImplicitGSU"]]
                .query("GICConductance != 0 and GICGenIncludeImplicitGSU != 'NO'")
                .merge(buses[['BusNum', 'SubNum']], on='BusNum', how='inner'))

        # ---- Transformer high/low winding assignment ----
        cfg = xf['GICXFConfigUsed'].astype(str).str.lower().str.split('-')
        kv = buses.set_index('BusNum')['BusNomVolt']
        fromV, toV = xf['BusNum3W'].map(kv).to_numpy(), xf['BusNum3W:1'].map(kv).to_numpy()

        def _hilo(a, b):
            """Sort paired from/to values into (high-side, low-side) by voltage."""
            return np.where(fromV >= toV, a, b), np.where(fromV >= toV, b, a)

        high_bus, low_bus = _hilo(xf['BusNum3W'],  xf['BusNum3W:1'])
        high_sub, low_sub = _hilo(xf['SubNum'],    xf['SubNum:1'])
        high_cfg, low_cfg = _hilo(cfg.str[0],      cfg.str[-1])
        g_from, g_to = 1.0 / xf['GICXFCoilR1'].replace(0, MOHM), 1.0 / xf['GICXFCoilR1:1'].replace(0, MOHM)
        high_g, low_g     = _hilo(g_from, g_to)
        highV, lowV = np.maximum(fromV, toV), np.maximum(np.minimum(fromV, toV), 1.0)

        HWYE, LWYE = high_cfg == 'gwye', low_cfg == 'gwye'
        BD   = xf['GICBlockDevice'].astype(str).str.upper() == 'YES'
        AUTO = xf['GICAutoXFUsed'].astype(str).str.upper() == 'YES'
        K    = xf['GICModelKUsed']
        MVA  = xf['GICXFMVABase']

        # ---- Index maps & helpers ----
        ns, nb, nx, nl, ng = len(subs), len(buses), len(xf), len(lines), len(gens)
        ncol = ns + nb
        sub_map = {v: i for i, v in enumerate(subs['SubNum'])}
        bus_map = {v: i + ns for i, v in enumerate(buses['BusNum'])}

        def _perm(ids, lookup=bus_map):
            cols = np.array([lookup[v] for v in ids])
            return csr_matrix((np.ones(len(cols)), (np.arange(len(cols)), cols)),
                              shape=(len(cols), ncol))

        def _mask(mat, m):
            return diags(np.asarray(m, dtype=float)) @ mat

        def _g(vals, blocked=None):
            g = np.asarray(vals, dtype=float)
            if blocked is not None:
                g = np.where(blocked, 0.0, g)
            return np.where(g == 0, 1 / MOHM, g)

        # ---- Incidence matrix ----
        SH, SL = _perm(high_sub, sub_map), _perm(low_sub, sub_map)
        BH, BL = _perm(high_bus), _perm(low_bus)

        A = vstack([
            _mask(-SH + BH, HWYE & ~AUTO) + _mask(BH - BL, ~HWYE | AUTO),  # high
            _mask(-SL + BL, LWYE & ~AUTO) + _mask(SL - BL, AUTO),          # low
            _perm(lines['BusNum']) - _perm(lines['BusNum:1']),               # lines
            _perm(gens['SubNum'], sub_map) - _perm(gens['BusNum']),         # GSUs
        ])

        # ---- Conductances ----
        Gd = diags(np.concatenate([
            3 * _g(high_g, BD & HWYE & ~AUTO),
            3 * _g(low_g, BD & (LWYE | AUTO)),
            3 * _g(lines['GICConductance']),
            _g(gens['GICConductance']),
        ]))
        Gs = diags(np.concatenate([
            1 / subs['GICUsedSubGroundOhms'].replace(0, MOHM),
            np.full(nb, 1 / MOHM),
        ]))

        # ---- Core computations ----
        Eff  = hstack([speye(nx), diags(highV / lowV), csr_matrix((nx, nl + ng))])
        Px   = _perm(xf['BusNum3W'])[:, ns:].T
        G    = A.T @ Gd @ A + Gs
        Gi   = sinv(G.tocsc())
        H    = Eff @ (Gd - Gd @ A @ Gi @ A.T @ Gd) / 3
        K    = diags(K * highV / (1e3 * MVA * np.sqrt(2 / 3)))
        zeta = K @ H

        self._A, self._G, self._H = A, G, H
        self._eff, self._zeta, self._Px = Eff, zeta, Px
        return self

    # --- Model Properties ---

    @property
    def A(self):
        """
        General incidence matrix of the GIC network.

        The first N columns are substation neutral buses, and the remaining
        M columns are bus nodes. The first 2X rows are high and low windings,
        and the remaining rows are non-winding branches.

        Returns
        -------
        scipy.sparse matrix
            Shape (branches, N+M).
        """
        return self._A

    @property
    def G(self):
        """
        Conductance Laplacian of the GIC network.

        The first N nodes are substation neutral buses, and the remaining
        M nodes are bus nodes. Computed as: G = A.T @ Gd @ A + Gs

        Returns
        -------
        scipy.sparse matrix
            Shape (N+M, N+M).
        """
        return self._G

    @property
    def H(self):
        """
        Linear GIC function matrix (H-matrix).

        Maps induced line voltages to signed effective transformer GICs.
        Values are in actual current (Amps), not per-unit.

        Returns
        -------
        scipy.sparse matrix
            Shape (nxfmr, nbranches).
        """
        return self._H

    @property
    def zeta(self):
        """
        Per-unit linear GIC model.

        Returns the constant-current load (prior to absolute value) in
        per-unit for each transformer. This is the fastest option for
        modeling GICs in power flow studies.

        Returns
        -------
        scipy.sparse matrix
            Per-unit GIC model matrix.
        """
        return self._zeta

    @property
    def Px(self):
        """
        Bus assignment permutation matrix.

        Maps each transformer to the bus used to model losses
        (default: from-bus).

        Returns
        -------
        scipy.sparse matrix
            Shape (nbus, nxfmr).
        """
        return self._Px

    @property
    def eff(self):
        """
        Effective GIC operator matrix.

        Calculates effective transformer GICs when applied to the vector
        of branch GICs. Includes non-winding branches; trim dimensions
        for faster computation when only line voltages are used.

        Returns
        -------
        scipy.sparse matrix
            Shape (nxfmr, nbranches).
        """
        return self._eff
