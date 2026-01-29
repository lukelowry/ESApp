"""
Network Matrix Utilities
========================

Provides network topology analysis including incidence matrices,
graph Laplacians with various weighting schemes, and branch parameter
calculations for power system analysis.

Classes
-------
Network
    Network matrix construction and branch weight calculations.
BranchType
    Enumeration of supported branch weight types for Laplacian construction.

Key Features
------------
- Sparse incidence matrix construction (with optional HVDC lines)
- Weighted graph Laplacian with multiple weighting schemes
- Branch parameter calculations (impedance, admittance, propagation delay)
- Support for transformer pseudo-lengths based on impedance

Example
-------
Basic network matrix operations::

    >>> from esapp import GridWorkBench
    >>> wb = GridWorkBench("case.pwb")
    >>> A = wb.net.incidence()  # Incidence matrix
    >>> L = wb.net.laplacian(BranchType.LENGTH)  # Length-weighted Laplacian

See Also
--------
esapp.apps.gic : GIC analysis with network topology.
esapp.saw.matrices : Matrix retrieval from PowerWorld.
"""

from enum import Enum
from typing import Union

import numpy as np
from pandas import Series, concat
from scipy.sparse import diags, coo_matrix, csc_matrix

from ..components import Branch, Bus, DCTransmissionLine
from ..indexable import Indexable

__all__ = ['Network', 'BranchType']


class BranchType(Enum):
    """
    Branch weighting schemes for Laplacian construction.

    These weights determine how branches contribute to the graph Laplacian,
    affecting spectral properties and analysis results.

    Attributes
    ----------
    LENGTH : int
        Weight by inverse squared physical length (km^-2).
        Emphasizes short connections in the network topology.
    RES_DIST : int
        Weight by inverse impedance magnitude (resistance distance).
        Reflects electrical distance between nodes.
    DELAY : int
        Weight by inverse squared propagation delay (s^-2).
        Based on effective LC time constants of branches.
    """
    LENGTH = 1
    RES_DIST = 2
    DELAY = 3


class Network(Indexable):
    """
    Network matrix construction and analysis.

    Provides methods for building sparse network matrices (incidence,
    Laplacian) and computing branch electrical parameters. Supports
    both AC branches and optionally HVDC transmission lines.

    Attributes
    ----------
    A : scipy.sparse.csc_matrix or None
        Cached incidence matrix. Recomputed when remake=True.

    Notes
    -----
    Matrix dimensions follow PowerWorld bus ordering. Use busmap()
    to translate between bus numbers and matrix indices.
    """

    A = None

    def busmap(self) -> Series:
        """
        Create mapping from bus numbers to matrix indices.

        Returns
        -------
        pd.Series
            Series indexed by BusNum with positional values.

        Example
        -------
        >>> bmap = wb.net.busmap()
        >>> matrix_idx = bmap[bus_number]
        """
        bus_nums = self[Bus]
        return Series(bus_nums.index, bus_nums["BusNum"])

    def incidence(self, remake: bool = True, hvdc: bool = False) -> csc_matrix:
        """
        Construct the sparse arc-incidence matrix.

        The incidence matrix A has shape (branches, buses) where each
        row represents a branch with +1 at the to-bus and -1 at the
        from-bus.

        Parameters
        ----------
        remake : bool, default True
            If True, recomputes even if cached.
        hvdc : bool, default False
            If True, includes HVDC transmission lines.

        Returns
        -------
        scipy.sparse.csc_matrix
            Sparse incidence matrix (branches x buses).
        """
        if self.A is not None and not remake:
            return self.A

        # Retrieve branch data
        fields = ["BusNum", "BusNum:1"]
        branches = self[Branch][fields]

        if hvdc:
            hvdc_branches = self[DCTransmissionLine, fields][fields]
            branches = concat([branches, hvdc_branches], ignore_index=True)

        # Create bus mapping
        bmap = self.busmap()
        from_bus = branches["BusNum"].map(bmap).to_numpy()
        to_bus = branches["BusNum:1"].map(bmap).to_numpy()

        nbranches = len(branches)
        nbuses = len(bmap)

        # Build sparse matrix using COO format (efficient construction)
        rows = np.concatenate([np.arange(nbranches), np.arange(nbranches)])
        cols = np.concatenate([from_bus, to_bus])
        data = np.concatenate([-np.ones(nbranches), np.ones(nbranches)])

        A = coo_matrix((data, (rows, cols)), shape=(nbranches, nbuses))
        self.A = A.tocsc()

        return self.A

    def laplacian(
        self,
        weights: Union[BranchType, np.ndarray],
        longer_xfmr_lens: bool = True,
        len_thresh: float = 0.01,
        hvdc: bool = False
    ) -> csc_matrix:
        """
        Construct weighted graph Laplacian.

        Computes L = A.T @ W @ A where W is a diagonal weight matrix
        determined by the weighting scheme.

        Parameters
        ----------
        weights : BranchType or np.ndarray
            Weighting scheme or custom weight vector.
        longer_xfmr_lens : bool, default True
            Use impedance-based pseudo-lengths for transformers.
        len_thresh : float, default 0.01
            Threshold (km) below which branches are treated as transformers.
        hvdc : bool, default False
            Include HVDC transmission lines.

        Returns
        -------
        scipy.sparse.csc_matrix
            Sparse weighted Laplacian matrix (buses x buses).
        """
        if weights == BranchType.LENGTH:
            W = 1 / self.lengths(longer_xfmr_lens, len_thresh, hvdc) ** 2
        elif weights == BranchType.RES_DIST:
            W = 1 / self.zmag(hvdc)
        elif weights == BranchType.DELAY:
            W = 1 / self.delay() ** 2
        else:
            W = weights

        A = self.incidence(hvdc=hvdc)
        LAP = A.T @ diags(W) @ A

        return LAP.tocsc()

    def lengths(
        self,
        longer_xfmr_lens: bool = False,
        length_thresh_km: float = 0.01,
        hvdc: bool = False
    ) -> Series:
        """
        Get branch lengths in kilometers.

        Parameters
        ----------
        longer_xfmr_lens : bool, default False
            Calculate pseudo-lengths for transformers based on
            their impedance relative to average line impedance per km.
        length_thresh_km : float, default 0.01
            Branches shorter than this are treated as transformers.
        hvdc : bool, default False
            Include HVDC transmission lines.

        Returns
        -------
        pd.Series
            Branch lengths in kilometers.

        Notes
        -----
        When longer_xfmr_lens=True, transformer pseudo-length is
        computed as: Z_xfmr / (average Z per km of lines).
        """
        field = ["LineLengthByParameters", "LineLengthByParameters:2"]
        ell = self[Branch, field][field]

        # Prefer user-specified length over calculated
        ell_user = ell["LineLengthByParameters"]
        ell.loc[ell_user > 0, "LineLengthByParameters:2"] = ell.loc[ell_user > 0, "LineLengthByParameters"]
        ell = ell["LineLengthByParameters:2"]

        if hvdc:
            hvdc_field = "LineLengthByParameters"
            hvdc_ell = self[DCTransmissionLine, hvdc_field][hvdc_field]
            ell = concat([ell, hvdc_ell], ignore_index=True)

        if longer_xfmr_lens:
            fields = ["LineX:2", "LineR:2"]
            branches = self[Branch, fields][fields]

            is_long_line = ell > length_thresh_km
            lines = branches.loc[is_long_line]
            xfmrs = branches.loc[~is_long_line]

            line_z = np.abs(lines["LineR:2"] + 1j * lines["LineX:2"])
            xfmr_z = np.abs(xfmrs["LineR:2"] + 1j * xfmrs["LineX:2"])

            # Average ohms per km for transmission lines
            z_per_km = (line_z / ell[is_long_line]).mean()

            # Pseudo-length based on impedance
            pseudo_length = (xfmr_z / z_per_km).to_numpy()
            ell.loc[~is_long_line] = pseudo_length
        else:
            # Assume transformers are 10 meters
            ell.loc[ell == 0] = 0.01

        return ell

    def zmag(self, hvdc: bool = False) -> Series:
        """
        Get branch impedance magnitudes.

        Parameters
        ----------
        hvdc : bool, default False
            Include HVDC transmission lines.

        Returns
        -------
        pd.Series
            Impedance magnitude |Z| for each branch.
        """
        Y = self.ybranch(hvdc=hvdc)
        return 1 / np.abs(Y)

    def ybranch(self, asZ: bool = False, hvdc: bool = False) -> Series:
        """
        Get branch admittance (or impedance) in complex form.

        Parameters
        ----------
        asZ : bool, default False
            If True, return impedance Z. If False, return admittance Y.
        hvdc : bool, default False
            Include HVDC transmission lines (uses small impedance).

        Returns
        -------
        pd.Series
            Complex admittance Y = 1/(R + jX) or impedance Z = R + jX.
        """
        branches = self[Branch, ["LineR:2", "LineX:2"]]

        R = branches["LineR:2"]
        X = branches["LineX:2"]
        Z = R + 1j * X

        if hvdc:
            # Use small impedance for HVDC lines
            cnt = len(self[DCTransmissionLine])
            Zdc = Z[:cnt].copy()
            Zdc[:] = 0.001
            Z = concat([Z, Zdc], ignore_index=True)

        if asZ:
            return Z
        return 1 / Z

    def yshunt(self) -> Series:
        """
        Get branch shunt admittance in complex form.

        Returns
        -------
        pd.Series
            Complex shunt admittance Y = G + jB.
        """
        branches = self[Branch, ["LineG", "LineC"]]
        G = branches["LineG"]
        B = branches["LineC"]
        return G + 1j * B

    def gamma(self) -> Series:
        """
        Compute propagation constants for each branch.

        Returns
        -------
        pd.Series
            Complex propagation constant gamma = sqrt(Z * Y).
        """
        ell = self.lengths()
        Z = self.ybranch(asZ=True)
        Y = self.yshunt()

        # Handle zero values
        Z[Z == 0] = 0.000446 + 0.002878j
        Y[Y == 0] = 0.000463j

        # Per-unit length parameters
        Z = Z / ell
        Y = Y / ell

        return np.sqrt(Y * Z)

    def delay(self, min_delay: float = 10e-4) -> Series:
        r"""
        Compute effective propagation delay for network branches.

        Calculates the lossless propagation delay (beta) used to construct
        the Delay Graph Laplacian: L = A^T @ T^{-2} @ A.

        The effective branch capacitance accounts for capacitor banks and
        constant impedance reactive loads by averaging nodal capacitances
        at branch terminals (pi-model assumption).

        Parameters
        ----------
        min_delay : float, default 10e-4
            Minimum delay value to prevent numerical overflow when
            computing 1/delay^2 in the Laplacian.

        Returns
        -------
        pd.Series
            Effective propagation parameter beta for each branch.

        Notes
        -----
        Mathematical derivation:

        - Branch inductance: omega * L_ij = Im(Z^br_ij)
        - Effective capacitance: C_ij = (C_i + C_j) / 2
        - Propagation delay: omega * tau_ij = Im(sqrt(Z_ij * Y_ij)) = beta_ij

        For numerical stability, returns beta rather than tau = beta/omega.
        """
        # Edge series impedance
        Z = self.ybranch(asZ=True)

        # Effective edge shunt admittance (averaged from bus shunts)
        Ybus = self.esa.get_ybus()
        SUM = np.ones(Ybus.shape[0])
        AVG = np.abs(self.incidence()) / 2
        Y = AVG @ Ybus @ SUM

        # Propagation constant
        gam = np.sqrt(Z * Y)
        beta = np.imag(gam)

        # Enforce lower bound for numerical stability
        beta[beta < min_delay] = min_delay

        return beta
