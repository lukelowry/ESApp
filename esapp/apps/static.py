"""
Static Analysis Module
======================

This module provides specialized tools for static power system analysis,
including power flow operations, solver options, continuation power flow (CPF),
generator limit checking, and bus-level analysis.

Classes
-------
Statics
    Static analysis application with power flow, CPF, and state management.

Key Features
------------
- Power flow solution with configurable solver methods
- Solver option management (iterations, angle rotation, etc.)
- Continuation power flow for maximum transfer capability analysis
- State chain management for iterative algorithms
- Generator P/Q limit violation detection
- ZIP load injection interface
- Bus mismatch, net injection, and voltage violation analysis
- Branch admittance and Jacobian matrix computation

Example
-------
Basic power flow and analysis::

    >>> from esapp import GridWorkBench
    >>> wb = GridWorkBench("case.pwb")
    >>> wb.statics.pflow()
    >>> P, Q = wb.statics.mismatch()
    >>> viols = wb.statics.violations(v_min=0.95, v_max=1.05)

Continuation power flow::

    >>> interface = np.array([1, -1, 0, ...])
    >>> for mw in wb.statics.continuation_pf(interface, maxiter=100):
    ...     print(f"Converged at {mw:.2f} MW")

See Also
--------
esapp.apps.dynamics : Transient stability simulation.
esapp.apps.network : Network matrix construction.
"""

import warnings
from typing import Optional, Iterator, Union, Tuple

import numpy as np
from numpy import nan, exp, any
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix

from ..indexable import Indexable
from ..components import Gen, Load, Bus, Branch, Sim_Solution_Options

# Suppress FutureWarnings from pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

__all__ = ['Statics']


class Statics(Indexable):
    """
    Static analysis application for PowerWorld integration.

    Provides power flow operations, solver configuration, continuation power
    flow, generator limit checking, bus-level analysis, and matrix retrieval.

    This class uses lazy initialization - generator limits and ZIP load
    dispatch tables are cached on first use rather than at construction time.

    Attributes
    ----------
    genqmax : pd.Series
        Maximum reactive power limits (cached on first use).
    genqmin : pd.Series
        Minimum reactive power limits (cached on first use).
    genpmax : pd.Series
        Maximum active power limits (cached on first use).
    genpmin : pd.Series
        Minimum active power limits (cached on first use).
    DispatchPQ : DataFrame
        DataFrame for ZIP load dispatch at each bus (created on first use).
    """

    def __init__(self) -> None:
        super().__init__()
        self._gen_limits_cached = False
        self._dispatch_initialized = False

    # ------------------------------------------------------------------
    # Lazy initialization helpers
    # ------------------------------------------------------------------

    def _ensure_gen_limits(self) -> None:
        """Cache generator limits from PowerWorld on first access."""
        if self._gen_limits_cached:
            return
        gens = self[Gen, ['GenMVRMin', 'GenMVRMax', 'GenMWMax', 'GenMWMin']]
        self.genqmax = gens['GenMVRMax']
        self.genqmin = gens['GenMVRMin']
        self.genpmax = gens['GenMWMax']
        self.genpmin = gens['GenMWMin']
        self._gen_limits_cached = True

    _ZIP_FIELDS = ['LoadSMW', 'LoadSMVR', 'LoadIMW', 'LoadIMVR',
                    'LoadZMW', 'LoadZMVR']

    def _ensure_dispatch(self) -> None:
        """Create dispatch loads (LoadID='99') at every bus if not yet done.

        PowerWorld requires EDIT mode to create new objects, so this method
        temporarily enters EDIT mode, creates the loads, then returns to
        RUN mode.
        """
        if self._dispatch_initialized:
            return

        buses = self[Bus, ['BusNum', 'BusName_NomVolt']]

        dispatch = DataFrame({
            'BusNum':          buses['BusNum'].values,
            'BusName_NomVolt': buses['BusName_NomVolt'].values,
            'LoadID':          '99',
            'LoadStatus':      'Closed',
            **{zf: 0.0 for zf in self._ZIP_FIELDS},
        })

        # Enter EDIT mode so PowerWorld will create objects that don't exist
        self.esa.EnterMode('EDIT')
        try:
            self[Load] = dispatch
        finally:
            self.esa.EnterMode('RUN')

        # Cache the editable subset for fast updates in setload / clearloads
        self.DispatchPQ = dispatch[['BusNum', 'LoadID'] + self._ZIP_FIELDS].copy()
        self._dispatch_initialized = True

    # ------------------------------------------------------------------
    # State management (delegating to SAW)
    # ------------------------------------------------------------------

    def _store_state(self, name: str) -> None:
        """Save current PowerWorld state under given name."""
        self.esa.StoreState(name)

    def _restore_state(self, name: str) -> None:
        """Restore a previously saved PowerWorld state."""
        self.esa.RestoreState(name)

    def _delete_state(self, name: str) -> None:
        """Delete a previously saved PowerWorld state."""
        self.esa.DeleteState(name)

    # ------------------------------------------------------------------
    # Power flow
    # ------------------------------------------------------------------

    def pflow(
        self,
        getvolts: bool = True,
        method: str = "POLARNEWT",
    ) -> Optional[Union[pd.Series, Tuple[pd.Series, pd.Series]]]:
        """
        Solve power flow.

        Parameters
        ----------
        getvolts : bool, default True
            If True, return bus voltages after solving.
        method : str, default "POLARNEWT"
            Solution method (POLARNEWT, RECTNEWT, GAUSSSEIDEL, etc.).

        Returns
        -------
        pd.Series, tuple, or None
            Complex voltages if getvolts=True, else None.
        """
        self.esa.SolvePowerFlow(method)
        if getvolts:
            return self.voltage()

    def flatstart(self) -> None:
        """Reset the case to flat start (1.0 pu, 0 degrees)."""
        self.esa.ResetToFlatStart()

    def voltage(
        self,
        complex: bool = True,
        pu: bool = True,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Retrieve bus voltages.

        Parameters
        ----------
        complex : bool, default True
            If True, return complex voltage. Else (magnitude, angle_rad).
        pu : bool, default True
            If True, per-unit. Else kV.

        Returns
        -------
        pd.Series or tuple of pd.Series
        """
        fields = ["BusPUVolt", "BusAngle"] if pu else ["BusKVVolt", "BusAngle"]
        df = self[Bus, fields]
        mag = df[fields[0]]
        ang = df['BusAngle'] * np.pi / 180.0
        if complex:
            return mag * np.exp(1j * ang)
        return mag, ang

    # ------------------------------------------------------------------
    # Solver options
    # ------------------------------------------------------------------

    def _set_option(self, key: str, enable: bool) -> None:
        """Set a Sim_Solution_Options boolean flag."""
        self[Sim_Solution_Options, key] = 'YES' if enable else 'NO'

    def set_do_one_iteration(self, enable: bool = True) -> None:
        """Enable/disable single iteration mode for power flow."""
        self._set_option('DoOneIteration', enable)

    def set_max_iterations(self, val: int = 250) -> None:
        """Set maximum number of iterations for power flow convergence."""
        self[Sim_Solution_Options, 'MaxItr'] = val

    def set_disable_angle_rotation(self, enable: bool = True) -> None:
        """Enable/disable angle rotation during power flow."""
        self._set_option('DisableAngleRotation', enable)

    def set_disable_opt_mult(self, enable: bool = True) -> None:
        """Enable/disable optimal multiplier during power flow."""
        self._set_option('DisableOptMult', enable)

    def enable_inner_ss_check(self, enable: bool = True) -> None:
        """Enable/disable inner steady-state contingency check."""
        self._set_option('SSContPFInnerLoop', enable)

    def disable_gen_mvr_check(self, enable: bool = True) -> None:
        """Enable/disable generator MVAR limit checking."""
        self._set_option('DisableGenMVRCheck', enable)

    def enable_inner_check_gen_vars(self, enable: bool = True) -> None:
        """Enable/disable inner loop generator VAR checking."""
        self._set_option('ChkVars', enable)

    def enable_inner_backoff_gen_vars(self, enable: bool = True) -> None:
        """Enable/disable inner loop generator VAR backoff."""
        self._set_option('ChkVars:1', enable)

    # ------------------------------------------------------------------
    # Bus-level analysis
    # ------------------------------------------------------------------

    def violations(self, v_min: float = 0.9, v_max: float = 1.1) -> DataFrame:
        """
        Return bus voltage violations.

        Parameters
        ----------
        v_min : float, default 0.9
            Low voltage threshold (pu).
        v_max : float, default 1.1
            High voltage threshold (pu).

        Returns
        -------
        DataFrame
            Columns 'Low' and 'High' with violating bus voltages.
        """
        v = self.voltage(complex=False, pu=True)[0]
        low = v[v < v_min]
        high = v[v > v_max]
        return DataFrame({'Low': low, 'High': high})

    def mismatch(self, asComplex: bool = False):
        """
        Return bus power mismatches.

        Parameters
        ----------
        asComplex : bool, default False
            If True, return P + jQ as complex Series.

        Returns
        -------
        tuple of pd.Series or pd.Series
            (P, Q) mismatches or complex S = P + jQ.
        """
        df = self[Bus, ["BusMismatchP", "BusMismatchQ"]]
        P = df['BusMismatchP']
        Q = df['BusMismatchQ']
        if asComplex:
            return P + 1j * Q
        return P, Q

    def netinj(self, asComplex: bool = False):
        """
        Sum of all generator, load, bus shunt, and switched shunt P and Q.

        Parameters
        ----------
        asComplex : bool, default False
            If True, return P + jQ as complex array.

        Returns
        -------
        tuple of np.ndarray or np.ndarray
            (P, Q) or complex S = P + jQ.
        """
        df = self[Bus, ['BusNetMW', 'BusNetMVR']]
        P = df['BusNetMW'].to_numpy()
        Q = df['BusNetMVR'].to_numpy()
        if asComplex:
            return P + 1j * Q
        return P, Q

    # ------------------------------------------------------------------
    # Matrix retrieval
    # ------------------------------------------------------------------

    def ybus(self, dense: bool = False):
        """
        Return the Y-Bus matrix.

        Parameters
        ----------
        dense : bool, default False
            If True, return dense array. Else sparse CSR.

        Returns
        -------
        np.ndarray or scipy.sparse.csr_matrix
        """
        return self.esa.get_ybus(dense)

    def branch_admittance(self) -> Tuple[csr_matrix, csr_matrix]:
        """
        Compute branch admittance matrices Yf and Yt.

        Returns
        -------
        tuple of csr_matrix
            (Yf, Yt) branch-bus admittance matrices.
        """
        bus_df = self[Bus, ["BusNum"]]
        branch_df = self[Branch, [
            "BusNum", "BusNum:1", "LineCircuit",
            "LineR", "LineX", "LineC", "LineTap", "LinePhase",
        ]]

        nb = len(bus_df)
        nl = len(branch_df)

        Ys = 1 / (branch_df["LineR"].to_numpy()
                   + 1j * branch_df["LineX"].to_numpy())
        Bc = branch_df["LineC"].to_numpy()
        tap = (branch_df["LineTap"].to_numpy()
               * np.exp(1j * np.pi / 180 * branch_df["LinePhase"].to_numpy()))
        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / (tap * np.conj(tap))
        Yft = -Ys / np.conj(tap)
        Ytf = -Ys / tap

        bus_to_idx = {bus: idx for idx, bus in enumerate(bus_df["BusNum"])}
        f = np.array([bus_to_idx[b] for b in branch_df["BusNum"]])
        t = np.array([bus_to_idx[b] for b in branch_df["BusNum:1"]])

        i = np.r_[range(nl), range(nl)]
        Yf = csr_matrix(
            (np.hstack([Yff.reshape(-1), Yft.reshape(-1)]),
             (i, np.hstack([f, t]))),
            (nl, nb),
        )
        Yt = csr_matrix(
            (np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]),
             (i, np.hstack([f, t]))),
            (nl, nb),
        )
        return Yf, Yt

    def jacobian(self, dense: bool = False, form: str = 'R'):
        """
        Get the power flow Jacobian matrix.

        Parameters
        ----------
        dense : bool, default False
            If True, return dense array. Else sparse CSR.
        form : str, default 'R'
            Coordinate form: 'R' (rectangular), 'P' (polar), 'DC' (B').

        Returns
        -------
        np.ndarray or scipy.sparse.csr_matrix
        """
        return self.esa.get_jacobian(dense, form=form)

    def jacobian_with_ids(self, dense: bool = False, form: str = 'R'):
        """
        Get the Jacobian matrix with row/column ID labels.

        Parameters
        ----------
        dense : bool, default False
            If True, return dense array. Else sparse CSR.
        form : str, default 'R'
            Coordinate form: 'R' (rectangular), 'P' (polar), 'DC' (B').

        Returns
        -------
        tuple
            (matrix, row_ids) where row_ids describes each row/column.
        """
        return self.esa.get_jacobian_with_ids(dense, form=form)

    def set_voltages(self, V: np.ndarray) -> None:
        """
        Set bus voltages from a complex vector.

        Parameters
        ----------
        V : np.ndarray
            Complex voltage vector (per-unit).
        """
        V_df = np.vstack([np.abs(V), np.angle(V, deg=True)]).T
        self[Bus, ["BusPUVolt", "BusAngle"]] = V_df

    # ------------------------------------------------------------------
    # Generator limit checking
    # ------------------------------------------------------------------

    def gens_above_pmax(
        self,
        p: Optional[np.ndarray] = None,
        is_closed: Optional[np.ndarray] = None,
        tol: float = 0.001,
    ) -> bool:
        """
        Check if any closed generators exceed P limits.

        Parameters
        ----------
        p : array-like, optional
            Generator MW output. If None, reads from case.
        is_closed : array-like, optional
            Boolean mask of closed generators. If None, reads from case.
        tol : float, default 0.001
            Tolerance for violation (MW).

        Returns
        -------
        bool
            True if any closed generator violates P limits.
        """
        self._ensure_gen_limits()
        if p is None:
            p = self[Gen, 'GenMW']['GenMW']
        is_high = p > self.genpmax + tol
        is_low = p < self.genpmin - tol
        if is_closed is None:
            is_closed = self[Gen, 'GenStatus']['GenStatus'] == 'Closed'
        violation = is_closed & (is_high | is_low)
        return any(violation)

    def gens_above_qmax(
        self,
        q: Optional[np.ndarray] = None,
        is_closed: Optional[np.ndarray] = None,
        tol: float = 0.001,
    ) -> bool:
        """
        Check if any closed generators exceed Q limits.

        Parameters
        ----------
        q : array-like, optional
            Generator MVAr output. If None, reads from case.
        is_closed : array-like, optional
            Boolean mask of closed generators. If None, reads from case.
        tol : float, default 0.001
            Tolerance for violation (MVAr).

        Returns
        -------
        bool
            True if any closed generator violates Q limits.
        """
        self._ensure_gen_limits()
        if q is None:
            q = self[Gen, 'GenMVR']['GenMVR']
        is_high = q > self.genqmax + tol
        is_low = q < self.genqmin - tol
        if is_closed is None:
            is_closed = self[Gen, 'GenStatus']['GenStatus'] == 'Closed'
        violation = is_closed & (is_high | is_low)
        return any(violation)

    # ------------------------------------------------------------------
    # Continuation power flow (predictor-corrector)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cpf_dFdlam(
        interface: np.ndarray,
        bus_to_idx: dict,
        jac_ids: list,
        sbase: float,
    ) -> np.ndarray:
        """Map a bus-indexed interface vector into Jacobian row ordering.

        For each row labelled ``"dP <busnum>"`` the corresponding entry
        is ``-interface[bus_idx] / sbase``.  Rows labelled ``"dQ ..."``
        (reactive power equations) are left at zero because the load
        increase pattern only affects active power mismatches.

        Parameters
        ----------
        interface : np.ndarray
            Bus-indexed MW injection pattern.
        bus_to_idx : dict
            ``{bus_number: array_index}`` mapping.
        jac_ids : list[str]
            Row labels returned by ``jacobian_with_ids(form='P')``.
        sbase : float
            System MVA base for per-unit conversion.

        Returns
        -------
        np.ndarray
            ``dF/dlambda`` vector of length ``len(jac_ids)``.
        """
        dF = np.zeros(len(jac_ids))
        for row, raw_label in enumerate(jac_ids):
            parts = raw_label.strip().strip("'\"").split()
            if len(parts) < 2:
                continue
            if not parts[0].lower().startswith('dp'):
                continue
            try:
                bus = int(parts[1])
            except ValueError:
                continue
            idx = bus_to_idx.get(bus)
            if idx is not None:
                dF[row] = -interface[idx] / sbase
        return dF

    # ---- small helper used inside the CPF loop ----------------------------

    def _cpf_halve_step(self, step: float, min_step: float) -> float:
        """Return ``step / 2``, or ``-1`` if below *min_step*."""
        step *= 0.5
        return step if step >= min_step else -1.0

    # ---- main CPF entry point ---------------------------------------------

    def continuation_pf(
        self,
        interface: np.ndarray,
        initialmw: float = 0,
        step_size: float = 0.05,
        min_step: float = 0.001,
        max_step: float = 0.1,
        maxiter: int = 200,
        verbose: bool = False,
        restore_when_done: bool = False,
        qlim_tol: Optional[float] = 0,
        plim_tol: Optional[float] = None,
        sbase: float = 100.0,
    ) -> Iterator[float]:
        """Predictor-corrector continuation power flow.

        Traces the PV curve by computing tangent vectors from an
        augmented Jacobian (predictor) then solving power flow at the
        predicted operating point (corrector).  Parameterisation is
        switched automatically from *lambda* to the voltage magnitude
        with the largest sensitivity near the nose point.

        Parameters
        ----------
        interface : np.ndarray
            Bus-indexed injection pattern (MW).  Length must equal the
            number of buses.
        initialmw : float, default 0
            Starting interface transfer level (MW).
        step_size : float, default 0.05
            Initial normalised arc-length step.
        min_step : float, default 0.001
            Minimum step size before the algorithm terminates.
        max_step : float, default 0.1
            Maximum step size.
        maxiter : int, default 200
            Maximum predictor-corrector iterations.
        verbose : bool, default False
            Print progress to stdout.
        restore_when_done : bool, default False
            Restore the original PowerWorld state on exit.
        qlim_tol : float or None, default 0
            Reactive-power limit tolerance (MVAr).  ``None`` disables.
        plim_tol : float or None, default None
            Active-power limit tolerance (MW).  ``None`` disables.
        sbase : float, default 100.0
            System MVA base for per-unit conversion.

        Yields
        ------
        float
            Interface transfer level (MW) after each converged solution.
        """
        # -- lazy caches & optional backup ----------------------------------
        self._ensure_gen_limits()
        self._ensure_dispatch()

        log = (lambda msg, **kw: print(msg, **kw)) if verbose else (lambda *a, **k: None)

        if restore_when_done:
            self._store_state('CPF_BACKUP')

        # -- base-case solution ---------------------------------------------
        lam_current = initialmw
        self.setload(SP=-lam_current * interface)
        self.pflow(getvolts=False)
        self._store_state('CPF_PREV')
        yield lam_current

        # -- Jacobian structure & dF/dlambda --------------------------------
        J0, jac_ids = self.jacobian_with_ids(dense=True, form='P')
        n_jac = J0.shape[0]

        bus_nums = self[Bus, 'BusNum']['BusNum'].to_numpy()
        bus_to_idx = {int(b): i for i, b in enumerate(bus_nums)}

        dF_dlam = self._build_cpf_dFdlam(interface, bus_to_idx, jac_ids, sbase)

        # -- continuation state ---------------------------------------------
        step = step_size
        cont_param = n_jac               # index of continuation variable (lambda)
        tangent_prev = np.zeros(n_jac + 1)
        tangent_prev[-1] = 1.0           # initial direction: pure lambda increase
        crossed_nose = False

        for it in range(maxiter):
            # ============ PREDICTOR ========================================
            J = self.jacobian(dense=True, form='P')

            # Augmented system:  [J  dF/dlam] [dx  ]   [0]
            #                    [e_k    0  ] [dlam] = [1]
            J_aug = np.zeros((n_jac + 1, n_jac + 1))
            J_aug[:n_jac, :n_jac] = J
            J_aug[:n_jac, n_jac]  = dF_dlam
            J_aug[n_jac, cont_param] = 1.0

            rhs = np.zeros(n_jac + 1)
            rhs[n_jac] = 1.0

            try:
                tangent = np.linalg.solve(J_aug, rhs)
            except np.linalg.LinAlgError:
                log(f'  [{it}] Singular augmented Jacobian')
                step = self._cpf_halve_step(step, min_step)
                if step < 0:
                    break
                continue

            # Normalise and orient consistently with previous tangent
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-15:
                break
            tangent /= tnorm
            if np.dot(tangent, tangent_prev) < 0:
                tangent = -tangent

            lam_pred = lam_current + step * tangent[-1]
            log(f'  [{it}] Predict: lam={lam_pred:.2f} MW  '
                f'(step={step:.4f}, dlam={step * tangent[-1]:.2f})',
                end='')

            # ============ CORRECTOR ========================================
            self.setload(SP=-lam_pred * interface)
            try:
                self.pflow(getvolts=False)
            except Exception:
                log(' FAIL', end='')
                step = self._cpf_halve_step(step, min_step)
                if step < 0:
                    log(f'\n  Step below minimum ({min_step})')
                    break
                self._restore_state('CPF_PREV')
                log(f' -> retry (step={step:.4f})')
                continue

            # ============ GENERATOR LIMIT CHECKS ===========================
            reject = False
            if qlim_tol is not None:
                gen_df = self[Gen, ['GenMVR', 'GenStatus']]
                closed = gen_df['GenStatus'] == 'Closed'
                if self.gens_above_qmax(gen_df['GenMVR'], closed, tol=qlim_tol):
                    log(' Q-LIM', end='')
                    reject = True

            if not reject and plim_tol is not None:
                if self.gens_above_pmax(tol=plim_tol):
                    log(' P-LIM', end='')
                    reject = True

            if reject:
                step = self._cpf_halve_step(step, min_step)
                if step < 0:
                    break
                self._restore_state('CPF_PREV')
                continue

            # ============ NOSE DETECTION ===================================
            if tangent_prev[-1] > 0 and tangent[-1] < 0:
                crossed_nose = True
                log(' NOSE', end='')

            # ============ PARAMETERISATION SWITCH ==========================
            if abs(tangent[-1]) < 0.1:
                # Switch to the voltage variable with the largest |dV/ds|.
                # In the polar Jacobian the |V| entries occupy the second
                # half of the state vector.
                n_half = n_jac // 2
                v_sens = np.abs(tangent[n_half:n_jac])
                if len(v_sens) > 0:
                    best = int(np.argmax(v_sens)) + n_half
                    if cont_param != best:
                        cont_param = best
                        log(f' SWITCH(V[{best - n_half}])', end='')
            else:
                cont_param = n_jac  # back to lambda

            # ============ STEP-SIZE ADAPTATION =============================
            cos_angle = np.clip(np.dot(tangent, tangent_prev), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < 0.05:
                step = min(step * 1.5, max_step)
            elif angle > 0.3:
                step = max(step * 0.5, min_step)

            # ============ ACCEPT STEP ======================================
            self._store_state('CPF_PREV')
            tangent_prev = tangent.copy()
            lam_current = lam_pred

            log(f'  OK (lam={lam_current:.2f})')
            yield lam_current

            if crossed_nose and lam_current < initialmw:
                log(f'  Lambda returned below initial ({initialmw})')
                break

        # -- cleanup --------------------------------------------------------
        self.clearloads()
        if restore_when_done:
            self._restore_state('CPF_BACKUP')

    # ------------------------------------------------------------------
    # State chain management
    # ------------------------------------------------------------------

    def chain(self, maxstates: int = 2) -> None:
        """
        Initialize state chain for iterative algorithms.

        Parameters
        ----------
        maxstates : int, default 2
            Maximum number of states to retain.
        """
        self.maxstates = maxstates
        self.stateidx = -1

    def pushstate(self, verbose: bool = False) -> None:
        """
        Push current state onto the state chain.

        Parameters
        ----------
        verbose : bool, default False
            Print state management info.
        """
        self.stateidx += 1
        self._store_state(f'GWBState{self.stateidx}')
        if verbose:
            print(f'Pushed States -> {self.stateidx}')
        if self.stateidx >= self.maxstates:
            self._delete_state(f'GWBState{self.stateidx - self.maxstates}')

    def istore(self, n: int = 0, verbose: bool = False) -> None:
        """
        Update the nth state in the chain with current state.

        Parameters
        ----------
        n : int, default 0
            State offset from current (0 = most recent).
        verbose : bool, default False
            Print state management info.

        Raises
        ------
        Exception
            If n exceeds available states.
        """
        if n > self.maxstates or n > self.stateidx:
            raise Exception("State index out of range")
        if verbose:
            print(f'Store -> {self.stateidx - n}')
        self._store_state(f'GWBState{self.stateidx - n}')

    def irestore(self, n: int = 1, verbose: bool = False) -> None:
        """
        Restore the nth previous state from the chain.

        Parameters
        ----------
        n : int, default 1
            State offset from current (1 = previous state).
        verbose : bool, default False
            Print state management info.

        Raises
        ------
        Exception
            If n exceeds available states.
        """
        if n > self.maxstates or n > self.stateidx:
            if verbose:
                print('Restoration Failure')
            raise Exception("State index out of range")
        if verbose:
            print(f'Restore -> {self.stateidx - n}')
        self._restore_state(f'GWBState{self.stateidx - n}')

    # ------------------------------------------------------------------
    # ZIP load interface
    # ------------------------------------------------------------------

    def setload(
        self,
        SP: Optional[np.ndarray] = None,
        SQ: Optional[np.ndarray] = None,
        IP: Optional[np.ndarray] = None,
        IQ: Optional[np.ndarray] = None,
        ZP: Optional[np.ndarray] = None,
        ZQ: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set ZIP load components on the dispatch loads (LoadID='99').

        Only the components that are explicitly provided are written to
        PowerWorld; the rest retain their previous values.  Each array
        must be bus-indexed (same length and order as ``self[Bus]``).

        Parameters
        ----------
        SP, SQ : np.ndarray, optional
            Constant-power MW / MVAr.
        IP, IQ : np.ndarray, optional
            Constant-current MW / MVAr.
        ZP, ZQ : np.ndarray, optional
            Constant-impedance MW / MVAr.
        """
        self._ensure_dispatch()

        _col_map = {
            'LoadSMW': SP,  'LoadSMVR': SQ,
            'LoadIMW': IP,  'LoadIMVR': IQ,
            'LoadZMW': ZP,  'LoadZMVR': ZQ,
        }

        # Determine which columns were actually provided
        changed_cols = []
        for col, arr in _col_map.items():
            if arr is not None:
                self.DispatchPQ[col] = arr
                changed_cols.append(col)

        if not changed_cols:
            return  # nothing to do

        # Write only the key columns + changed fields
        write_cols = ['BusNum', 'LoadID'] + changed_cols
        self[Load] = self.DispatchPQ[write_cols]

    def clearloads(self) -> None:
        """Zero all six ZIP components on the dispatch loads."""
        self._ensure_dispatch()
        self.DispatchPQ[self._ZIP_FIELDS] = 0.0
        self[Load] = self.DispatchPQ

    # ------------------------------------------------------------------
    # Random load variation
    # ------------------------------------------------------------------

    load_nom = None
    load_df = None

    def randomize_load(self, scale: float = 1.0, sigma: float = 0.1) -> None:
        """
        Apply random variation to system loads.

        Parameters
        ----------
        scale : float, default 1.0
            Base scale factor.
        sigma : float, default 0.1
            Standard deviation of log-normal distribution.
        """
        if self.load_nom is None or self.load_df is None:
            self.load_df = self[Load, 'LoadMW']
            self.load_nom = self.load_df['LoadMW']
        random_factors = exp(sigma * np.random.random(len(self.load_nom)))
        self[Load, 'LoadMW'] = scale * self.load_nom * random_factors

    randload = randomize_load
