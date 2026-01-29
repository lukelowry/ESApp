"""
Static Analysis Module
======================

This module provides specialized tools for advanced static power system analysis,
including continuation power flow (CPF), random load variation, and generator
limit checking.

Classes
-------
Statics
    Research-focused static analysis application with CPF and state management.

Key Features
------------
- Continuation power flow for maximum transfer capability analysis
- State chain management for iterative algorithms
- Generator P/Q limit violation detection
- ZIP load injection interface

Example
-------
Basic continuation power flow::

    >>> from esapp import GridWorkBench
    >>> wb = GridWorkBench("case.pwb")
    >>> interface = np.array([1, -1, 0, ...])  # Injection pattern
    >>> for mw in wb.statics.continuation_pf(interface, maxiter=100):
    ...     print(f"Converged at {mw:.2f} MW")

See Also
--------
esapp.apps.dynamics : Transient stability simulation.
esapp.apps.network : Network matrix construction.
"""

import warnings
from typing import Optional, Callable, Iterator

import numpy as np
from numpy import nan, exp, any, arange, inf
from pandas import DataFrame

from ..indexable import Indexable
from ..components import Gen, Load, Bus
from ..saw._exceptions import (
    BifurcationException,
    GeneratorLimitException,
)

# Suppress FutureWarnings from pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

__all__ = ['Statics']


class Statics(Indexable):
    """
    Research-focused static analysis application.

    Provides specialized functions for continuation power flow (CPF),
    random load variation, and advanced static analysis methods.
    These functions are intentionally untested as they support highly
    specific research and data analysis workflows.

    For general-purpose functions, use GridWorkBench methods:
    - ``wb.gens_above_pmax()`` / ``wb.gens_above_qmax()`` for limit checking
    - ``wb.init_state_chain()`` / ``wb.push_state()`` for state management
    - ``wb.set_zip_load()`` / ``wb.clear_zip_loads()`` for load injection

    Attributes
    ----------
    genqmax : pd.Series
        Maximum reactive power limits for all generators.
    genqmin : pd.Series
        Minimum reactive power limits for all generators.
    genpmax : pd.Series
        Maximum active power limits for all generators.
    genpmin : pd.Series
        Minimum active power limits for all generators.
    DispatchPQ : DataFrame
        DataFrame for ZIP load dispatch at each bus.
    """

    io: Indexable

    def __init__(self) -> None:
        gens = self[Gen, ['GenMVRMin', 'GenMVRMax']]
        buses = self[Bus]

        zipfields = ['LoadSMW', 'LoadSMVR', 'LoadIMW', 'LoadIMVR', 'LoadZMW', 'LoadZMVR']

        # Generator Q limits
        self.genqmax = gens['GenMVRMax']
        self.genqmin = gens['GenMVRMin']

        # Generator P limits
        self.genpmax = gens['GenMWMax']
        self.genpmin = gens['GenMWMin']

        # Create DataFrame for manipulable loads at all buses
        load_df = buses[['BusNum', 'BusName_NomVolt']].copy()
        load_df.loc[:, zipfields] = 0.0
        load_df['LoadID'] = 99  # Large ID to avoid interference
        load_df['LoadStatus'] = 'Closed'
        load_df = load_df.fillna(0)

        # Send to PowerWorld
        self[Load] = load_df

        # Smaller DataFrame for updating constant power at buses
        self.DispatchPQ = load_df[['BusNum', 'LoadID'] + zipfields].copy()

    # State for random load variation
    load_nom = None
    load_df = None

    def randomize_load(self, scale: float = 1.0, sigma: float = 0.1) -> None:
        """
        Apply random variation to system loads.

        Temporarily modifies load values with log-normal random scaling.
        Original load values are cached for restoration.

        Parameters
        ----------
        scale : float, default 1.0
            Base scale factor for all loads.
        sigma : float, default 0.1
            Standard deviation of log-normal distribution.
        """
        if self.load_nom is None or self.load_df is None:
            self.load_df = self[Load, 'LoadMW']
            self.load_nom = self.load_df['LoadMW']

        random_factors = exp(sigma * np.random.random(len(self.load_nom)))
        self[Load, 'LoadMW'] = scale * self.load_nom * random_factors

    # Backwards compatibility alias
    randload = randomize_load

    def gens_above_pmax(
        self,
        p: Optional[np.ndarray] = None,
        is_closed: Optional[np.ndarray] = None,
        tol: float = 0.001
    ) -> bool:
        """
        Check if any closed generators exceed P limits.

        Parameters
        ----------
        p : np.ndarray, optional
            Generator MW output. If None, reads from case.
        is_closed : np.ndarray, optional
            Boolean mask of closed generators. If None, reads from case.
        tol : float, default 0.001
            Tolerance for limit violation (MW).

        Returns
        -------
        bool
            True if any closed generator violates P limits.
        """
        if p is None:
            p = self[Gen, 'GenMW']['GenMW']

        is_high = p > self.genpmax + tol
        is_low = p < self.genpmin - tol

        if is_closed is None:
            is_closed = self[Gen, 'GenStatus']['GenStatus'] == 'Closed'

        violation = is_closed & (is_high | is_low)
        return any(violation)

    # Backwards compatibility alias
    gensAbovePMax = gens_above_pmax

    def gens_above_qmax(
        self,
        q: Optional[np.ndarray] = None,
        is_closed: Optional[np.ndarray] = None,
        tol: float = 0.001
    ) -> bool:
        """
        Check if any closed generators exceed Q limits.

        Parameters
        ----------
        q : np.ndarray, optional
            Generator MVAr output. If None, reads from case.
        is_closed : np.ndarray, optional
            Boolean mask of closed generators. If None, reads from case.
        tol : float, default 0.001
            Tolerance for limit violation (MVAr).

        Returns
        -------
        bool
            True if any closed generator violates Q limits.
        """
        if q is None:
            q = self[Gen, 'GenMVR']['GenMVR']

        is_high = q > self.genqmax + tol
        is_low = q < self.genqmin - tol

        if is_closed is None:
            is_closed = self[Gen, 'GenStatus']['GenStatus'] == 'Closed'

        violation = is_closed & (is_high | is_low)
        return any(violation)

    # Backwards compatibility alias
    gensAboveQMax = gens_above_qmax

    def continuation_pf(
        self,
        interface: np.ndarray,
        initialmw: float = 0,
        minstep: float = 1,
        maxstep: float = 50,
        maxiter: int = 200,
        nrtol: float = 0.0001,
        verbose: bool = False,
        boundary_func: Optional[Callable] = None,
        restore_when_done: bool = False,
        qlimtol: Optional[float] = 0,
        plimtol: Optional[float] = None,
        bifur_check: bool = True
    ) -> Iterator[float]:
        """
        Continuation power flow for maximum transfer capability.

        Iteratively increases interface injection until the system reaches
        a voltage stability boundary (bifurcation point) or generator limits.

        Parameters
        ----------
        interface : np.ndarray
            Injection pattern vector (positive = supply, negative = demand).
        initialmw : float, default 0
            Starting interface MW level.
        minstep : float, default 1
            Minimum step size (MW) - determines solution accuracy.
        maxstep : float, default 50
            Maximum step size (MW) per iteration.
        maxiter : int, default 200
            Maximum number of iterations.
        nrtol : float, default 0.0001
            Newton-Raphson MVA tolerance.
        verbose : bool, default False
            Print progress information.
        boundary_func : callable, optional
            Function to call at the boundary. Result stored in func.X.
        restore_when_done : bool, default False
            Restore original state after completion.
        qlimtol : float, optional
            Q limit tolerance. None disables Q limit checking.
        plimtol : float, optional
            P limit tolerance. None disables P limit checking.
        bifur_check : bool, default True
            Enable bifurcation detection.

        Yields
        ------
        float
            Interface MW at each stable solution point.

        Notes
        -----
        The algorithm uses adaptive step sizing with binary search backstep
        on failure. Stability is detected by monitoring total reactive power
        output - a drop indicates approaching the nose of the PV curve.
        """
        def log(x, **kwargs):
            if verbose:
                print(x, **kwargs)

        # Save state if restoration requested
        if restore_when_done:
            self.save_state('BACKUP')

        # Initialize state chain
        self.chain()
        self.pushstate()
        self.pushstate()

        # For solution continuity
        self.save_state('PREV')

        # Set NR tolerance
        self.set_mva_tol(nrtol)

        log(f'Starting Injection at: {initialmw:.4f} MW')

        # Iteration tracking
        backstep_percent = 0.25
        pnow, step = initialmw, maxstep
        pstable, pprev = initialmw, initialmw
        qstable, qprev = -inf, -inf
        qmax, pmax = -inf, initialmw
        last_stable_index = 0

        # Continuation loop
        for i in arange(maxiter):
            # Set injection for this iteration
            self.setload(SP=-pnow * interface)

            try:
                # Solve power flow
                log(f'\nPF: {pnow:>12.4f} MW', end='\t')
                self.pflow()

                # Check generator limits
                qall = self[Gen, ['GenMVR', 'GenStatus']]
                qclosed = qall['GenStatus'] == 'Closed'

                if qlimtol is not None and self.gens_above_qmax(qall['GenMVR'], qclosed, tol=qlimtol):
                    log(' Q+ ', end=' ')
                    raise GeneratorLimitException

                if plimtol is not None and self.gens_above_pmax(None, qclosed, tol=plimtol):
                    log(' P+ ', end=' ')
                    raise GeneratorLimitException

                # Stability indicator
                qsum = qall['GenMVR'].sum()

                # Stability criteria:
                # - At least 1 previous solution
                # - Net Q risen above previous stable
                # - Net Q risen above known maximum
                # - MW injection above previous stable point
                is_stable = (
                    (i > 0) and
                    (qsum > qstable) and
                    (qsum > qmax) and
                    (pnow > pstable) and
                    (pnow > pmax)
                )

                # Stable solution handling
                if is_stable:
                    log(' ST ', end=' ')
                    self.pushstate()

                    if last_stable_index > 0:
                        self.irestore(1)
                        yield pprev
                        self.irestore(0)

                    last_stable_index = i
                    pstable, qstable = pprev, qprev

                # Bifurcation detection
                if bifur_check and (i - last_stable_index > 4):
                    log(' SL+ ', end=' ')
                    raise BifurcationException

                # Store as solved (but not necessarily stable)
                self.save_state('PREV')

                pmax, qmax = max(pnow, pprev), max(qsum, qprev)
                pprev, qprev = pnow, qsum

            except BifurcationException:
                pnow = pstable
                pprev = pstable
                qprev = qstable
                step *= backstep_percent
                self.irestore(1)

            except (Exception, GeneratorLimitException):
                log('XXX', end=' ')

                if i == 0:
                    log('First Injection Failed. Check for LV solution or already past boundary.')
                    self.restore_state('PREV')
                    log('-----------EXIT-----------\n\n')
                    return

                pnow = pprev
                step *= backstep_percent
                if pprev != 0:
                    self.irestore(1)

            # Termination condition
            if step < minstep:
                break

            # Advance injection
            pnow += step

        # Execute boundary function
        if boundary_func is not None:
            self.irestore(1)
            log(f'BD: {pprev:>12.4f} MW\t ! ')
            log('Calling Boundary Function...')
            boundary_func.X = boundary_func()

        # Reset dispatch loads
        self.setload(SP=0 * interface)

        # Restore original state if requested
        if restore_when_done:
            self.restore_state('BACKUP')
        log('-----------EXIT-----------\n\n')

    def chain(self, maxstates: int = 2) -> None:
        """
        Initialize state chain for iterative algorithms.

        Creates a queue-based state management system for algorithms
        that need to track and restore multiple previous states.

        Parameters
        ----------
        maxstates : int, default 2
            Maximum number of states to retain in the chain.
        """
        self.maxstates = maxstates
        self.stateidx = -1

    def pushstate(self, verbose: bool = False) -> None:
        """
        Push current state onto the state chain.

        Saves the current power flow state and removes the oldest
        state if the chain exceeds maxstates.

        Parameters
        ----------
        verbose : bool, default False
            Print state management information.
        """
        self.stateidx += 1
        self.save_state(f'GWBState{self.stateidx}')

        if verbose:
            print(f'Pushed States -> {self.stateidx}, Delete -> {self.stateidx - self.maxstates}')

        if self.stateidx >= self.maxstates:
            self.delete_state(f'GWBState{self.stateidx - self.maxstates}')

    def istore(self, n: int = 0, verbose: bool = False) -> None:
        """
        Update the nth state in the chain with current state.

        Parameters
        ----------
        n : int, default 0
            State offset from current (0 = most recent).
        verbose : bool, default False
            Print state management information.

        Raises
        ------
        Exception
            If n exceeds available states.
        """
        if n > self.maxstates or n > self.stateidx:
            raise Exception("State index out of range")

        if verbose:
            print(f'Store -> {self.stateidx - n}')

        self.save_state(f'GWBState{self.stateidx - n}')

    def irestore(self, n: int = 1, verbose: bool = False) -> None:
        """
        Restore the nth previous state from the chain.

        Consecutive calls restore the same state (non-destructive).

        Parameters
        ----------
        n : int, default 1
            State offset from current (1 = previous state).
        verbose : bool, default False
            Print state management information.

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

        self.restore_state(f'GWBState{self.stateidx - n}')

    def setload(
        self,
        SP: Optional[np.ndarray] = None,
        SQ: Optional[np.ndarray] = None,
        IP: Optional[np.ndarray] = None,
        IQ: Optional[np.ndarray] = None,
        ZP: Optional[np.ndarray] = None,
        ZQ: Optional[np.ndarray] = None
    ) -> None:
        """
        Set ZIP load components at each bus.

        Provides a fast interface for applying load deltas independent
        of existing loads. Uses LoadID=99 to avoid interference.

        Parameters
        ----------
        SP : np.ndarray, optional
            Constant active power (MW) at each bus.
        SQ : np.ndarray, optional
            Constant reactive power (MVAr) at each bus.
        IP : np.ndarray, optional
            Constant real current component.
        IQ : np.ndarray, optional
            Constant reactive current component.
        ZP : np.ndarray, optional
            Constant resistance component.
        ZQ : np.ndarray, optional
            Constant reactance component.

        Notes
        -----
        All vectors must include every bus. These loads are temporary
        and can be overwritten by other GridWorkBench functions.
        """
        fields = ['BusNum', 'LoadID']

        if SP is not None:
            fields.append('LoadSMW')
            self.DispatchPQ.loc[:, 'LoadSMW'] = SP
        if SQ is not None:
            fields.append('LoadSMVR')
            self.DispatchPQ.loc[:, 'LoadSMVR'] = SQ
        if IP is not None:
            fields.append('LoadIMW')
            self.DispatchPQ.loc[:, 'LoadIMW'] = IP
        if IQ is not None:
            fields.append('LoadIMVR')
            self.DispatchPQ.loc[:, 'LoadIMVR'] = IQ
        if ZP is not None:
            fields.append('LoadZMW')
            self.DispatchPQ.loc[:, 'LoadZMW'] = ZP
        if ZQ is not None:
            fields.append('LoadZMVR')
            self.DispatchPQ.loc[:, 'LoadZMVR'] = ZQ

        self[Load] = self.DispatchPQ.loc[:, fields]

    def clearloads(self) -> None:
        """
        Clear all script-applied ZIP loads.

        Resets all ZIP load components set via setload() to zero.
        """
        zipfields = ['LoadSMW', 'LoadSMVR', 'LoadIMW', 'LoadIMVR', 'LoadZMW', 'LoadZMVR']
        self.DispatchPQ.loc[:, zipfields] = 0
        self[Load] = self.DispatchPQ
