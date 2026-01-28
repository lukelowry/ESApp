"""
Transient Stability Simulation Module
=====================================

This module provides a high-level interface for running transient stability
simulations in PowerWorld Simulator.

Example:
    >>> from esapp import GridWorkBench, TS
    >>> from esapp.grid import Gen, Bus
    >>>
    >>> wb = GridWorkBench("case.pwb")
    >>> wb.dyn.runtime = 10.0
    >>> wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])
    >>>
    >>> (wb.dyn.contingency("Bus_Fault")
    ...        .at(1.0).fault_bus("101")
    ...        .at(1.1).clear_fault("101"))
    >>>
    >>> meta, results = wb.dyn.solve("Bus_Fault")
    >>> wb.dyn.plot(meta, results)
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Tuple, Dict, Union, Optional, Any, Type

from pandas import DataFrame, concat

from ..indexable import Indexable
from ..gobject import GObject
from ..ts_fields import TS
from esapp.grid import TSContingency, TSContingencyElement

# Configure logger
logger = logging.getLogger(__name__)

# Re-export TS for backward compatibility - users can import from either location
__all__ = ['Dynamics', 'ContingencyBuilder', 'SimAction', 'TS']


class SimAction(str, Enum):
    """Enumeration of standard simulation actions to prevent magic string errors."""
    FAULT_3PB = "FAULT 3PB SOLID"
    CLEAR_FAULT = "CLEARFAULT"
    OPEN = "OPEN"
    CLOSE = "CLOSE"

class ContingencyBuilder:
    """
    Fluent builder for Transient Stability (TS) contingencies.
    
    Constructs a timeline of events to be simulated.
    """

    def __init__(self, name: str, runtime: float = 10.0):
        self.name = name
        self.runtime = runtime
        self._current_time = 0.0
        self._events: List[Tuple[float, str, str, str]] = []

    def at(self, t: float) -> 'ContingencyBuilder':
        """Sets the current time cursor for subsequent events."""
        if t < 0:
            raise ValueError(f"Time cannot be negative: {t}")
        self._current_time = t
        return self

    def add_event(self, obj_type: str, who: str, action: Union[str, SimAction]) -> 'ContingencyBuilder':
        """Generic method to add an event at the current time cursor."""
        act_str = action.value if isinstance(action, SimAction) else str(action)
        self._events.append((self._current_time, obj_type, who, act_str))
        return self

    def fault_bus(self, bus: Any) -> 'ContingencyBuilder':
        return self.add_event("Bus", str(bus), SimAction.FAULT_3PB)

    def clear_fault(self, bus: Any) -> 'ContingencyBuilder':
        return self.add_event("Bus", str(bus), SimAction.CLEAR_FAULT)

    def trip_gen(self, bus: Any, gid: str = "1") -> 'ContingencyBuilder':
        return self.add_event("Gen", f"{bus} '{gid}'", SimAction.OPEN)

    def trip_branch(self, f_bus: Any, t_bus: Any, ckt: str = "1") -> 'ContingencyBuilder':
        return self.add_event("Branch", f"{f_bus} {t_bus} '{ckt}'", SimAction.OPEN)

    def to_dataframes(self) -> Tuple[DataFrame, DataFrame]:
        """
        Generates DataFrames matching the ESA GObject schemas.
        
        Returns:
            Tuple[DataFrame, DataFrame]: (Contingency Definition, Element Definitions)
        """
        # 1. Contingency Header
        ctg_df = DataFrame({
            'TSCTGName': [self.name],
            'StartTime': [0.0],
            'EndTime': [self.runtime],
            'CTGSkip': ['NO']
        })

        # 2. Elements
        if not self._events:
            return ctg_df, DataFrame()

        # Vectorized list creation is generally fast enough here
        ele_rows = [
            {
                'TSCTGName': self.name,
                'TSEventString': f"{action} {obj_type} {who}",
                'TSTimeInSeconds': t,
                'WhoAmI': f"{obj_type} {who}",
                'TSTimeInCycles': t * 60.0,
            }
            for t, obj_type, who, action in self._events
        ]
        
        return ctg_df, DataFrame(ele_rows)


class Dynamics(Indexable):
    """
    Transient stability simulation application manager.
    
    Handles contingency definition, simulation execution, and result retrieval.
    """

    def __init__(self):
        super().__init__()
        self.runtime: float = 5.0
        self._pending_ctgs: Dict[str, ContingencyBuilder] = {}
        self._watch_fields: Dict[Any, List[str]] = {}

    def watch(self, gtype: Type[GObject], fields: List[Any]) -> 'Dynamics':
        """
        Register fields to record during simulation for a specific object type.

        Args:
            gtype: The GObject type to watch (e.g., Gen, Bus, Branch)
            fields: List of TS field constants or field name strings
                   Example: [TS.Gen.P, TS.Gen.W] or ["TSGenP", "TSGenW"]

        Returns:
            Self for method chaining

        Example:
            >>> wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])
            >>> wb.dyn.watch(Bus, [TS.Bus.VPU, TS.Bus.Freq])
        """
        # Convert TSField objects to their string names
        field_names = [str(f) for f in fields]
        self._watch_fields[gtype] = field_names
        return self

    def contingency(self, name: str) -> ContingencyBuilder:
        """Start building a new contingency."""
        builder = ContingencyBuilder(name, self.runtime)
        self._pending_ctgs[name] = builder
        return builder

    def bus_fault(self, name: str, bus: Any, fault_time: float = 1.0, duration: float = 0.0833) -> None:
        """Helper to quickly define a bus fault contingency."""
        (self.contingency(name)
             .at(fault_time).fault_bus(bus)
             .at(fault_time + duration).clear_fault(bus))

    def _prepare_environment(self) -> List[str]:
        """
        Configures the ESA environment for simulation.
        
        IMPORTANT: Must be called BEFORE TSSolve to capture results.
        """
        fields = []
        for gtype, flds in self._watch_fields.items():
            # Enable storage for this specific object type
            self.esa.TSResultStorageSetAll(object=gtype.TYPE, value=True)
            
            # Retrieve ObjectIDs for the requested types
            objs = self[gtype, ['ObjectID']]
            
            if objs is not None and not objs.empty:
                # Filter out NaNs and ensure unique IDs
                valid_ids = objs['ObjectID'].dropna().unique()
                for oid in valid_ids:
                    # Format: "ObjectID | FieldName"
                    fields.extend(f"{oid} | {f}" for f in flds)
        
        return fields

    def upload_contingency(self, name: str) -> None:
        """Compiles and uploads a pending contingency to the simulation engine."""
        if name not in self._pending_ctgs:
            raise ValueError(f"Contingency '{name}' not found in pending list.")
        
        builder = self._pending_ctgs.pop(name)
        builder.runtime = self.runtime
        
        ctg_df, ele_df = builder.to_dataframes()
        
        # 1. Create the Contingency Object
        self[TSContingency] = ctg_df
        
        # 2. Create the Element Objects (if any)
        if not ele_df.empty:
            self[TSContingencyElement] = ele_df
        
        logger.info(f"Uploaded contingency: {name} with {len(ele_df)} events.")

    def _solve_single_contingency(self, ctg_name: str, retrieval_fields: List[str]) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """Executes simulation for a single contingency and retrieves raw results."""
        self.esa.TSSolve(ctg_name)
        return self.esa.TSGetContingencyResults(ctg_name, retrieval_fields)

    def _process_results(self, meta: DataFrame, df: DataFrame, ctg_name: str) -> Tuple[DataFrame, DataFrame]:
        """Helper to clean and format raw simulation results."""
        # Set index
        df = df.set_index("time")
        
        # Map headers
        col_map = {int(k): v for k, v in meta['ColHeader'].to_dict().items()}
        existing_cols = [c for c in col_map if c in df.columns]
        
        if not existing_cols:
            return DataFrame(), DataFrame()

        # Rename and cast
        df_processed = (df[existing_cols]
                        .rename(columns=col_map)
                        .astype(np.float32))
        
        # Clean metadata
        meta = meta.rename(columns={
            'ObjectType': 'Object', 
            'PrimaryKey': 'ID-A', 
            'SecondaryKey': 'ID-B', 
            'VariableName': 'Metric'
        })
        meta["Contingency"] = ctg_name
        
        return meta, df_processed

    def solve(self, ctgs: Union[str, List[str]]) -> Tuple[DataFrame, DataFrame]:
        """
        Runs the simulation for the specified contingencies.

        Args:
            ctgs: A single contingency name or a list of names.

        Returns:
            Tuple[DataFrame, DataFrame]: (Metadata, Time-Series Data)
        """
        ctgs_to_solve = [ctgs] if isinstance(ctgs, str) else list(ctgs)
        
        # 1. Upload any pending contingencies
        for ctg in ctgs_to_solve:
            if ctg in self._pending_ctgs:
                self.upload_contingency(ctg)
        
        # 2. Configure Storage & Build Fields
        # Crucial: Must be done before TSInitialize
        retrieval_fields = self._prepare_environment()

        if not retrieval_fields:
            logger.warning("No fields watched. Simulation will run but no results will be retrieved.")

        # 3. Initialize Simulation
        self.esa.TSAutoCorrect()
        self.esa.TSInitialize()

        all_meta_frames = []
        all_data_frames = {}

        # 4. Run Simulation Loop
        for ctg in ctgs_to_solve:
            logger.info(f"Solving contingency: {ctg}")
            meta, df = self._solve_single_contingency(ctg, retrieval_fields)
            
            if meta is None or df is None or df.empty:
                logger.warning(f"No results returned for contingency: {ctg}")
                continue

            meta, df = self._process_results(meta, df, ctg)
            if not df.empty:
                all_data_frames[ctg] = df
                all_meta_frames.append(meta)

        if not all_meta_frames:
            return DataFrame(), DataFrame()
        
        # 5. Concatenate Results
        final_meta = concat(all_meta_frames, axis=0, ignore_index=True).set_index('ColHeader')
        
        # keys=... creates a MultiIndex on columns (Contingency, Field)
        final_data = concat(all_data_frames.values(), axis=1, keys=all_data_frames.keys()).sort_index(axis=1)

        return final_meta, final_data

    def plot(self, meta: DataFrame, df: DataFrame, xlim: Optional[Tuple[float, float]] = None, **kwargs):
        """
        Plots simulation results grouped by Object and Metric.

        Args:
            meta: Metadata DataFrame returned by solve().
            df: Time-series DataFrame returned by solve().
            xlim: Optional tuple (min, max) for x-axis limits.
            **kwargs: Arguments passed to plt.subplots().
        """
        if meta.empty or df.empty:
            logger.warning("No results to plot.")
            return

        grouped = meta.groupby(['Object', 'Metric'])
        n_groups = len(grouped)
        
        if n_groups == 0:
            logger.warning("No data groups found to plot.")
            return

        if xlim is None:
            xlim = (df.index.min(), df.index.max())

        # Intelligent figure sizing
        fig_height = max(n_groups * 3.0, 5)
        fig, axes = plt.subplots(n_groups, 1, sharex=True,
                                 figsize=(10, fig_height),
                                 squeeze=False, **kwargs)
        axes_flat = axes.flatten()

        for ax, ((obj, metric), grp) in zip(axes_flat, grouped):
            # Iterate through contingencies (Top level of MultiIndex columns)
            ctg_list = df.columns.get_level_values(0).unique()
            
            for ctg in ctg_list:
                # Subset data for this contingency
                ctg_data = df[ctg]
                
                # Find columns in this contingency that match the current group (Object/Metric)
                # matching_cols are the specific "ColHeader" strings (e.g., "Bus 5 | Volt")
                matching_cols = grp.index.intersection(ctg_data.columns)
                
                for col in matching_cols:
                    # Construct label safely
                    id_a = grp.at[col, 'ID-A']
                    id_b = grp.at[col, 'ID-B'] if 'ID-B' in grp.columns else None
                    
                    # Handle NaNs purely for string formatting
                    id_a_str = str(id_a) if id_a is not None and str(id_a).lower() != 'nan' else ""
                    id_b_str = str(id_b) if id_b is not None and str(id_b).lower() != 'nan' else ""
                    
                    label_parts = [p for p in [id_a_str, id_b_str] if p]
                    lbl = " ".join(label_parts)
                    
                    plot_label = f"{ctg} | {lbl}" if lbl else ctg
                    ax.plot(ctg_data.index, ctg_data[col], label=plot_label, linewidth=1.5)

            ax.set_ylabel(f"{obj}\n{metric}", fontsize=10, fontweight='bold')
            ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.minorticks_on()
            
            if xlim:
                ax.set_xlim(xlim)

        axes_flat[-1].set_xlabel("Time (s)", fontsize=10, fontweight='bold')
        plt.tight_layout(pad=2.0)
        plt.show()