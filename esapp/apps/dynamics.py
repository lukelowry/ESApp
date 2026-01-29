"""
Transient Stability Simulation Module
=====================================

This module provides a high-level interface for running transient stability
simulations in PowerWorld Simulator. It enables users to define contingencies,
execute simulations, and analyze results through a fluent API.

Classes
-------
Dynamics
    Main simulation manager for transient stability analysis. Handles
    contingency definition, simulation execution, and result retrieval.
ContingencyBuilder
    Fluent builder for constructing contingency event sequences with
    time-based actions (faults, trips, switching operations).
SimAction
    Enumeration of standard simulation action strings to prevent typos.

Key Features
------------
- Fluent contingency definition with method chaining
- Automatic result storage configuration for watched fields
- Multi-contingency batch simulation support
- Built-in plotting with grouped visualization by object type and metric
- Model inventory listing via ``list_models()``

Example
-------
Basic bus fault simulation::

    >>> from esapp import GridWorkBench
    >>> from esapp.components import TS, Gen, Bus
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

See Also
--------
esapp.components.TS : Transient stability field constants for IDE autocomplete.
esapp.components.TSContingency : PowerWorld contingency data object.
esapp.components.TSContingencyElement : PowerWorld contingency element data object.
"""
import logging
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Tuple, Dict, Union, Optional, Any, Type

from pandas import DataFrame, concat

from ..indexable import Indexable
from ..components import GObject
from ..components import TS, TSContingency, TSContingencyElement
from esapp.saw._helpers import get_temp_filepath

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

    Constructs a timeline of events to be simulated using method chaining.

    Parameters
    ----------
    name : str
        Unique name for the contingency.
    runtime : float, optional
        Simulation duration in seconds (default: 10.0).

    Attributes
    ----------
    name : str
        The contingency name.
    runtime : float
        Simulation end time in seconds.

    Example
    -------
    >>> builder = ContingencyBuilder("GenTrip", runtime=5.0)
    >>> builder.at(1.0).fault_bus("101").at(1.1).clear_fault("101")
    """

    def __init__(self, name: str, runtime: float = 10.0):
        self.name = name
        self.runtime = runtime
        self._current_time: float = 0.0
        self._events: List[Tuple[float, str, str, str]] = []

    def at(self, t: float) -> 'ContingencyBuilder':
        """
        Set the current time cursor for subsequent events.

        Parameters
        ----------
        t : float
            Time in seconds (must be non-negative).

        Returns
        -------
        ContingencyBuilder
            Self for method chaining.

        Raises
        ------
        ValueError
            If time is negative.
        """
        if t < 0:
            raise ValueError(f"Time cannot be negative: {t}")
        self._current_time = t
        return self

    def add_event(self, obj_type: str, who: str, action: Union[str, SimAction]) -> 'ContingencyBuilder':
        """
        Add a generic event at the current time cursor.

        Parameters
        ----------
        obj_type : str
            PowerWorld object type (e.g., "Bus", "Gen", "Branch").
        who : str
            Object identifier string.
        action : Union[str, SimAction]
            Action to perform (e.g., SimAction.OPEN or "OPEN").

        Returns
        -------
        ContingencyBuilder
            Self for method chaining.
        """
        act_str = action.value if isinstance(action, SimAction) else str(action)
        self._events.append((self._current_time, obj_type, who, act_str))
        return self

    def fault_bus(self, bus: Any) -> 'ContingencyBuilder':
        """Apply a 3-phase solid fault to a bus at the current time."""
        return self.add_event("Bus", str(bus), SimAction.FAULT_3PB)

    def clear_fault(self, bus: Any) -> 'ContingencyBuilder':
        """Clear the fault at a bus at the current time."""
        return self.add_event("Bus", str(bus), SimAction.CLEAR_FAULT)

    def trip_gen(self, bus: Any, gid: str = "1") -> 'ContingencyBuilder':
        """Trip (open) a generator at the current time."""
        return self.add_event("Gen", f"{bus} '{gid}'", SimAction.OPEN)

    def trip_branch(self, f_bus: Any, t_bus: Any, ckt: str = "1") -> 'ContingencyBuilder':
        """Trip (open) a branch at the current time."""
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
    Inherits from Indexable to provide DataFrame-like access to grid components.

    Attributes
    ----------
    runtime : float
        Default simulation duration in seconds (default: 5.0).

    Example
    -------
    >>> wb.dyn.runtime = 10.0
    >>> wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W])
    >>> wb.dyn.bus_fault("Fault1", "101", fault_time=1.0, duration=0.1)
    >>> meta, data = wb.dyn.solve("Fault1")
    """

    def __init__(self):
        super().__init__()
        self.runtime: float = 5.0
        self._pending_ctgs: Dict[str, ContingencyBuilder] = {}
        self._watch_fields: Dict[Type[GObject], List[str]] = {}

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

    def get_results(self, ctg: str, fields: List[str]) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """
        Retrieve results for a single contingency using TSGetResults.

        Parameters
        ----------
        ctg : str
            Contingency name.
        fields : List[str]
            List of fields/plots to retrieve (e.g., ["Bus 101 | TSBusVPU"]).

        Returns
        -------
        Tuple[Optional[DataFrame], Optional[DataFrame]]
            Tuple of (Metadata DataFrame, Data DataFrame), or (None, None)
            if no results are available.
        """
        result = self.esa.TSGetResults("SEPARATE", [ctg], fields)
        if result is None:
            return None, None
        return result

    def contingency(self, name: str) -> ContingencyBuilder:
        """
        Start building a new contingency.

        Parameters
        ----------
        name : str
            Unique name for the contingency.

        Returns
        -------
        ContingencyBuilder
            A builder instance for defining contingency events.
        """
        builder = ContingencyBuilder(name, self.runtime)
        self._pending_ctgs[name] = builder
        return builder

    def bus_fault(self, name: str, bus: Any, fault_time: float = 1.0, duration: float = 0.0833) -> None:
        """
        Define a simple bus fault contingency.

        Creates a contingency with a 3-phase solid fault applied and cleared.

        Parameters
        ----------
        name : str
            Unique name for the contingency.
        bus : Any
            Bus number or identifier to fault.
        fault_time : float, optional
            Time to apply the fault in seconds (default: 1.0).
        duration : float, optional
            Fault duration in seconds (default: 0.0833, ~5 cycles at 60Hz).
        """
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
        """
        Compile and upload a pending contingency to the simulation engine.

        Parameters
        ----------
        name : str
            Name of the contingency to upload (must exist in pending list).

        Raises
        ------
        ValueError
            If the contingency name is not found in the pending list.
        """
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
        """
        Execute simulation for a single contingency and retrieve raw results.

        Parameters
        ----------
        ctg_name : str
            Name of the contingency to solve.
        retrieval_fields : List[str]
            List of field specifications for result retrieval.

        Returns
        -------
        Tuple[Optional[DataFrame], Optional[DataFrame]]
            Tuple of (Metadata, Data) DataFrames.
        """
        self.esa.TSSolve(ctg_name)
        return self.get_results(ctg_name, retrieval_fields)

    def _process_results(self, meta: DataFrame, df: DataFrame, ctg_name: str) -> Tuple[DataFrame, DataFrame]:
        """
        Clean and format raw simulation results.

        Parameters
        ----------
        meta : DataFrame
            Raw metadata from TSGetResults.
        df : DataFrame
            Raw time-series data from TSGetResults.
        ctg_name : str
            Contingency name for labeling.

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            Tuple of (processed metadata, processed data) DataFrames.
            Returns empty DataFrames if no valid data is found.
        """
        if df is None or df.empty:
            return DataFrame(), DataFrame()

        # Set index (load_ts_csv_results ensures 'time' column exists and is normalized)
        if "time" in df.columns:
            df = df.set_index("time")

        # Filter columns that are in meta
        valid_headers = set(meta['ColHeader']) if 'ColHeader' in meta.columns else set()
        existing_cols = [c for c in df.columns if c in valid_headers]

        if not existing_cols:
            return DataFrame(), DataFrame()

        # Subset and cast
        df_processed = df[existing_cols].astype(np.float32)

        # Clean metadata
        meta = meta.rename(columns={
            'ObjectType': 'Object',
            'PrimaryKey': 'ID-A',
            'SecondaryKey': 'ID-B',
            'VariableName': 'Metric'
        })
        meta["Contingency"] = ctg_name

        return meta, df_processed

    def list_models(self, diff_case_modified_only: bool = False) -> DataFrame:
        """
        List transient stability models present in the case.

        Parses the AUX output from TSWriteModels to categorize dynamic models
        by type (Machine, Exciter, Governor, Stabilizer, etc.).

        Parameters
        ----------
        diff_case_modified_only : bool, optional
            If True, only list models modified relative to the difference
            case base (default: False).

        Returns
        -------
        DataFrame
            DataFrame with columns ['Category', 'Model', 'Object Type']
            sorted by Category and Model. Returns empty DataFrame if no
            models are found.
        """
        temp_path = get_temp_filepath(".aux")
        try:
            self.esa.TSWriteModels(temp_path, diff_case_modified_only)
            if not os.path.exists(temp_path):
                logger.warning("TSWriteModels did not create a file.")
                return DataFrame()
            
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Regex to find DATA headers: DATA (ObjType,
            object_types = re.findall(r'DATA\s*\(\s*([^,]+)\s*,', content, re.IGNORECASE)
            
            data = []
            for obj_type in object_types:
                obj_type = obj_type.strip()
                
                # Categorize
                category = "Other"
                model_name = obj_type
                
                if obj_type.startswith("MachineModel_"):
                    category = "Machine"
                    model_name = obj_type[13:]
                elif obj_type.startswith("Exciter_"):
                    category = "Exciter"
                    model_name = obj_type[8:]
                elif obj_type.startswith("Governor_"):
                    category = "Governor"
                    model_name = obj_type[9:]
                elif obj_type.startswith("Stabilizer_"):
                    category = "Stabilizer"
                    model_name = obj_type[11:]
                elif obj_type.startswith("PlantController_"):
                    category = "Plant Controller"
                    model_name = obj_type[16:]
                elif obj_type.startswith("RelayModel_"):
                    category = "Relay"
                    model_name = obj_type[11:]
                elif obj_type.startswith("LoadModel_"):
                    category = "Load Characteristic"
                    model_name = obj_type[10:]
                elif obj_type in ["Gen", "Load", "Bus", "Shunt", "Branch"]:
                    continue # Skip network objects
                
                data.append({"Category": category, "Model": model_name, "Object Type": obj_type})
            
            if not data:
                return DataFrame(columns=["Category", "Model", "Object Type"])
                
            return DataFrame(data).drop_duplicates().sort_values(["Category", "Model"]).reset_index(drop=True)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

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