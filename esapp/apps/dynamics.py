import logging
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Tuple, Dict, Union, Optional, Any, Type
from pandas import DataFrame, concat

from ..indexable import Indexable
from ..gobject import GObject
# Import from grid (TSContingency, TSContingencyElement)
from esapp.grid import TSContingency, TSContingencyElement

# Configure logger
logger = logging.getLogger(__name__)

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

    def watch(self, gtype: Type[GObject], fields: List[str]) -> None:
        """Register fields to record during simulation for a specific object type."""
        self._watch_fields[gtype] = list(fields)

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
        # Enable storage for ALL objects.
        self.esa.TSResultStorageSetAll(object="ALL", value=True)
        
        fields = []
        for gtype, flds in self._watch_fields.items():
            # Retrieve ObjectIDs for the requested types
            objs = self[gtype, ['ObjectID'] + list(gtype.keys)]
            
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
        
        # Clear pending map to prevent double-uploading if called again
        # (Though we already popped inside upload_contingency, strict safety is good)
        keys_to_remove = [k for k in self._pending_ctgs if k in ctgs_to_solve]
        for k in keys_to_remove: 
            del self._pending_ctgs[k]

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

    def plot(self, meta: DataFrame, df: DataFrame, **kwargs):
        """
        Plots simulation results grouped by Object and Metric.

        Args:
            meta: Metadata DataFrame returned by solve().
            df: Time-series DataFrame returned by solve().
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

        # Intelligent figure sizing
        fig_height = max(n_groups * 2.5, 4)
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
                    ax.plot(ctg_data.index, ctg_data[col], label=plot_label)

            ax.set_ylabel(f"{obj}\n{metric}")
            ax.grid(True, alpha=0.3)

        axes_flat[-1].set_xlabel("Time (s)")
        plt.tight_layout(pad=1.5)
        plt.show()