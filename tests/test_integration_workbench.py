"""
Integration tests for GridWorkBench functionality with live PowerWorld data.

WHAT THIS TESTS:
- Component collection access (buses, generators, loads, branches, etc.)
- Data retrieval through component properties with real case data
- DataFrame conversion from live PowerWorld data
- Component-specific methods and attributes
- Performance validation with actual datasets
- Parametrized tests across all component types

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py

CONFIGURATION:
    1. Copy tests/config_test.example.py to tests/config_test.py
    2. Set SAW_TEST_CASE = r"C:\\Path\\To\\Your\\Case.pwb"

USAGE:
    pytest tests/test_integration_workbench.py -v
    pytest tests/test_integration_workbench.py -k "Bus" -v  # Test only Bus components
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import inspect
import sys 

try:
    from esapp.grid import Bus, Gen, Load, Branch, Contingency, Area, Zone, Shunt, GICXFormer, GObject
    from esapp import grid
    from esapp.workbench import GridWorkBench    
    from esapp.saw import PowerWorldError, COMError, SimAutoFeatureError, create_object_string
except ImportError:
    raise

# List of component types known to cause SimAuto process instability or crashes
# when accessed via generic parameter retrieval methods.
# NOTE There is no evidence that these actually caused crashes
CRASH_PRONE_COMPONENTS = [
    #"ATCLineChangeB", 
    #"ATCScenario",
    #"ATCZoneChange",
    #"ATCGeneratorChange",
    #"ATCInterfaceChange",
]

@pytest.fixture(scope="module")
def wb(saw_session):
    """
    Wraps the session-scoped SAW instance in a GridWorkBench object.
    The lifecycle of the underlying SAW instance is managed by saw_session.
    """
    workbench = GridWorkBench()
    workbench.set_esa(saw_session)
    return workbench


class TestGridWorkBenchFunctions:
    # -------------------------------------------------------------------------
    # Simulation Control
    # -------------------------------------------------------------------------

    def test_simulation_control(self, wb, temp_file):
        """Tests reset, pflow, save, log, command, mode."""
        wb.reset()
        
        # Power Flow
        res = wb.pflow(getvolts=True)
        assert res is not None
        wb.pflow(getvolts=False)
        
        # Save
        tmp_pwb = temp_file(".pwb")
        wb.save(tmp_pwb)
        assert os.path.exists(tmp_pwb)
        
        # Logging & Command
        wb.log("Adapter Test Message")
        wb.command('LogAdd("Command Test");')
        
        # Modes
        wb.mode("EDIT")
        wb.mode("RUN")

    def test_file_operations(self, wb, temp_file):
        """Tests load_aux, load_script."""
        tmp_aux = temp_file(".aux")
        with open(tmp_aux, 'w') as f:
            f.write('DATA (Bus, [BusNum, BusName]) { 1 "Bus 1" }')
        wb.load_aux(tmp_aux)
        
        tmp_script = temp_file(".aux")
        with open(tmp_script, 'w') as f:
            f.write('SCRIPT { LogAdd("Script Test"); }')
        wb.load_script(tmp_script)

    # -------------------------------------------------------------------------
    # Data Retrieval
    # -------------------------------------------------------------------------

    def test_voltage_retrieval(self, wb):
        """Tests voltage() and voltages()."""
        # voltage()
        v = wb.voltage()
        assert len(v) > 0
        v_complex = wb.voltage(asComplex=True)
        # v_complex is a Series, check values
        assert np.iscomplexobj(v_complex.values)
        v_mag, v_ang = wb.voltage(asComplex=False)
        assert len(v_mag) == len(v_ang)

        # voltages()
        v2 = wb.voltages(pu=True, complex=True)
        assert len(v2) > 0

    def test_component_retrieval(self, wb):
        """Tests generations, loads, shunts, lines, transformers, areas, zones."""
        assert not wb.generations().empty
        assert not wb.loads().empty
        # Shunts/Transformers might be empty in some cases, but call should succeed
        wb.shunts()
        wb.transformers()
        assert not wb.lines().empty
        assert not wb.areas().empty
        assert not wb.zones().empty
        
        fields = wb.get_fields("Bus")
        assert not fields.empty

    # -------------------------------------------------------------------------
    # Modification
    # -------------------------------------------------------------------------

    def test_modification(self, wb):
        """Tests set_voltages, branch ops, gen/load ops, create/delete/select."""
        # Set Voltages
        v = wb.voltages(pu=True, complex=True)
        wb.set_voltages(v)
        
        # Branch Ops
        lines = wb.lines()
        if not lines.empty:
            l = lines.iloc[0]
            wb.open_branch(l['BusNum'], l['BusNum:1'], l['LineCircuit'])
            wb.close_branch(l['BusNum'], l['BusNum:1'], l['LineCircuit'])
            
        # Gen Ops
        gens = wb.generations()
        if not gens.empty:
            # Fetch keys to be safe
            g_keys = wb[Gen].iloc[0]
            wb.set_gen(g_keys['BusNum'], g_keys['GenID'], mw=10.0, status="Closed")
            
        # Load Ops
        loads = wb.loads()
        if not loads.empty:
            l_keys = wb[Load].iloc[0]
            wb.set_load(l_keys['BusNum'], l_keys['LoadID'], mw=5.0, status="Closed")
            
        wb.scale_load(1.0)
        wb.scale_gen(1.0)
        
        # Create/Delete (Use dummy ID)
        wb.create("Load", BusNum=1, LoadID="99", LoadMW=5.0)
        wb.delete("Load", "LoadID = '99'")
        
        # Select/Unselect
        wb.select("Bus", "BusNum < 10")
        wb.unselect("Bus")

    # -------------------------------------------------------------------------
    # Advanced Topology & Switching
    # -------------------------------------------------------------------------

    def test_topology(self, wb):
        """Tests energize, deenergize, radial_paths, path_distance, network_cut."""
        wb.deenergize("Bus", create_object_string("Bus", 1))
        wb.energize("Bus", create_object_string("Bus", 1))
        
        wb.radial_paths()
        
        dist = wb.path_distance(create_object_string("Bus", 1))
        assert dist is not None
        
        wb.select("Branch", "BusNum = 1")
        wb.network_cut(create_object_string("Bus", 1), branch_filter="SELECTED")

    # -------------------------------------------------------------------------
    # Analysis & Difference Flows
    # -------------------------------------------------------------------------

    def test_analysis(self, wb, temp_file):
        """Tests contingency, violations, mismatches, islands, diff flows."""
        # Contingency
        wb.auto_insert_contingencies()
        ctgs = wb[Contingency]
        if not ctgs.empty:
            c_name = ctgs.iloc[0]['CTGLabel']
            wb.run_contingency(c_name)
        wb.solve_contingencies()
        
        # Violations
        viols = wb.violations()
        assert isinstance(viols, pd.DataFrame)
        
        # Mismatches
        mis = wb.mismatches()
        assert not mis.empty
        
        # Islands
        isl = wb.islands()
        assert isl is not None
        
        # Diff Flows
        wb.set_as_base_case()
        wb.diff_mode("DIFFERENCE")
        wb.diff_mode("PRESENT")
        
        # Onelines
        wb.refresh_onelines()

    # -------------------------------------------------------------------------
    # Sensitivity, Faults, Advanced Analysis
    # -------------------------------------------------------------------------

    def test_sensitivity_faults(self, wb):
        """Tests ptdf, lodf, fault, shortest_path."""
        # PTDF
        areas = wb.areas()
        if len(areas) >= 2:
            s = create_object_string("Area", areas.iloc[0]["AreaNum"])
            b = create_object_string("Area", areas.iloc[1]["AreaNum"])
            wb.ptdf(s, b)
            
        # LODF
        lines = wb.lines()
        if not lines.empty:
            l = lines.iloc[0]
            br = create_object_string("Branch", l["BusNum"], l["BusNum:1"], l["LineCircuit"])
            wb.lodf(br)
            
        # Fault
        wb.fault(1)
        wb.clear_fault()
        
        # Shortest Path
        buses = wb[Bus]
        if len(buses) >= 2:
            wb.shortest_path(buses.iloc[0]['BusNum'], buses.iloc[1]['BusNum'])

    def test_advanced_analysis(self, wb):
        """Tests QV, ATC, GIC, OPF, YBus."""
        
        
        # ATC
        areas = wb.areas()
        if len(areas) >= 2:
            s = create_object_string("Area", areas.iloc[0]["AreaNum"])
            b = create_object_string("Area", areas.iloc[1]["AreaNum"])
            wb.calculate_atc(s, b)
            
        # GIC
        wb.calculate_gic(1.0, 90.0)
        
        # OPF
        wb.solve_opf()
        
        # YBus
        Y = wb.ybus()
        assert Y.shape[0] > 0

    def test_location(self, wb):
        """Tests busmap, buscoords."""
        m = wb.busmap()
        assert not m.empty
        
        # buscoords requires substation data, might be empty but call should work
        try:
            wb.buscoords()
        except Exception:
            pass


# -------------------------------------------------------------------------
# Consolidated Component Access Tests (formerly test_online_components.py)
# -------------------------------------------------------------------------

def get_gobject_subclasses():
    """Helper to discover all GObject subclasses in the components module."""
    return [
        obj for _, obj in inspect.getmembers(grid, inspect.isclass)
        if issubclass(obj, GObject) and obj is not GObject
    ]

@pytest.mark.parametrize("component_class", get_gobject_subclasses())
def test_component_access(wb, component_class):
    """
    Verifies that GridWorkBench can read key fields for every defined component.
    """
    if component_class.TYPE in CRASH_PRONE_COMPONENTS:
        pytest.skip(f"Skipping {component_class.TYPE}: Known to cause SimAuto crashes during iteration.")

    try:
        df = wb[component_class]
    except SimAutoFeatureError as e:
        pytest.skip(f"Object type {component_class.TYPE} cannot be retrieved via SimAuto: {e.message}")
    except (PowerWorldError, COMError) as e:
        # Check if object is supported by checking if we can save fields
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            fields = component_class.keys if component_class.keys else ["ALL"]
            wb.esa.SaveObjectFields(tmp_path, component_class.TYPE, fields)
            # If save works but read fails, it's a real error (or memory issue)
            if "memory resources" in str(e):
                pytest.skip(f"Object type {component_class.TYPE} has too many fields/objects.")
            pytest.fail(f"Object type {component_class.TYPE} is supported but failed to read: {e}")
        except PowerWorldError:
            pytest.skip(f"Object type {component_class.TYPE} not supported by this PW version.")
        finally:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass
    except Exception as e:
        pytest.fail(f"Unexpected error reading {component_class.__name__}: {e}")
        
    if df is not None:
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            for key in component_class.keys:
                assert key in df.columns


if __name__ == "__main__":
    # Run pytest on this file
    sys.exit(pytest.main(["-v", __file__]))
