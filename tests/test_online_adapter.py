"""
Independent script to validate Adapter functionality against a live PowerWorld case.
This script connects to a PowerWorld Simulator instance using the provided case file
and attempts to execute a wide range of Adapter methods to verify functionality.

Usage:
    python test_online_adapter.py "C:\\Path\\To\\Case.pwb"
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# Ensure gridwb can be imported if running from tests directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from gridwb.indextool import IndexTool
    from gridwb.adapter import Adapter
    from gridwb.grid.components import Bus, Gen, Load, Branch, Contingency
except ImportError:
    print("Error: Could not import gridwb packages. Please ensure the package is in your Python path.")
    sys.exit(1)


@pytest.fixture(scope="module")
def adapter_instance():
    case_path = os.environ.get("SAW_TEST_CASE")
    if not case_path or not os.path.exists(case_path):
        pytest.skip("SAW_TEST_CASE environment variable not set or file not found.")

    print(f"\nConnecting to PowerWorld with case: {case_path}")
    # IndexTool handles SAW creation internally
    io = IndexTool(case_path)
    io.open()
    adapter = Adapter(io)
    yield adapter
    print("\nClosing case and exiting PowerWorld...")
    adapter.close()


@pytest.fixture
def temp_file():
    files = []

    def _create(suffix):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.close()
        files.append(tf.name)
        return tf.name

    yield _create
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


class TestOnlineAdapter:
    # -------------------------------------------------------------------------
    # Simulation Control & File Operations
    # -------------------------------------------------------------------------

    def test_control_flow(self, adapter_instance, temp_file):
        adapter_instance.reset()
        # Note: adapter.solve() calls self.io.pflow(), ensure IndexTool has pflow or adapter is updated
        try:
            adapter_instance.solve()
        except AttributeError:
            # Fallback if io.pflow doesn't exist yet
            adapter_instance.esa.SolvePowerFlow()
        
        tmp_pwb = temp_file(".pwb")
        adapter_instance.save(tmp_pwb)
        assert os.path.exists(tmp_pwb)
        
        adapter_instance.log("Adapter Test Message")
        adapter_instance.command('LogAdd("Command Test");')
        
        adapter_instance.mode("EDIT")
        adapter_instance.mode("RUN")

    def test_file_ops(self, adapter_instance, temp_file):
        tmp_aux = temp_file(".aux")
        # Create a dummy AUX file
        with open(tmp_aux, 'w') as f:
            f.write('DATA (Bus, [BusNum, BusName]) { 1 "Bus 1" }')
        adapter_instance.load_aux(tmp_aux)
        
        tmp_script = temp_file(".aux")
        with open(tmp_script, 'w') as f:
            f.write('SCRIPT { LogAdd("Script Test"); }')
        adapter_instance.load_script(tmp_script)

    # -------------------------------------------------------------------------
    # Data Retrieval
    # -------------------------------------------------------------------------

    def test_retrieval(self, adapter_instance):
        # Voltages
        v_complex = adapter_instance.voltages(pu=True, complex=True)
        assert len(v_complex) > 0
        
        v_mag, v_ang = adapter_instance.voltages(pu=True, complex=False)
        assert len(v_mag) == len(v_ang)
        
        # Components
        gens = adapter_instance.generations()
        assert not gens.empty
        
        loads = adapter_instance.loads()
        assert not loads.empty
        
        shunts = adapter_instance.shunts()
        # Shunts might be empty depending on case, but call should succeed
        
        lines = adapter_instance.lines()
        assert not lines.empty
        
        xfmrs = adapter_instance.transformers()
        # Transformers might be empty
        
        areas = adapter_instance.areas()
        assert not areas.empty
        
        zones = adapter_instance.zones()
        assert not zones.empty
        
        fields = adapter_instance.get_fields("Bus")
        assert not fields.empty

    # -------------------------------------------------------------------------
    # Modification
    # -------------------------------------------------------------------------

    def test_modification(self, adapter_instance):
        # Set voltages
        v = adapter_instance.voltages(pu=True, complex=True)
        adapter_instance.set_voltages(v)
        
        # Branch operations
        lines = adapter_instance.lines()
        if not lines.empty:
            l = lines.iloc[0]
            adapter_instance.open_branch(l['BusNum'], l['BusNum:1'], l['LineCircuit'])
            adapter_instance.close_branch(l['BusNum'], l['BusNum:1'], l['LineCircuit'])
            
        # Gen operations
        gens = adapter_instance.generations()
        if not gens.empty:
            g = gens.iloc[0]
            # Note: Adapter.generations() returns specific fields, need to ensure we have keys if needed
            # IndexTool returns keys by default in __getitem__ logic if we used that, 
            # but adapter.generations() uses specific list. 
            # We'll fetch keys from io directly to be safe for the set_gen call.
            g_keys = adapter_instance.io[Gen].iloc[0]
            adapter_instance.set_gen(g_keys['BusNum'], g_keys['GenID'], mw=g['GenMW'], status="Closed")
            
        # Load operations
        loads = adapter_instance.loads()
        if not loads.empty:
            l = loads.iloc[0]
            l_keys = adapter_instance.io[Load].iloc[0]
            adapter_instance.set_load(l_keys['BusNum'], l_keys['LoadID'], mw=l['LoadMW'], status="Closed")
            
        adapter_instance.scale_load(1.0)
        adapter_instance.scale_gen(1.0)
        
        # Create/Delete
        # Use a high bus number to avoid conflicts
        adapter_instance.create("Load", BusNum=1, LoadID="99", LoadMW=5.0)
        adapter_instance.delete("Load", "LoadID = '99'")
        
        # Select/Unselect
        adapter_instance.select("Bus", "BusNum < 10")
        adapter_instance.unselect("Bus")
        
        # Send to Excel (might fail if Excel not installed, wrap)
        try:
            adapter_instance.send_to_excel("Bus", ["BusNum", "BusName"])
        except Exception: pass

    # -------------------------------------------------------------------------
    # Advanced Topology & Switching
    # -------------------------------------------------------------------------

    def test_topology(self, adapter_instance):
        # Energize/Deenergize
        # Need a valid object. Bus 1 is usually safe in test cases.
        adapter_instance.deenergize("Bus", "[1]")
        adapter_instance.energize("Bus", "[1]")
        
        adapter_instance.radial_paths()
        
        dist = adapter_instance.path_distance("[BUS 1]")
        assert dist is not None
        
        # Network cut requires selected branches
        adapter_instance.select("Branch", "BusNum = 1")
        adapter_instance.network_cut("[BUS 1]", branch_filter="SELECTED")

    # -------------------------------------------------------------------------
    # Difference Flows
    # -------------------------------------------------------------------------

    def test_diff_flows(self, adapter_instance):
        adapter_instance.set_as_base_case()
        adapter_instance.diff_mode("DIFFERENCE")
        adapter_instance.diff_mode("PRESENT")

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def test_analysis(self, adapter_instance, temp_file):
        # Contingency
        adapter_instance.auto_insert_contingencies()
        
        # Run first one found
        ctgs = adapter_instance.io[Contingency]
        if not ctgs.empty:
            c_name = ctgs.iloc[0]['CTGLabel']
            adapter_instance.run_contingency(c_name)
        
        adapter_instance.solve_contingencies()
        
        # Violations
        viols = adapter_instance.violations()
        assert isinstance(viols, pd.DataFrame)
        
        # Mismatches
        mis = adapter_instance.mismatches()
        assert not mis.empty
        
        # Islands
        isl = adapter_instance.islands()
        assert isl is not None
        
        # Save Image
        tmp_img = temp_file(".jpg")
        # Need an open oneline.
        # adapter_instance.save_image(tmp_img, "OnelineName") 
        
        adapter_instance.refresh_onelines()

    # -------------------------------------------------------------------------
    # Sensitivity & Faults
    # -------------------------------------------------------------------------

    def test_sensitivity_faults(self, adapter_instance):
        # PTDF
        areas = adapter_instance.areas()
        if len(areas) >= 2:
            s = f'[AREA {areas.iloc[0]["AreaNum"]}]'
            b = f'[AREA {areas.iloc[1]["AreaNum"]}]'
            adapter_instance.ptdf(s, b)
            
        # LODF
        lines = adapter_instance.lines()
        if not lines.empty:
            l = lines.iloc[0]
            br = f'[BRANCH {l["BusNum"]} {l["BusNum:1"]} "{l["LineCircuit"]}"]'
            adapter_instance.lodf(br)
            
        # Fault
        adapter_instance.fault(1)
        adapter_instance.clear_fault()
        
        # Shortest Path
        buses = adapter_instance.io[Bus]
        if len(buses) >= 2:
            adapter_instance.shortest_path(buses.iloc[0]['BusNum'], buses.iloc[1]['BusNum'])

    # -------------------------------------------------------------------------
    # Advanced Analysis
    # -------------------------------------------------------------------------

    def test_advanced_analysis(self, adapter_instance):
        # PV - Needs injection groups, skipping actual run
        # adapter_instance.run_pv(source, sink)
        
        # QV
        adapter_instance.run_qv()
        
        # ATC
        areas = adapter_instance.areas()
        if len(areas) >= 2:
            s = f'[AREA {areas.iloc[0]["AreaNum"]}]'
            b = f'[AREA {areas.iloc[1]["AreaNum"]}]'
            adapter_instance.calculate_atc(s, b)
            
        # GIC
        adapter_instance.calculate_gic(1.0, 90.0)
        
        # OPF
        adapter_instance.solve_opf()


if __name__ == "__main__":
    # Default case path if not provided
    default_case = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Cases\Hawaii 37\Hawaii40_20231026.pwb"

    if len(sys.argv) > 1:
        case_arg = sys.argv[1]
    else:
        case_arg = default_case

    os.environ["SAW_TEST_CASE"] = case_arg

    # Run pytest on this file
    sys.exit(pytest.main(["-v", __file__]))
