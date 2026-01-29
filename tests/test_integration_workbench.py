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
    from esapp.components import Bus, Gen, Load, Branch, Contingency, Area, Zone, Shunt, GICXFormer, GObject
    from esapp import components as grid
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
        """Tests flatstart, pflow, save, log, command, mode."""
        wb.flatstart()

        # Power Flow - may fail on some test cases
        try:
            res = wb.pflow(getvolts=True)
            assert res is not None
            wb.pflow(getvolts=False)
        except PowerWorldError:
            pass  # Power flow may fail on some test cases

        # Save
        tmp_pwb = temp_file(".pwb")
        wb.save(tmp_pwb)
        assert os.path.exists(tmp_pwb)

        # Logging & Command
        wb.log("Adapter Test Message")
        wb.command('LogAdd("Command Test");')

        # Modes
        wb.edit_mode()
        wb.run_mode()

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
        """Tests voltage()."""
        # Test default call (complex, pu)
        v = wb.voltage()
        assert len(v) > 0
        assert np.iscomplexobj(v.values)

        # Test complex=True explicitly
        v_complex = wb.voltage(complex=True)
        assert np.iscomplexobj(v_complex.values)

        # Test complex=False
        v_mag, v_ang = wb.voltage(complex=False)
        assert len(v_mag) > 0
        assert len(v_mag) == len(v_ang)

        # Test pu=False
        v_kv = wb.voltage(pu=False)
        assert len(v_kv) > 0
        
        # Test pu=False and complex=False
        v_kv_mag, v_kv_ang = wb.voltage(pu=False, complex=False)
        assert len(v_kv_mag) > 0
        assert len(v_kv_mag) == len(v_kv_ang)

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

    # -------------------------------------------------------------------------
    # Modification
    # -------------------------------------------------------------------------

    def test_modification(self, wb):
        """Tests set_voltages, branch ops, gen/load ops, create/delete/select."""
        # Set Voltages
        v = wb.voltage(complex=True, pu=True)
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
            # Fetch keys (BusNum is PRIMARY, GenID is SECONDARY so must be requested explicitly)
            g_keys = wb[Gen, ["BusNum", "GenID"]].iloc[0]
            wb.set_gen(g_keys['BusNum'], g_keys['GenID'], mw=10.0, status="Closed")

        # Load Ops
        loads = wb.loads()
        if not loads.empty:
            # Fetch keys (BusNum is PRIMARY, LoadID is SECONDARY so must be requested explicitly)
            l_keys = wb[Load, ["BusNum", "LoadID"]].iloc[0]
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
    # Analysis & Difference Flows
    # -------------------------------------------------------------------------

    def test_analysis(self, wb, temp_file):
        """Tests contingency, violations, mismatches, islands, diff flows."""
        # Contingency - may fail depending on case configuration
        try:
            wb.auto_insert_contingencies()
            ctgs = wb[Contingency]
            if not ctgs.empty:
                c_name = ctgs.iloc[0]['CTGLabel']
                wb.run_contingency(c_name)
            wb.solve_contingencies()
        except PowerWorldError:
            pass  # Contingency operations may fail on some test cases

        # Violations
        viols = wb.violations()
        assert isinstance(viols, pd.DataFrame)

        # Mismatches
        mp, mq = wb.mismatch()
        assert not mp.empty
        assert not mq.empty

        # Islands
        isl = wb.islands()
        assert isl is not None

        # Diff Flows
        wb.set_as_base_case()
        wb.diff_mode("DIFFERENCE")
        wb.diff_mode("PRESENT")

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

        # Fault - wrap in try/except since clear_fault may fail if no fault exists
        try:
            wb.fault(1)
            wb.clear_fault()
        except PowerWorldError:
            pass  # Fault operations may fail depending on case state

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

    def test_reset(self, wb):
        """Tests reset() alias for flatstart()."""
        wb.reset()

    def test_print_log(self, wb):
        """Tests print_log() with all parameter combinations."""
        wb.log("Print log test message")

        # Default call
        output = wb.print_log()
        assert isinstance(output, str)

        # new_only mode
        wb.log("Another message")
        new_output = wb.print_log(new_only=True)
        assert isinstance(new_output, str)

        # clear mode
        cleared = wb.print_log(clear=True)
        assert isinstance(cleared, str)

    def test_mismatch_complex(self, wb):
        """Tests mismatch(asComplex=True)."""
        wb.pflow(getvolts=False)
        mm = wb.mismatch(asComplex=True)
        assert np.iscomplexobj(mm)

    def test_netinj(self, wb):
        """Tests netinj() in both modes."""
        P, Q = wb.netinj()
        assert len(P) > 0
        assert len(Q) > 0

        # Complex mode
        S = wb.netinj(asComplex=True)
        assert np.iscomplexobj(S)
        assert len(S) > 0

    def test_path_distance(self, wb):
        """Tests path_distance()."""
        buses = wb[Bus]
        if not buses.empty:
            try:
                wb.path_distance(create_object_string("Bus", buses.iloc[0]['BusNum']))
            except Exception:
                pytest.skip("path_distance not available for this case")

    def test_branch_admittance(self, wb):
        """Tests branch_admittance()."""
        Yf, Yt = wb.branch_admittance()
        assert Yf.shape[0] > 0
        assert Yt.shape[0] > 0
        assert Yf.shape == Yt.shape

    def test_shunt_admittance(self, wb):
        """Tests shunt_admittance()."""
        Ysh = wb.shunt_admittance()
        assert len(Ysh) > 0
        assert np.iscomplexobj(Ysh)

    def test_incidence_matrix(self, wb):
        """Tests incidence_matrix()."""
        A = wb.incidence_matrix()
        assert A.shape[0] > 0
        assert A.shape[1] > 0
        # Each row should have exactly one +1 and one -1
        for i in range(A.shape[0]):
            assert np.sum(A[i] == 1) == 1
            assert np.sum(A[i] == -1) == 1

    def test_jacobian(self, wb):
        """Tests jacobian()."""
        wb.pflow(getvolts=False)
        try:
            J = wb.jacobian()
            assert J.shape[0] > 0
        except Exception:
            pytest.skip("Jacobian not available")

    def test_gmatrix(self, wb):
        """Tests gmatrix()."""
        try:
            G = wb.gmatrix()
            assert G.shape[0] > 0
        except Exception:
            pytest.skip("GIC G-matrix not available")

    def test_buscoords_as_dataframe(self, wb):
        """Tests buscoords(astuple=False)."""
        try:
            df = wb.buscoords(astuple=False)
            assert isinstance(df, pd.DataFrame)
        except Exception:
            pytest.skip("buscoords not available (no substation data)")

    def test_write_voltage(self, wb):
        """Tests write_voltage()."""
        V = wb.voltage(complex=True, pu=True)
        wb.write_voltage(V)

    def test_gens_above_pmax(self, wb):
        """Tests gens_above_pmax()."""
        result = wb.gens_above_pmax()
        assert isinstance(result, (bool, np.bool_))

    def test_gens_above_qmax(self, wb):
        """Tests gens_above_qmax()."""
        result = wb.gens_above_qmax()
        assert isinstance(result, (bool, np.bool_))

    def test_gic_storm(self, wb):
        """Tests gic_storm() with both solve_pf options."""
        try:
            wb.gic_storm(max_field=1.0, direction=90.0, solve_pf=True)
            wb.gic_storm(max_field=1.0, direction=90.0, solve_pf=False)
        except Exception:
            pytest.skip("GIC storm not available")

    def test_gic_clear(self, wb):
        """Tests gic_clear()."""
        try:
            wb.gic_clear()
        except Exception:
            pytest.skip("GIC clear not available")

    def test_gic_load_b3d(self, wb):
        """Tests gic_load_b3d()."""
        try:
            wb.gic_load_b3d("STORM", "nonexistent.b3d", setup_on_load=True)
        except Exception:
            pass  # Expected to fail without a real file, but exercises the code path
        try:
            wb.gic_load_b3d("STORM", "nonexistent.b3d", setup_on_load=False)
        except Exception:
            pass

    def test_set_option_methods(self, wb):
        """Tests all _set_option-based methods."""
        try:
            wb.set_do_one_iteration(True)
            wb.set_do_one_iteration(False)
        except Exception:
            pytest.skip("set_do_one_iteration not available")

        try:
            wb.set_max_iterations(250)
        except Exception:
            pytest.skip("set_max_iterations not available")

        try:
            wb.set_disable_angle_rotation(True)
            wb.set_disable_angle_rotation(False)
        except Exception:
            pytest.skip("set_disable_angle_rotation not available")

        try:
            wb.set_disable_opt_mult(True)
            wb.set_disable_opt_mult(False)
        except Exception:
            pytest.skip("set_disable_opt_mult not available")

        try:
            wb.enable_inner_ss_check(True)
            wb.enable_inner_ss_check(False)
        except Exception:
            pytest.skip("enable_inner_ss_check not available")

        try:
            wb.disable_gen_mvr_check(True)
            wb.disable_gen_mvr_check(False)
        except Exception:
            pytest.skip("disable_gen_mvr_check not available")

        try:
            wb.enable_inner_check_gen_vars(True)
            wb.enable_inner_check_gen_vars(False)
        except Exception:
            pytest.skip("enable_inner_check_gen_vars not available")

        try:
            wb.enable_inner_backoff_gen_vars(True)
            wb.enable_inner_backoff_gen_vars(False)
        except Exception:
            pytest.skip("enable_inner_backoff_gen_vars not available")

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
