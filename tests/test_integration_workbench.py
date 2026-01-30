"""
Integration tests for GridWorkBench functionality with live PowerWorld data.

Tests workbench-level facade methods. Domain-specific tests for Statics,
Network, GIC, and Dynamics are in test_integration_analysis.py.

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py

USAGE:
    pytest tests/test_integration_workbench.py -v
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

CRASH_PRONE_COMPONENTS = []

@pytest.fixture(scope="module")
def wb(saw_session):
    """
    Wraps the session-scoped SAW instance in a GridWorkBench object.
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

        try:
            res = wb.pflow(getvolts=True)
            assert res is not None
            wb.pflow(getvolts=False)
        except PowerWorldError:
            pass

        tmp_pwb = temp_file(".pwb")
        wb.save(tmp_pwb)
        assert os.path.exists(tmp_pwb)

        wb.log("Adapter Test Message")
        wb.command('LogAdd("Command Test");')

        wb.edit_mode()
        wb.run_mode()

    # -------------------------------------------------------------------------
    # Data Retrieval
    # -------------------------------------------------------------------------

    def test_voltage_retrieval(self, wb):
        """Tests voltage() delegation to statics."""
        v = wb.voltage()
        assert len(v) > 0
        assert np.iscomplexobj(v.values)

        v_complex = wb.voltage(complex=True)
        assert np.iscomplexobj(v_complex.values)

        v_mag, v_ang = wb.voltage(complex=False)
        assert len(v_mag) > 0
        assert len(v_mag) == len(v_ang)

        v_kv = wb.voltage(pu=False)
        assert len(v_kv) > 0

        v_kv_mag, v_kv_ang = wb.voltage(pu=False, complex=False)
        assert len(v_kv_mag) > 0

    def test_component_retrieval(self, wb):
        """Tests generations, loads, shunts, lines, transformers, areas, zones."""
        assert not wb.generations().empty
        assert not wb.loads().empty
        wb.shunts()
        wb.transformers()
        assert not wb.lines().empty
        assert not wb.areas().empty
        assert not wb.zones().empty

    # -------------------------------------------------------------------------
    # Modification
    # -------------------------------------------------------------------------

    def test_modification(self, wb):
        """Tests set_voltages, branch ops, gen/load ops, scaling."""
        v = wb.voltage(complex=True, pu=True)
        wb.set_voltages(v)

        lines = wb.lines()
        if not lines.empty:
            l = lines.iloc[0]
            wb.open_branch(l['BusNum'], l['BusNum:1'], l['LineCircuit'])
            wb.close_branch(l['BusNum'], l['BusNum:1'], l['LineCircuit'])

        gens = wb.generations()
        if not gens.empty:
            g_keys = wb[Gen, ["BusNum", "GenID"]].iloc[0]
            wb.set_gen(g_keys['BusNum'], g_keys['GenID'], mw=10.0, status="Closed")

        loads = wb.loads()
        if not loads.empty:
            l_keys = wb[Load, ["BusNum", "LoadID"]].iloc[0]
            wb.set_load(l_keys['BusNum'], l_keys['LoadID'], mw=5.0, status="Closed")

        wb.scale_load(1.0)
        wb.scale_gen(1.0)

    # -------------------------------------------------------------------------
    # Analysis & Difference Flows
    # -------------------------------------------------------------------------

    def test_analysis(self, wb, temp_file):
        """Tests contingency, violations, mismatches, islands, diff flows."""

        viols = wb.violations()
        assert isinstance(viols, pd.DataFrame)

        mp, mq = wb.mismatch()
        assert not mp.empty
        assert not mq.empty

        isl = wb.islands()
        assert isl is not None

        wb.set_as_base_case()
        wb.diff_mode("DIFFERENCE")
        wb.diff_mode("PRESENT")

    # -------------------------------------------------------------------------
    # Sensitivity, Faults
    # -------------------------------------------------------------------------

    def test_sensitivity_faults(self, wb):
        """Tests ptdf, lodf, fault, shortest_path."""
        areas = wb.areas()
        if len(areas) >= 2:
            s = create_object_string("Area", areas.iloc[0]["AreaNum"])
            b = create_object_string("Area", areas.iloc[1]["AreaNum"])
            wb.ptdf(s, b)

        lines = wb.lines()
        if not lines.empty:
            l = lines.iloc[0]
            br = create_object_string("Branch", l["BusNum"], l["BusNum:1"], l["LineCircuit"])
            wb.lodf(br)

        try:
            wb.fault(1)
        except PowerWorldError:
            pass

        buses = wb[Bus]
        if len(buses) >= 2:
            wb.shortest_path(buses.iloc[0]['BusNum'], buses.iloc[1]['BusNum'])

    def test_advanced_analysis(self, wb):
        """Tests OPF, YBus."""
        wb.solve_opf()

        Y = wb.ybus()
        assert Y.shape[0] > 0

    def test_print_log(self, wb):
        """Tests print_log() with all parameter combinations."""
        wb.log("Print log test message")

        output = wb.print_log()
        assert isinstance(output, str)

        wb.log("Another message")
        new_output = wb.print_log(new_only=True)
        assert isinstance(new_output, str)

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
        """Tests branch_admittance() delegation."""
        Yf, Yt = wb.branch_admittance()
        assert Yf.shape[0] > 0
        assert Yt.shape[0] > 0
        assert Yf.shape == Yt.shape

    def test_jacobian(self, wb):
        """Tests jacobian() delegation."""
        wb.pflow(getvolts=False)
        try:
            J = wb.jacobian()
            assert J.shape[0] > 0
        except Exception:
            pytest.skip("Jacobian not available")

    def test_buscoords_as_dataframe(self, wb):
        """Tests buscoords(astuple=False) delegation."""
        try:
            df = wb.buscoords(astuple=False)
            assert isinstance(df, pd.DataFrame)
        except Exception:
            pytest.skip("buscoords not available (no substation data)")

    def test_location(self, wb):
        """Tests busmap, buscoords."""
        m = wb.busmap()
        assert not m.empty

        try:
            wb.buscoords()
        except Exception:
            pass


# -------------------------------------------------------------------------
# Consolidated Component Access Tests
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
        pytest.skip(f"Skipping {component_class.TYPE}: Known to cause SimAuto crashes.")

    try:
        df = wb[component_class]
    except SimAutoFeatureError as e:
        pytest.skip(f"Object type {component_class.TYPE} cannot be retrieved via SimAuto: {e.message}")
    except (PowerWorldError, COMError) as e:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            fields = component_class.keys if component_class.keys else ["ALL"]
            wb.esa.SaveObjectFields(tmp_path, component_class.TYPE, fields)
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
    sys.exit(pytest.main(["-v", __file__]))
