"""
Integration tests for the PowerWorld facade.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They test the top-level
PowerWorld facade: case I/O, simulation control, component retrieval,
data modification, delegation to sub-modules, and workbench-level static
analysis (power flow, voltage, Y-bus, Jacobian).

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

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
    from esapp.workbench import PowerWorld
    from esapp.saw import PowerWorldError, COMError, SimAutoFeatureError, create_object_string
except ImportError:
    raise

CRASH_PRONE_COMPONENTS = []

@pytest.fixture(scope="module")
def wb(saw_session):
    """
    Wraps the session-scoped SAW instance in a PowerWorld object.
    """
    workbench = PowerWorld()
    workbench.esa = saw_session
    return workbench


class TestPowerWorldFunctions:

    def test_simulation_control(self, wb, temp_file):
        """Tests flatstart, pflow, save, log, command, mode."""
        wb.flatstart()

        res = wb.pflow(getvolts=True)
        assert res is not None
        wb.pflow(getvolts=False)

        tmp_pwb = temp_file(".pwb")
        wb.save(tmp_pwb)
        assert os.path.exists(tmp_pwb)

        wb.log("Adapter Test Message")

        wb.edit_mode()
        wb.run_mode()

    def test_voltage_retrieval(self, wb):
        """Tests voltage() in all modes."""
        v = wb.voltage()
        assert len(v) > 0
        assert np.iscomplexobj(v.values)

        v_mag, v_ang = wb.voltage(complex=False)
        assert len(v_mag) > 0
        assert len(v_mag) == len(v_ang)

        v_kv = wb.voltage(pu=False)
        assert len(v_kv) > 0

        v_kv_mag, v_kv_ang = wb.voltage(pu=False, complex=False)
        assert len(v_kv_mag) > 0

    def test_component_retrieval(self, wb):
        """Tests gens, loads, shunts, lines, transformers, areas, zones."""
        assert not wb.gens().empty
        assert not wb.loads().empty
        assert isinstance(wb.shunts(), pd.DataFrame)
        assert isinstance(wb.transformers(), pd.DataFrame)
        assert not wb.lines().empty
        assert not wb.areas().empty
        assert not wb.zones().empty

    def test_set_voltages(self, wb):
        """Tests set_voltages round-trip."""
        v = wb.voltage(complex=True, pu=True)
        wb.set_voltages(v)

    def test_analysis(self, wb):
        """Tests violations, mismatches, net injection."""
        viols = wb.violations()
        assert isinstance(viols, pd.DataFrame)

        P, Q = wb.mismatch()
        assert not P.empty
        assert not Q.empty

        S = wb.mismatch(asComplex=True)
        assert np.iscomplexobj(S)

        Pn, Qn = wb.netinj()
        assert len(Pn) > 0
        Sn = wb.netinj(asComplex=True)
        assert np.iscomplexobj(Sn)

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

    def test_location(self, wb):
        """Tests busmap, buscoords."""
        m = wb.busmap()
        assert not m.empty
        wb.buscoords()
        df = wb.buscoords(astuple=False)
        assert isinstance(df, pd.DataFrame)


class TestWorkbenchStatics:
    """Workbench-level static analysis: power flow, voltage, Y-bus, Jacobian."""

    def test_pflow(self, wb):
        """Power flow solve with and without voltage retrieval."""
        v = wb.pflow(getvolts=True)
        assert v is not None
        assert len(v) > 0
        assert np.iscomplexobj(v.values)

        result = wb.pflow(getvolts=False)
        assert result is None

    def test_voltage(self, wb):
        """Voltage retrieval in all modes: complex/tuple, pu/kV."""
        v = wb.voltage(complex=True, pu=True)
        assert np.iscomplexobj(v.values)
        assert len(v) > 0

        mag, ang = wb.voltage(complex=False, pu=True)
        assert len(mag) > 0
        assert len(ang) > 0

        v_kv = wb.voltage(complex=True, pu=False)
        assert len(v_kv) > 0

        wb.set_voltages(v)

    def test_violations(self, wb):
        """Bus voltage violations with normal and tight limits."""
        viols = wb.violations(v_min=0.9, v_max=1.1)
        assert isinstance(viols, pd.DataFrame)
        assert 'Low' in viols.columns
        assert 'High' in viols.columns

        viols_tight = wb.violations(v_min=0.999, v_max=1.001)
        assert isinstance(viols_tight, pd.DataFrame)

    def test_mismatch_and_netinj(self, wb):
        """Bus power mismatches and net injection."""
        P, Q = wb.mismatch()
        assert not P.empty
        assert not Q.empty

        S = wb.mismatch(asComplex=True)
        assert np.iscomplexobj(S)

        Pn, Qn = wb.netinj()
        assert len(Pn) > 0
        Sn = wb.netinj(asComplex=True)
        assert np.iscomplexobj(Sn)

    def test_ybus(self, wb):
        """Y-Bus matrix retrieval, sparse and dense."""
        Y = wb.ybus()
        assert Y.shape[0] > 0
        assert Y.shape[0] == Y.shape[1]

        Y_dense = wb.ybus(dense=True)
        assert isinstance(Y_dense, np.ndarray)
        assert Y_dense.shape[0] > 0

    def test_jacobian(self, wb):
        """Jacobian: sparse, dense, polar form, with IDs."""
        wb.pflow(getvolts=False)

        J = wb.jacobian()
        assert J.shape[0] > 0

        J_dense = wb.jacobian(dense=True)
        assert isinstance(J_dense, np.ndarray)

        J_polar = wb.jacobian(dense=True, form='P')
        assert isinstance(J_polar, np.ndarray)
        assert J_polar.shape[0] > 0
        assert J_polar.shape[0] == J_polar.shape[1]

        J_ids, ids = wb.jacobian(dense=True, form='P', ids=True)
        assert isinstance(J_ids, np.ndarray)
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_solver_options(self, wb):
        """Solver option descriptors."""
        wb.do_one_iteration = True
        wb.do_one_iteration = False

        wb.max_iterations = 250

        wb.disable_angle_rotation = True
        wb.disable_angle_rotation = False

        wb.disable_opt_mult = True
        wb.disable_opt_mult = False

        wb.inner_ss_check = True
        wb.inner_ss_check = False

        wb.disable_gen_mvr_check = True
        wb.disable_gen_mvr_check = False

        wb.inner_check_gen_vars = True
        wb.inner_check_gen_vars = False

        wb.inner_backoff_gen_vars = True
        wb.inner_backoff_gen_vars = False


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
    Verifies that PowerWorld can read key fields for every defined component.
    """
    if component_class.TYPE in CRASH_PRONE_COMPONENTS:
        pytest.skip(f"Skipping {component_class.TYPE}: Known to cause SimAuto crashes.")

    try:
        df = wb[component_class]
    except SimAutoFeatureError as e:
        pytest.skip(f"Object type {component_class.TYPE} cannot be retrieved via SimAuto: {e.message}")
    except (PowerWorldError, COMError) as e:
        err_msg = str(e)
        if "Access violation" in err_msg or "memory resources" in err_msg:
            pytest.skip(f"Object type {component_class.TYPE} causes PW crash: {e}")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            fields = component_class.keys if component_class.keys else ["ALL"]
            wb.esa.SaveObjectFields(tmp_path, component_class.TYPE, fields)
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
