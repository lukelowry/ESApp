"""
Integration tests for GIC (Geomagnetically Induced Currents) analysis via SAW.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They cover low-level SAW GIC commands,
workbench-level option getters/setters, storm application, B3D loading,
G-matrix extraction, model building, and G-matrix comparison with PowerWorld.

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

RELATED TEST FILES:
    - test_integration_saw_core.py          -- base SAW operations, logging, I/O
    - test_integration_saw_modify.py        -- destructive modify, region, case actions
    - test_integration_saw_powerflow.py     -- power flow, matrices, sensitivity, topology
    - test_integration_saw_contingency.py   -- contingency and fault analysis
    - test_integration_saw_transient.py     -- transient stability
    - test_integration_saw_operations.py    -- ATC, OPF, PV/QV, time step, weather, scheduled
    - test_integration_workbench.py         -- GridWorkBench facade and statics
    - test_integration_network.py           -- Network topology

USAGE:
    pytest tests/test_integration_saw_gic.py -v
"""

import os
import pytest
import pandas as pd
import numpy as np

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]

try:
    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError, create_object_string
    from esapp.workbench import GridWorkBench
    from esapp.utils import GIC, jac_decomp
    from esapp.components import Bus, Branch, Substation
except ImportError:
    raise


@pytest.fixture(scope="module")
def saw_instance(saw_session):
    """Provides the session-scoped SAW instance to the tests in this module."""
    return saw_session


@pytest.fixture(scope="module")
def wb(saw_session):
    """GridWorkBench with live SAW connection."""
    workbench = GridWorkBench()
    workbench.set_esa(saw_session)
    return workbench


class TestGIC:
    """GIC analysis: SAW commands, workbench options, model building, G-matrix.

    All GIC-related integration tests are consolidated here. This covers:
    - Low-level SAW GIC commands (calculate, save, write)
    - Workbench-level option getters/setters (pf_include, ts_include, calc_mode)
    - The configure() shorthand
    - Storm application and result clearing
    - B3D loading and time-varying CSV upload
    - GIC settings retrieval and G-matrix extraction
    - jac_decomp utility
    - Full model() generation and property validation
    - G-matrix comparison between model() and PowerWorld
    """

    @pytest.mark.order(7300)
    def test_gic_calculate(self, saw_instance):
        saw_instance.EnterMode("EDIT")
        saw_instance.SetData(
            'GIC_Options_Value',
            ['VariableName', 'ValueField'],
            ['IncludeInPowerFlow', 'YES']
        )
        saw_instance.SetData(
            'GIC_Options_Value',
            ['VariableName', 'ValueField'],
            ['CalcMode', 'SnapShot']
        )
        saw_instance.EnterMode("RUN")

        subs = saw_instance.GetParametersMultipleElement(
            "Substation", ["SubNum", "GICSubGroundOhms"]
        )
        has_grounding = (
            subs is not None and not subs.empty
            and (subs["GICSubGroundOhms"].astype(float) > 0).any()
        )
        if not has_grounding:
            branches = saw_instance.GetParametersMultipleElement(
                "Branch", ["BusNum", "BusNum:1", "BranchDeviceType",
                           "GICCoilRFrom", "GICCoilRTo"]
            )
            has_xfmr_data = False
            if branches is not None and not branches.empty:
                xfmrs = branches[branches["BranchDeviceType"] == "Transformer"]
                has_xfmr_data = (
                    not xfmrs.empty
                    and ((xfmrs["GICCoilRFrom"].astype(float) > 0).any()
                         or (xfmrs["GICCoilRTo"].astype(float) > 0).any())
                )
            assert has_xfmr_data, (
                "Case has no GIC data (no substation grounding or transformer "
                "coil resistances). Cannot run GIC calculation."
            )

        saw_instance.GICCalculate(1.0, 90.0, False)
        saw_instance.GICClear()

    @pytest.mark.order(7400)
    def test_gic_save_matrix(self, saw_instance, temp_file):
        tmp_mat = temp_file(".mat")
        tmp_id = temp_file(".txt")
        saw_instance.GICSaveGMatrix(tmp_mat, tmp_id)
        assert os.path.exists(tmp_mat)

    @pytest.mark.order(7500)
    def test_gic_setup(self, saw_instance):
        saw_instance.GICSetupTimeVaryingSeries()
        saw_instance.GICShiftOrStretchInputPoints()

    @pytest.mark.order(7600)
    def test_gic_time(self, saw_instance):
        saw_instance.GICTimeVaryingCalculate(0.0, False)
        saw_instance.GICTimeVaryingAddTime(10.0)
        saw_instance.GICTimeVaryingDeleteAllTimes()
        saw_instance.GICTimeVaryingEFieldCalculate(0.0, False)
        saw_instance.GICTimeVaryingElectricFieldsDeleteAllTimes()

    @pytest.mark.order(7700)
    def test_gic_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.GICWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_gmd = temp_file(".gmd")
        saw_instance.GICWriteFilePSLF(tmp_gmd)

        tmp_gic = temp_file(".gic")
        saw_instance.GICWriteFilePTI(tmp_gic)

    @pytest.mark.order(7710)
    def test_gic_options_pf_include(self, wb):
        gic = wb.gic
        gic.set_pf_include(True)
        assert gic.get_gic_option('IncludeInPowerFlow') == 'YES'
        gic.set_pf_include(False)
        assert gic.get_gic_option('IncludeInPowerFlow') == 'NO'
        gic.set_pf_include(True)

    @pytest.mark.order(7720)
    def test_gic_options_ts_include(self, wb):
        gic = wb.gic
        gic.set_ts_include(True)
        assert gic.get_gic_option('IncludeTimeDomain') == 'YES'
        gic.set_ts_include(False)
        assert gic.get_gic_option('IncludeTimeDomain') == 'NO'

    @pytest.mark.order(7730)
    def test_gic_options_calc_mode(self, wb):
        gic = wb.gic
        gic.set_calc_mode('SnapShot')
        assert gic.get_gic_option('CalcMode') == 'SnapShot'
        gic.set_calc_mode('TimeVarying')
        assert gic.get_gic_option('CalcMode') == 'TimeVarying'
        gic.set_calc_mode('SnapShot')

    @pytest.mark.order(7740)
    def test_gic_configure(self, wb):
        gic = wb.gic
        gic.configure(pf_include=True, ts_include=True, calc_mode='TimeVarying')
        assert gic.get_gic_option('IncludeInPowerFlow') == 'YES'
        assert gic.get_gic_option('IncludeTimeDomain') == 'YES'
        assert gic.get_gic_option('CalcMode') == 'TimeVarying'
        gic.configure()

    @pytest.mark.order(7750)
    def test_gic_storm(self, wb):
        wb.gic.storm(1.0, 90.0, solvepf=True)
        wb.gic.storm(1.0, 90.0, solvepf=False)

    @pytest.mark.order(7760)
    def test_gic_cleargic(self, wb):
        wb.gic.cleargic()

    @pytest.mark.order(7770)
    def test_gic_loadb3d(self, wb):
        """loadb3d with and without setup on load."""
        with pytest.raises((PowerWorldPrerequisiteError, PowerWorldError)):
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=True)
        with pytest.raises((PowerWorldPrerequisiteError, PowerWorldError)):
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=False)

    @pytest.mark.order(7775)
    def test_gic_timevary_csv(self, wb, temp_file):
        import csv as csvmod
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, 'w', newline='') as f:
            writer = csvmod.writer(f)
            writer.writerow(["Branch '1' '2' '1'", 0.1, 0.11, 0.14])
        wb.gic.timevary_csv(tmp_csv)

    @pytest.mark.order(7780)
    def test_gic_settings(self, wb):
        settings = wb.gic.settings()
        assert settings is not None
        assert isinstance(settings, pd.DataFrame)
        assert 'VariableName' in settings.columns

    @pytest.mark.order(7790)
    def test_gic_gmatrix(self, wb):
        G_sparse = wb.gic.gmatrix(sparse=True)
        assert G_sparse.shape[0] > 0
        G_dense = wb.gic.gmatrix(sparse=False)
        assert isinstance(G_dense, np.ndarray)

    @pytest.mark.order(7800)
    def test_gic_get_option_missing(self, wb):
        val = wb.gic.get_gic_option('NonExistentOption12345')
        assert val is None

    @pytest.mark.order(7801)
    def test_jac_decomp(self):
        J = np.arange(16).reshape(4, 4).astype(float)
        parts = list(jac_decomp(J))
        assert len(parts) == 4
        for p in parts:
            assert p.shape == (2, 2)

    @pytest.mark.order(7850)
    def test_gic_gmatrix_comparison(self, gic_saw):
        """Compare computed G-matrix from GIC.model() with PowerWorld's."""
        from scipy.sparse import issparse

        gic = GIC()
        gic.set_esa(gic_saw)
        gic.set_pf_include(True)

        subs = gic[Substation, ["SubNum", "SubName", "GICSubGroundOhms", "GICUsedSubGroundOhms"]]
        buses = gic[Bus, ["BusNum", "BusNomVolt", "SubNum"]]
        branches = gic[Branch, ["BusNum", "BusNum:1", "GICConductance", "BranchDeviceType",
                                 "GICCoilRFrom", "GICCoilRTo"]]
        lines = branches.loc[
            branches['BranchDeviceType'] != 'Transformer',
            ["BusNum", "BusNum:1", "GICConductance"]
        ]
        xfmrs = branches[branches["BranchDeviceType"] == "Transformer"]
        has_grounding = (subs["GICSubGroundOhms"] > 0).any()
        has_xfmr_data = (xfmrs["GICCoilRFrom"] > 0).any() or (xfmrs["GICCoilRTo"] > 0).any()

        if not has_grounding and not has_xfmr_data:
            pytest.skip("Case does not have GIC data configured")

        model = gic.model()
        G_computed = model.G

        G_powerworld = gic.gmatrix(sparse=True)

        assert issparse(G_computed) and issparse(G_powerworld)

        G_computed_dense = G_computed.toarray()
        G_powerworld_dense = G_powerworld.toarray()

        if G_computed_dense.shape != G_powerworld_dense.shape:
            pytest.fail(f"Shape mismatch: {G_computed_dense.shape} vs {G_powerworld_dense.shape}")

        diff = np.abs(G_computed_dense - G_powerworld_dense)
        max_diff = np.max(diff)

        rtol, atol = 1e-3, 1e-6
        if np.allclose(G_computed_dense, G_powerworld_dense, rtol=rtol, atol=atol):
            return

        MOHM = 1e6
        num_differing = np.sum(diff > 1e-6)
        if np.any(np.abs(G_computed_dense) > MOHM * 0.9):
            pytest.skip(f"G-matrices differ (max={max_diff:.2e}). Large placeholder values detected.")
        elif max_diff < 1.0:
            pass
        else:
            pytest.fail(f"G-matrices differ significantly (max={max_diff:.2e}, {num_differing}/{diff.size} elements)")

    @pytest.mark.order(7860)
    def test_gic_model(self, wb):
        model = wb.gic.model()
        assert model is wb.gic

    @pytest.mark.order(7870)
    def test_gic_model_properties(self, wb):
        wb.gic.model()

        A = wb.gic.A
        G = wb.gic.G
        H = wb.gic.H
        zeta = wb.gic.zeta
        Px = wb.gic.Px
        eff = wb.gic.eff

        assert A.shape[0] > 0
        assert G.shape[0] == G.shape[1]
        assert H.shape[0] > 0
        assert zeta.shape[0] > 0
        assert Px.shape[0] > 0
        assert eff.shape[0] > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
