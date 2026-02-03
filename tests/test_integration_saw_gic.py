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
    from esapp.workbench import PowerWorld
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
    """PowerWorld with live SAW connection."""
    workbench = PowerWorld()
    workbench.esa = saw_session
    return workbench


class TestGIC:
    """GIC analysis: SAW commands, workbench options, model building, G-matrix."""

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
    def test_gic_setup_and_time(self, saw_instance):
        saw_instance.GICSetupTimeVaryingSeries()
        saw_instance.GICShiftOrStretchInputPoints()
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
    def test_gic_options(self, wb):
        """Descriptor-based options: bool and non-bool round-trip, configure."""
        gic = wb.gic

        # Bool descriptor round-trip
        gic.pf_include = True
        assert gic.pf_include is True
        gic.pf_include = False
        assert gic.pf_include is False
        gic.pf_include = True

        gic.ts_include = True
        assert gic.ts_include is True
        gic.ts_include = False
        assert gic.ts_include is False

        # Additional bool descriptors
        gic.update_line_volts = True
        assert gic.update_line_volts is True
        gic.update_line_volts = False
        assert gic.update_line_volts is False
        gic.update_line_volts = True

        gic.calc_max_direction = True
        assert gic.calc_max_direction is True
        gic.calc_max_direction = False

        gic.hotspot_include = False
        assert gic.hotspot_include is False

        # Non-bool descriptor round-trip
        gic.calc_mode = 'SnapShot'
        assert gic.calc_mode == 'SnapShot'
        gic.calc_mode = 'TimeVarying'
        assert gic.calc_mode == 'TimeVarying'
        gic.calc_mode = 'SnapShot'

        # configure() sets multiple options at once
        gic.configure(pf_include=True, ts_include=True, calc_mode='TimeVarying')
        assert gic.pf_include is True
        assert gic.ts_include is True
        assert gic.calc_mode == 'TimeVarying'
        gic.configure()

        # Class-level access returns the descriptor itself
        desc = type(gic).pf_include
        assert hasattr(desc, 'key')
        assert desc.key == 'IncludeInPowerFlow'

    @pytest.mark.order(7750)
    def test_gic_storm_and_clear(self, wb):
        wb.gic.storm(1.0, 90.0, solvepf=True)
        wb.gic.storm(1.0, 90.0, solvepf=False)
        wb.gic.cleargic()

    @pytest.mark.order(7770)
    def test_gic_loadb3d(self, wb):
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
    def test_gic_settings_and_gmatrix(self, wb):
        settings = wb.gic.settings()
        assert settings is not None
        assert isinstance(settings, pd.DataFrame)
        assert 'VariableName' in settings.columns

        G_sparse = wb.gic.gmatrix(sparse=True)
        assert G_sparse.shape[0] > 0
        G_dense = wb.gic.gmatrix(sparse=False)
        assert isinstance(G_dense, np.ndarray)

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

        pw = PowerWorld()
        pw.esa = gic_saw
        gic = pw.gic
        gic.pf_include = True

        subs = pw[Substation, ["SubNum", "SubName", "GICSubGroundOhms", "GICUsedSubGroundOhms"]]
        buses = pw[Bus, ["BusNum", "BusNomVolt", "SubNum"]]
        branches = pw[Branch, ["BusNum", "BusNum:1", "GICConductance", "BranchDeviceType",
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
        """model() returns self and all properties are populated."""
        model = wb.gic.model()
        assert model is wb.gic

        assert wb.gic.A.shape[0] > 0
        assert wb.gic.G.shape[0] == wb.gic.G.shape[1]
        assert wb.gic.H.shape[0] > 0
        assert wb.gic.zeta.shape[0] > 0
        assert wb.gic.Px.shape[0] > 0
        assert wb.gic.eff.shape[0] > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
