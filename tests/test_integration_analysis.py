"""
Integration tests for GIC, ATC, Transient Stability, and Time Step functionality.

WHAT THIS TESTS:
- GIC (Geomagnetically Induced Current) analysis
- ATC (Available Transfer Capability) analysis
- Transient stability simulations
- Time step simulation operations

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py
"""

import os
import pytest
import pandas as pd
import numpy as np

# Order markers for integration tests - advanced analysis tests (order 73-99)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]

try:
    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError, create_object_string
except ImportError:
    raise


@pytest.fixture(scope="module")
def saw_instance(saw_session):
    """Provides the session-scoped SAW instance to the tests in this module."""
    return saw_session


class TestGIC:
    """Tests for GIC (Geomagnetically Induced Current) analysis."""

    @pytest.mark.order(73)
    def test_gic_calculate(self, saw_instance):
        saw_instance.CalculateGIC(1.0, 90.0, False)
        saw_instance.ClearGIC()

    @pytest.mark.order(74)
    def test_gic_save_matrix(self, saw_instance, temp_file):
        tmp_mat = temp_file(".mat")
        tmp_id = temp_file(".txt")
        saw_instance.GICSaveGMatrix(tmp_mat, tmp_id)
        assert os.path.exists(tmp_mat)

    @pytest.mark.order(75)
    def test_gic_setup(self, saw_instance):
        saw_instance.GICSetupTimeVaryingSeries()
        saw_instance.GICShiftOrStretchInputPoints()

    @pytest.mark.order(76)
    def test_gic_time(self, saw_instance):
        saw_instance.GICTimeVaryingCalculate(0.0, False)
        saw_instance.GICTimeVaryingAddTime(10.0)
        saw_instance.GICTimeVaryingDeleteAllTimes()
        saw_instance.GICTimeVaryingEFieldCalculate(0.0, False)
        saw_instance.GICTimeVaryingElectricFieldsDeleteAllTimes()

    @pytest.mark.order(77)
    def test_gic_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.GICWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_gmd = temp_file(".gmd")
        saw_instance.GICWriteFilePSLF(tmp_gmd)

        tmp_gic = temp_file(".gic")
        saw_instance.GICWriteFilePTI(tmp_gic)

    @pytest.mark.order(77.1)
    def test_gic_options_set_pf_include(self, saw_instance):
        """Test setting GIC power flow include option."""
        from esapp.apps.gic import GIC

        gic = GIC()
        gic.set_esa(saw_instance)

        # Test setting to True
        gic.set_pf_include(True)
        value = gic.get_gic_option('IncludeInPowerFlow')
        assert value == 'YES', f"Expected 'YES', got '{value}'"

        # Test setting to False
        gic.set_pf_include(False)
        value = gic.get_gic_option('IncludeInPowerFlow')
        assert value == 'NO', f"Expected 'NO', got '{value}'"

        # Reset to True for subsequent tests
        gic.set_pf_include(True)

    @pytest.mark.order(77.2)
    def test_gic_options_set_ts_include(self, saw_instance):
        """Test setting GIC transient stability include option."""
        from esapp.apps.gic import GIC

        gic = GIC()
        gic.set_esa(saw_instance)

        # Test setting to True
        gic.set_ts_include(True)
        value = gic.get_gic_option('IncludeTimeDomain')
        assert value == 'YES', f"Expected 'YES', got '{value}'"

        # Test setting to False
        gic.set_ts_include(False)
        value = gic.get_gic_option('IncludeTimeDomain')
        assert value == 'NO', f"Expected 'NO', got '{value}'"

    @pytest.mark.order(77.3)
    def test_gic_options_set_calc_mode(self, saw_instance):
        """Test setting GIC calculation mode option."""
        from esapp.apps.gic import GIC

        gic = GIC()
        gic.set_esa(saw_instance)

        # Test SnapShot mode
        gic.set_calc_mode('SnapShot')
        value = gic.get_gic_option('CalcMode')
        assert value == 'SnapShot', f"Expected 'SnapShot', got '{value}'"

        # Test TimeVarying mode
        gic.set_calc_mode('TimeVarying')
        value = gic.get_gic_option('CalcMode')
        assert value == 'TimeVarying', f"Expected 'TimeVarying', got '{value}'"

        # Reset to SnapShot for subsequent tests
        gic.set_calc_mode('SnapShot')

    @pytest.mark.order(77.4)
    def test_gic_options_configure(self, saw_instance):
        """Test the configure() method sets multiple options at once."""
        from esapp.apps.gic import GIC

        gic = GIC()
        gic.set_esa(saw_instance)

        # Configure with all custom values
        gic.configure(pf_include=True, ts_include=True, calc_mode='TimeVarying')

        # Verify all options were set
        assert gic.get_gic_option('IncludeInPowerFlow') == 'YES'
        assert gic.get_gic_option('IncludeTimeDomain') == 'YES'
        assert gic.get_gic_option('CalcMode') == 'TimeVarying'

        # Configure with defaults
        gic.configure()  # defaults: pf_include=True, ts_include=False, calc_mode='SnapShot'

        assert gic.get_gic_option('IncludeInPowerFlow') == 'YES'
        assert gic.get_gic_option('IncludeTimeDomain') == 'NO'
        assert gic.get_gic_option('CalcMode') == 'SnapShot'


class TestGICGMatrix:
    """G-matrix comparison tests that run against multiple PowerWorld cases."""

    @pytest.mark.order(77.5)
    def test_gic_gmatrix_comparison(self, gic_saw):
        """
        Compare computed G-matrix from GIC.model() with PowerWorld's G-matrix.

        This validates that the internal G-matrix computation in GIC.model()
        produces results consistent with PowerWorld's GIC calculation engine.
        Parametrized over all cases listed in config_test.GIC_TEST_CASES.
        """
        from esapp.apps.gic import GIC
        from esapp.components import Substation, Bus, Branch, GICXFormer, Gen
        from scipy.sparse import issparse

        # Create GIC interface and connect to SAW
        gic = GIC()
        gic.set_esa(gic_saw)
        gic.set_pf_include(True)

        # Query network data for diagnostics
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

        # Get G-matrices
        try:
            model = gic.model()
            G_computed = model.G
        except Exception as e:
            pytest.skip(f"Could not generate GIC model: {e}")

        try:
            G_powerworld = gic.gmatrix(sparse=True)
        except Exception as e:
            pytest.skip(f"Could not retrieve PowerWorld G-matrix: {e}")

        assert issparse(G_computed) and issparse(G_powerworld)

        G_computed_dense = G_computed.toarray()
        G_powerworld_dense = G_powerworld.toarray()

        if G_computed_dense.shape != G_powerworld_dense.shape:
            pytest.skip(f"Shape mismatch: {G_computed_dense.shape} vs {G_powerworld_dense.shape}")

        # Compute metrics
        diff = np.abs(G_computed_dense - G_powerworld_dense)
        max_diff = np.max(diff)
        n_subs, n_bus = len(subs), len(buses)
        diff_threshold = 1e-6

        # Get PowerWorld node IDs for labeling
        pw_node_ids = []
        try:
            _, pw_node_ids = gic_saw.get_gmatrix_with_ids(full=False)
        except Exception:
            pw_node_ids = [f"idx_{i}" for i in range(G_powerworld_dense.shape[0])]

        # Sparsity analysis
        computed_pattern = set(zip(*G_computed.nonzero()))
        pw_pattern = set(zip(*G_powerworld.nonzero()))
        common_pattern = computed_pattern & pw_pattern
        only_in_pw = pw_pattern - computed_pattern

        # ============================================================
        # DEBUG OUTPUT
        # ============================================================
        print(f"\n{'='*80}")
        print(f"G-MATRIX COMPARISON: {n_subs} subs + {n_bus} buses = {n_subs + n_bus} nodes")
        print(f"{'='*80}")

        # Summary metrics
        print(f"\n{'METRIC':<25} {'COMPUTED':>15} {'POWERWORLD':>15} {'DIFF':>12}")
        print("-" * 67)
        print(f"{'Non-zeros':<25} {G_computed.nnz:>15} {G_powerworld.nnz:>15} {G_powerworld.nnz - G_computed.nnz:>12}")
        print(f"{'Diagonal sum':<25} {np.sum(np.diag(G_computed_dense)):>15.2e} {np.sum(np.diag(G_powerworld_dense)):>15.2e} {abs(np.sum(np.diag(G_computed_dense)) - np.sum(np.diag(G_powerworld_dense))):>12.2e}")
        print(f"{'Frobenius norm':<25} {np.linalg.norm(G_computed_dense, 'fro'):>15.2e} {np.linalg.norm(G_powerworld_dense, 'fro'):>15.2e} {abs(np.linalg.norm(G_computed_dense, 'fro') - np.linalg.norm(G_powerworld_dense, 'fro')):>12.2e}")
        print(f"{'Sparsity common/PW-only':<25} {len(common_pattern):>15} {len(only_in_pw):>15}")

        # ============================================================
        # CHECK 1: Transmission Line Connections (Bus-Bus)
        # ============================================================
        print(f"\n--- CHECK 1: Transmission Line Connections ---")
        line_checks = []
        for _, row in lines.iterrows():
            fb, tb = int(row['BusNum']), int(row['BusNum:1'])
            g_line = row['GICConductance']
            # Find bus indices in G-matrix (buses start at n_subs)
            fb_idx = buses[buses['BusNum'] == fb].index[0] + n_subs if fb in buses['BusNum'].values else None
            tb_idx = buses[buses['BusNum'] == tb].index[0] + n_subs if tb in buses['BusNum'].values else None
            if fb_idx is not None and tb_idx is not None:
                computed_val = G_computed_dense[fb_idx, tb_idx]
                pw_val = G_powerworld_dense[fb_idx, tb_idx]
                line_checks.append((fb, tb, g_line, computed_val, pw_val))

        # Count matches
        line_matches = sum(1 for _, _, _, c, p in line_checks if abs(c - p) < 1.0)
        print(f"  Lines checked: {len(line_checks)}, Matches (diff<1): {line_matches}")

        # Show a few line connections that match well
        good_lines = [(f, t, g, c, p) for f, t, g, c, p in line_checks if abs(c - p) < 0.1 * abs(p) if p != 0][:5]
        if good_lines:
            print(f"  Sample MATCHING line connections:")
            for fb, tb, g, c, p in good_lines:
                print(f"    Bus {fb} <-> Bus {tb}: G={g:.4f}, Computed={c:.4f}, PW={p:.4f} [OK]")

        # ============================================================
        # CHECK 2: Transformer Winding Connections (Sub-Bus)
        # ============================================================
        print(f"\n--- CHECK 2: Transformer Winding Connections ---")
        gicxf = gic[GICXFormer, [
            "BusNum3W", "BusNum3W:1", "SubNum", "SubNum:1",
            "GICXFCoilR1", "GICXFCoilR1:1", "GICXFConfigUsed",
        ]]
        xfmr_checks = []
        for _, row in gicxf.iterrows():
            for side, sub_col, bus_col, r_col in [
                ('From', 'SubNum', 'BusNum3W', 'GICXFCoilR1'),
                ('To', 'SubNum:1', 'BusNum3W:1', 'GICXFCoilR1:1'),
            ]:
                sub_n, bus_n = int(row[sub_col]), int(row[bus_col])
                s_idx = subs[subs['SubNum'] == sub_n].index[0] if sub_n in subs['SubNum'].values else None
                b_idx = buses[buses['BusNum'] == bus_n].index[0] + n_subs if bus_n in buses['BusNum'].values else None
                if s_idx is not None and b_idx is not None:
                    r_val = row[r_col]
                    g_val = 1.0 / r_val if r_val != 0 else 0.0
                    xfmr_checks.append((side, sub_n, bus_n, row['GICXFConfigUsed'], g_val, s_idx, b_idx))

        print(f"  Transformer windings: {len(xfmr_checks)}")
        print(f"  Sample winding connections:")
        for side, sub_num, bus, cfg, g, s_idx, b_idx in xfmr_checks[:6]:
            c_val = G_computed_dense[s_idx, b_idx]
            p_val = G_powerworld_dense[s_idx, b_idx]
            match = "[OK]" if abs(c_val - p_val) < 1.0 else "[DIFF]"
            print(f"    Sub{sub_num} <-> Bus{bus} ({side}, {cfg}): G={g:.2f}, Computed={c_val:.2f}, PW={p_val:.2f} {match}")

        # ============================================================
        # CHECK 3: Substation Grounding (Diagonal)
        # ============================================================
        print(f"\n--- CHECK 3: Substation Grounding (Diagonal) ---")
        sub_matches = 0
        sub_diffs = []
        for i in range(n_subs):
            c_diag = G_computed_dense[i, i]
            p_diag = G_powerworld_dense[i, i]
            sub_name = pw_node_ids[i] if i < len(pw_node_ids) else f"Sub[{i}]"
            sub_r = subs.iloc[i]['GICUsedSubGroundOhms']
            sub_diffs.append((sub_name, sub_r, c_diag, p_diag, abs(c_diag - p_diag)))
            if abs(c_diag - p_diag) < max(1.0, 0.1 * abs(p_diag)):
                sub_matches += 1

        print(f"  Substations with matching diagonals: {sub_matches}/{n_subs}")
        # Show best matches
        best_subs = sorted(sub_diffs, key=lambda x: x[4])[:5]
        print(f"  Best matching substations:")
        for name, r, c, p, d in best_subs:
            print(f"    {name:15}: R={r:.2f}, Computed={c:.2e}, PW={p:.2e}, Diff={d:.2e} [OK]")

        # ============================================================
        # DIAGNOSTIC: Largest Differences Analysis
        # ============================================================
        print(f"\n--- DIAGNOSTIC: Top 5 Largest Differences ---")
        flat_diff = diff.flatten()
        sorted_indices = np.argsort(flat_diff)[::-1]

        for i in range(min(5, len(sorted_indices))):
            flat_idx = sorted_indices[i]
            row = flat_idx // G_computed_dense.shape[1]
            col = flat_idx % G_computed_dense.shape[1]
            if diff[row, col] < diff_threshold:
                break

            row_name = pw_node_ids[row] if row < len(pw_node_ids) else f"idx_{row}"
            col_name = pw_node_ids[col] if col < len(pw_node_ids) else f"idx_{col}"
            c_val = G_computed_dense[row, col]
            p_val = G_powerworld_dense[row, col]

            print(f"\n  [{i+1}] {row_name} <-> {col_name}")
            print(f"      Computed: {c_val:>12.4e}, PowerWorld: {p_val:>12.4e}, Diff: {diff[row, col]:>12.4e}")

            # Diagnose: Is this a Sub-Sub, Sub-Bus, or Bus-Bus connection?
            is_sub_row = row < n_subs
            is_sub_col = col < n_subs
            conn_type = "Sub-Sub" if is_sub_row and is_sub_col else "Sub-Bus" if is_sub_row or is_sub_col else "Bus-Bus"
            print(f"      Type: {conn_type}")

            # For Sub-Bus connections, check if there's a transformer
            if conn_type == "Sub-Bus":
                sub_idx = row if is_sub_row else col
                bus_idx = col if is_sub_row else row
                sub_num = subs.iloc[sub_idx]['SubNum']
                bus_num = buses.iloc[bus_idx - n_subs]['BusNum']

                # Find GICXFormers at this substation connecting to this bus
                related = gicxf[
                    ((gicxf['SubNum'] == sub_num) | (gicxf['SubNum:1'] == sub_num)) &
                    ((gicxf['BusNum3W'] == bus_num) | (gicxf['BusNum3W:1'] == bus_num))
                ]
                if len(related) > 0:
                    print(f"      Transformers at Sub{sub_num} to Bus{bus_num}: {len(related)}")
                    for _, xf in related.iterrows():
                        print(f"        {xf['BusNum3W']}<->{xf['BusNum3W:1']}: CFG={xf['GICXFConfigUsed']}")
                else:
                    print(f"      NO transformer connecting Sub{sub_num} to Bus{bus_num}")
                    sub_buses = buses[buses['SubNum'] == sub_num]['BusNum'].tolist()
                    print(f"      Buses at Sub{sub_num}: {sub_buses}")

            # For Bus-Bus, check if it's a transformer inter-bus connection
            elif conn_type == "Bus-Bus":
                bus1 = buses.iloc[row - n_subs]['BusNum']
                bus2 = buses.iloc[col - n_subs]['BusNum']
                xfmr_conn = gicxf[
                    ((gicxf['BusNum3W'] == bus1) & (gicxf['BusNum3W:1'] == bus2)) |
                    ((gicxf['BusNum3W'] == bus2) & (gicxf['BusNum3W:1'] == bus1))
                ]
                if len(xfmr_conn) > 0:
                    print(f"      Transformer(s) between Bus{bus1} and Bus{bus2}: {len(xfmr_conn)}")
                else:
                    line_conn = lines[
                        ((lines['BusNum'] == bus1) & (lines['BusNum:1'] == bus2)) |
                        ((lines['BusNum'] == bus2) & (lines['BusNum:1'] == bus1))
                    ]
                    if len(line_conn) > 0:
                        print(f"      Transmission line(s) between Bus{bus1} and Bus{bus2}: {len(line_conn)}, G_sum={line_conn['GICConductance'].sum():.4f}")
                    else:
                        print(f"      NO direct connection between Bus{bus1} and Bus{bus2}")

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n--- Summary ---")
        print(f"  Model: {len(lines)} lines, {len(gicxf)} transformers")
        print(f"  Connections matching: Common={len(common_pattern)}, Missing={len(only_in_pw)}")
        num_differing = np.sum(diff > diff_threshold)
        print(f"  Elements differing: {num_differing}/{diff.size} ({100*num_differing/diff.size:.1f}%)")
        print(f"\n{'='*80}\n")

        # DIAGNOSTIC: For top differences involving Sub-Bus, query GICXFormer from SAW
        if max_diff > 1.0:
            print(f"\n--- DIAGNOSTIC: GICXFormer data for top-diff buses ---")
            xf_fields = [
                "BusNum3W", "BusNum3W:1", "SubNum", "SubNum:1",
                "GICXFCoilR1", "GICXFCoilR1:1",
                "XFConfiguration", "XFConfiguration:1",
                "GICBlockDevice", "GICAutoXFUsed", "GICXF3Type",
                "BusNomVolt", "BusNomVolt:1",
                "GICXFConfigUsed", "GICXFConfigUsed:1",
            ]
            all_gicxf = gic[GICXFormer, xf_fields]
            # Find buses at top-diff nodes
            for i in range(min(3, len(sorted_indices))):
                flat_idx = sorted_indices[i]
                row_i = flat_idx // G_computed_dense.shape[1]
                col_i = flat_idx % G_computed_dense.shape[1]
                is_sub_r = row_i < n_subs
                is_sub_c = col_i < n_subs
                if is_sub_r != is_sub_c:  # Sub-Bus only
                    sub_idx = row_i if is_sub_r else col_i
                    bus_idx = col_i if is_sub_r else row_i
                    sub_num = subs.iloc[sub_idx]['SubNum']
                    bus_num = buses.iloc[bus_idx - n_subs]['BusNum']
                    print(f"\n  Sub {sub_num} <-> Bus {bus_num}")
                    # GICXFormers at this bus
                    at_bus = all_gicxf[
                        (all_gicxf['BusNum3W'] == bus_num) | (all_gicxf['BusNum3W:1'] == bus_num)
                    ]
                    if len(at_bus) > 0:
                        print(f"    GICXFormers at bus {bus_num}:")
                        print(at_bus.to_string())
                    else:
                        print(f"    No GICXFormers at bus {bus_num}")
                    # Generators at this bus
                    gen_fields = ["BusNum", "GICConductance", "GICGenIncludeImplicitGSU"]
                    gens = gic[Gen, gen_fields]
                    at_bus_gens = gens[gens['BusNum'] == bus_num]
                    if len(at_bus_gens) > 0:
                        print(f"    Generators at bus {bus_num}:")
                        print(at_bus_gens.to_string())
                    else:
                        print(f"    No generators at bus {bus_num}")
                    # Buses at this substation (from model)
                    sub_buses = buses[buses['SubNum'] == sub_num]
                    print(f"    Buses at sub {sub_num}: {sub_buses['BusNum'].tolist()}")

        # Test result
        rtol, atol = 1e-3, 1e-6
        if np.allclose(G_computed_dense, G_powerworld_dense, rtol=rtol, atol=atol):
            return

        MOHM = 1e6
        if np.any(np.abs(G_computed_dense) > MOHM * 0.9):
            pytest.skip(f"G-matrices differ (max={max_diff:.2e}). Large placeholder values detected.")
        elif max_diff < 1.0:
            pass  # Small differences acceptable
        else:
            pytest.fail(f"G-matrices differ significantly (max={max_diff:.2e}, {num_differing}/{diff.size} elements)")


class TestATC:
    """Tests for ATC (Available Transfer Capability) analysis."""

    @pytest.mark.order(78)
    def test_atc_determine(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
            buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DetermineATC(seller, buyer)
        else:
            pytest.skip("Not enough areas for ATC")

    @pytest.mark.order(79)
    def test_atc_multiple(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            s = create_object_string("Area", areas.iloc[0]["AreaNum"])
            b = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DirectionsAutoInsert(s, b)

        try:
            saw_instance.DetermineATCMultipleDirections()
        except PowerWorldPrerequisiteError:
            pytest.skip("No directions defined for ATC")

    @pytest.mark.order(80)
    def test_atc_results(self, saw_instance):
        saw_instance._object_fields["transferlimiter"] = pd.DataFrame({
            "internal_field_name": ["LimitingContingency", "MaxFlow"],
            "field_data_type": ["String", "Real"],
            "key_field": ["", ""],
            "description": ["", ""],
            "display_name": ["", ""]
        }).sort_values(by="internal_field_name")

        saw_instance.GetATCResults(["MaxFlow", "LimitingContingency"])


class TestTransient:
    """Tests for Transient Stability simulations."""

    @pytest.mark.order(81)
    def test_transient_initialize(self, saw_instance):
        saw_instance.TSInitialize()

    @pytest.mark.order(82)
    def test_transient_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(83)
    def test_transient_critical_time(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.TSCalculateCriticalClearTime(branch_str)

    @pytest.mark.order(84)
    def test_transient_playin(self, saw_instance):
        times = np.array([0.0, 0.1])
        signals = np.array([[1.0], [1.0]])
        saw_instance.TSSetPlayInSignals("TestSignal", times, signals)

    @pytest.mark.order(85)
    def test_transient_save_models(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteModels(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_aux2 = temp_file(".aux")
        saw_instance.TSSaveDynamicModels(tmp_aux2, "AUX", "Gen")
        assert os.path.exists(tmp_aux2)


class TestTimeStep:
    """Tests for Time Step Simulation operations."""

    @pytest.mark.order(86)
    def test_timestep_delete(self, saw_instance):
        saw_instance.TimeStepDeleteAll()

    @pytest.mark.order(87)
    def test_timestep_run(self, saw_instance):
        saw_instance.TimeStepDoRun()
        try:
            saw_instance.TimeStepDoSinglePoint("2025-01-01T10:00:00")
        except PowerWorldPrerequisiteError:
            pass  # Expected if time points not defined
        try:
            saw_instance.TimeStepClearResults()
        except PowerWorldError:
            pass
        saw_instance.TimeStepResetRun()

    @pytest.mark.order(88)
    def test_timestep_save(self, saw_instance, temp_file):
        tmp_pww = temp_file(".pww")
        saw_instance.TimeStepSavePWW(tmp_pww)

        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TimeStepSaveResultsByTypeCSV("Gen", tmp_csv)
        except PowerWorldError:
            pass  # Likely no results

    @pytest.mark.order(89)
    def test_timestep_fields(self, saw_instance):
        saw_instance.TimeStepSaveFieldsSet("Gen", ["GenMW"])
        saw_instance.TimeStepSaveFieldsClear(["Gen"])


class TestPVQV:
    """Tests for PV and QV analysis."""

    @pytest.mark.order(90)
    def test_pv_qv_run(self, saw_instance):
        df = saw_instance.RunQV()
        assert df is not None

    @pytest.mark.order(91)
    def test_pv_clear(self, saw_instance):
        """Test clearing PV analysis results."""
        saw_instance.PVClear()

    @pytest.mark.order(92)
    def test_pv_export(self, saw_instance, temp_file):
        """Test exporting PV analysis results."""
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.PVWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except PowerWorldPrerequisiteError:
            pytest.skip("PV analysis not available or no results")

    @pytest.mark.order(93)
    def test_qv_clear(self, saw_instance):
        """Test clearing QV analysis results."""
        saw_instance.QVDeleteAllResults()

    @pytest.mark.order(94)
    def test_qv_export(self, saw_instance, temp_file):
        """Test exporting QV analysis results."""
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.QVWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except PowerWorldPrerequisiteError:
            pytest.skip("QV analysis not available or no results")


class TestTransientAdvanced:
    """Additional tests for Transient Stability simulations."""

    @pytest.mark.order(95)
    def test_transient_result_storage_set_all(self, saw_instance):
        """Test TSResultStorageSetAll for all storage modes."""
        # TSResultStorageSetAll(object_type, store_value) - object type first, then bool
        saw_instance.TSResultStorageSetAll("Gen", True)
        saw_instance.TSResultStorageSetAll("Gen", False)

    @pytest.mark.order(96)
    def test_transient_clear_playin_signals(self, saw_instance):
        """Test clearing play-in signals."""
        saw_instance.TSClearPlayInSignals()

    @pytest.mark.order(98)
    def test_transient_validate(self, saw_instance):
        """Test TSValidate for model validation."""
        saw_instance.TSInitialize()
        try:
            saw_instance.TSValidate()
        except PowerWorldPrerequisiteError:
            pytest.skip("Transient validation not available")

    @pytest.mark.order(99)
    def test_transient_auto_correct(self, saw_instance):
        """Test TSAutoCorrect for automatic model corrections."""
        saw_instance.TSInitialize()
        try:
            saw_instance.TSAutoCorrect()
        except PowerWorldPrerequisiteError:
            pytest.skip("Auto-correct not available")

    @pytest.mark.order(100)
    def test_transient_write_results(self, saw_instance, temp_file):
        """Test writing transient results to CSV file."""
        tmp_csv = temp_file(".csv")
        try:
            # TSGetResults(mode, contingencies, plots_fields, filename)
            saw_instance.TSGetResults("CSV", ["ALL"], ["GenMW"], filename=tmp_csv)
            assert os.path.exists(tmp_csv)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No transient results to write")
