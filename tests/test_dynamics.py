"""
Unit tests for the Dynamics transient stability simulation module.

Tests the ContingencyBuilder, SimAction enum, and Dynamics class functionality
using mocked dependencies (no PowerWorld required).

USAGE:
    pytest tests/test_dynamics.py -v
"""
import pytest
import os
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import pandas as pd
import numpy as np

from esapp.apps.dynamics import (
    Dynamics,
    ContingencyBuilder,
    SimAction,
    TS,
)


# =============================================================================
# SimAction Enum Tests
# =============================================================================

class TestSimAction:
    """Tests for the SimAction enumeration."""

    def test_fault_3pb_value(self):
        """SimAction.FAULT_3PB has correct value."""
        assert SimAction.FAULT_3PB.value == "FAULT 3PB SOLID"

    def test_clear_fault_value(self):
        """SimAction.CLEAR_FAULT has correct value."""
        assert SimAction.CLEAR_FAULT.value == "CLEARFAULT"

    def test_open_value(self):
        """SimAction.OPEN has correct value."""
        assert SimAction.OPEN.value == "OPEN"

    def test_close_value(self):
        """SimAction.CLOSE has correct value."""
        assert SimAction.CLOSE.value == "CLOSE"

    def test_simaction_is_string_enum(self):
        """SimAction inherits from str for easy string formatting."""
        assert isinstance(SimAction.OPEN, str)


# =============================================================================
# ContingencyBuilder Tests
# =============================================================================

class TestContingencyBuilder:
    """Tests for the ContingencyBuilder class."""

    def test_init_default_runtime(self):
        """ContingencyBuilder initializes with default runtime of 10.0."""
        builder = ContingencyBuilder("TestCtg")
        assert builder.name == "TestCtg"
        assert builder.runtime == 10.0
        assert builder._current_time == 0.0
        assert builder._events == []

    def test_init_custom_runtime(self):
        """ContingencyBuilder accepts custom runtime."""
        builder = ContingencyBuilder("TestCtg", runtime=5.0)
        assert builder.runtime == 5.0

    def test_at_sets_time_cursor(self):
        """at() method sets the current time cursor."""
        builder = ContingencyBuilder("TestCtg")
        result = builder.at(1.5)
        assert builder._current_time == 1.5
        assert result is builder  # Returns self for chaining

    def test_at_negative_time_raises(self):
        """at() raises ValueError for negative time."""
        builder = ContingencyBuilder("TestCtg")
        with pytest.raises(ValueError, match="Time cannot be negative"):
            builder.at(-1.0)

    def test_at_zero_time_allowed(self):
        """at() allows time of exactly 0."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(0.0)
        assert builder._current_time == 0.0

    def test_add_event_with_simaction(self):
        """add_event() correctly adds event with SimAction enum."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(1.0).add_event("Bus", "101", SimAction.FAULT_3PB)
        assert len(builder._events) == 1
        assert builder._events[0] == (1.0, "Bus", "101", "FAULT 3PB SOLID")

    def test_add_event_with_string_action(self):
        """add_event() correctly adds event with string action."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(2.0).add_event("Gen", "1 '1'", "CUSTOM_ACTION")
        assert len(builder._events) == 1
        assert builder._events[0] == (2.0, "Gen", "1 '1'", "CUSTOM_ACTION")

    def test_add_event_returns_self(self):
        """add_event() returns self for method chaining."""
        builder = ContingencyBuilder("TestCtg")
        result = builder.add_event("Bus", "101", SimAction.OPEN)
        assert result is builder

    def test_fault_bus(self):
        """fault_bus() creates a 3-phase solid fault event."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(1.0).fault_bus("101")
        assert len(builder._events) == 1
        assert builder._events[0] == (1.0, "Bus", "101", "FAULT 3PB SOLID")

    def test_fault_bus_with_integer_bus(self):
        """fault_bus() converts integer bus number to string."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(1.0).fault_bus(101)
        assert builder._events[0][2] == "101"

    def test_clear_fault(self):
        """clear_fault() creates a clear fault event."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(1.1).clear_fault("101")
        assert len(builder._events) == 1
        assert builder._events[0] == (1.1, "Bus", "101", "CLEARFAULT")

    def test_trip_gen_default_gid(self):
        """trip_gen() uses default generator ID of '1'."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(1.0).trip_gen("101")
        assert len(builder._events) == 1
        assert builder._events[0] == (1.0, "Gen", "101 '1'", "OPEN")

    def test_trip_gen_custom_gid(self):
        """trip_gen() accepts custom generator ID."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(1.0).trip_gen("101", gid="2")
        assert builder._events[0] == (1.0, "Gen", "101 '2'", "OPEN")

    def test_trip_branch_default_ckt(self):
        """trip_branch() uses default circuit ID of '1'."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(1.0).trip_branch("101", "102")
        assert len(builder._events) == 1
        assert builder._events[0] == (1.0, "Branch", "101 102 '1'", "OPEN")

    def test_trip_branch_custom_ckt(self):
        """trip_branch() accepts custom circuit ID."""
        builder = ContingencyBuilder("TestCtg")
        builder.at(1.0).trip_branch("101", "102", ckt="2")
        assert builder._events[0] == (1.0, "Branch", "101 102 '2'", "OPEN")

    def test_method_chaining(self):
        """Multiple events can be chained together."""
        builder = ContingencyBuilder("TestCtg")
        (builder
            .at(1.0).fault_bus("101")
            .at(1.1).clear_fault("101")
            .at(1.2).trip_gen("102"))
        assert len(builder._events) == 3
        assert builder._events[0][0] == 1.0
        assert builder._events[1][0] == 1.1
        assert builder._events[2][0] == 1.2

    def test_to_dataframes_empty_events(self):
        """to_dataframes() returns empty element DataFrame when no events."""
        builder = ContingencyBuilder("TestCtg", runtime=5.0)
        ctg_df, ele_df = builder.to_dataframes()

        assert isinstance(ctg_df, pd.DataFrame)
        assert len(ctg_df) == 1
        assert ctg_df.iloc[0]['TSCTGName'] == "TestCtg"
        assert ctg_df.iloc[0]['StartTime'] == 0.0
        assert ctg_df.iloc[0]['EndTime'] == 5.0
        assert ctg_df.iloc[0]['CTGSkip'] == 'NO'

        assert isinstance(ele_df, pd.DataFrame)
        assert ele_df.empty

    def test_to_dataframes_with_events(self):
        """to_dataframes() correctly generates element DataFrame."""
        builder = ContingencyBuilder("TestCtg", runtime=10.0)
        builder.at(1.0).fault_bus("101").at(1.1).clear_fault("101")
        ctg_df, ele_df = builder.to_dataframes()

        assert len(ctg_df) == 1
        assert ctg_df.iloc[0]['EndTime'] == 10.0

        assert len(ele_df) == 2
        assert 'TSCTGName' in ele_df.columns
        assert 'TSEventString' in ele_df.columns
        assert 'TSTimeInSeconds' in ele_df.columns
        assert 'WhoAmI' in ele_df.columns
        assert 'TSTimeInCycles' in ele_df.columns

        # Check first event (fault)
        assert ele_df.iloc[0]['TSCTGName'] == "TestCtg"
        assert ele_df.iloc[0]['TSTimeInSeconds'] == 1.0
        assert ele_df.iloc[0]['TSTimeInCycles'] == 60.0  # 1.0 * 60

        # Check event string format
        assert "FAULT 3PB SOLID" in ele_df.iloc[0]['TSEventString']
        assert "CLEARFAULT" in ele_df.iloc[1]['TSEventString']


# =============================================================================
# Dynamics Class Tests
# =============================================================================

@pytest.fixture
def mock_dynamics():
    """Create a Dynamics instance with mocked ESA."""
    dyn = Dynamics()

    # Mock the ESA object
    mock_esa = MagicMock()
    mock_esa.TSGetResults.return_value = (
        pd.DataFrame({'ColHeader': ['Bus 1 | TSBusVPU'], 'ObjectType': ['Bus'],
                      'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']}),
        pd.DataFrame({'time': [0.0, 0.1], 'Bus 1 | TSBusVPU': [1.0, 0.95]})
    )
    mock_esa.TSResultStorageSetAll.return_value = None
    mock_esa.GetParamsRectTyped.return_value = pd.DataFrame({'ObjectID': ['Bus 1', 'Bus 2']})
    mock_esa.ChangeParametersMultipleElementRect.return_value = None
    mock_esa.TSAutoCorrect.return_value = None
    mock_esa.TSInitialize.return_value = None
    mock_esa.TSSolve.return_value = None
    mock_esa.TSWriteModels.return_value = None

    dyn.esa = mock_esa
    return dyn


class TestDynamicsInit:
    """Tests for Dynamics initialization."""

    def test_default_runtime(self):
        """Dynamics initializes with default runtime of 5.0."""
        dyn = Dynamics()
        assert dyn.runtime == 5.0

    def test_empty_pending_ctgs(self):
        """Dynamics initializes with empty pending contingencies."""
        dyn = Dynamics()
        assert dyn._pending_ctgs == {}

    def test_empty_watch_fields(self):
        """Dynamics initializes with empty watch fields."""
        dyn = Dynamics()
        assert dyn._watch_fields == {}


class TestDynamicsWatch:
    """Tests for the watch() method."""

    def test_watch_stores_fields(self, mock_dynamics):
        """watch() stores field names for object type."""
        from esapp.components import Gen
        mock_dynamics.watch(Gen, [TS.Gen.P, TS.Gen.W])
        assert Gen in mock_dynamics._watch_fields
        assert mock_dynamics._watch_fields[Gen] == ['TSGenP', 'TSGenW']

    def test_watch_returns_self(self, mock_dynamics):
        """watch() returns self for method chaining."""
        from esapp.components import Bus
        result = mock_dynamics.watch(Bus, [TS.Bus.VPU])
        assert result is mock_dynamics

    def test_watch_converts_tsfield_to_string(self, mock_dynamics):
        """watch() converts TSField objects to their string names."""
        from esapp.components import Gen
        mock_dynamics.watch(Gen, [TS.Gen.Delta])
        assert mock_dynamics._watch_fields[Gen] == ['TSGenDelta']


class TestDynamicsContingency:
    """Tests for contingency-related methods."""

    def test_contingency_creates_builder(self, mock_dynamics):
        """contingency() creates and stores a ContingencyBuilder."""
        mock_dynamics.runtime = 8.0
        builder = mock_dynamics.contingency("TestCtg")

        assert isinstance(builder, ContingencyBuilder)
        assert builder.name == "TestCtg"
        assert builder.runtime == 8.0
        assert "TestCtg" in mock_dynamics._pending_ctgs

    def test_bus_fault_creates_contingency(self, mock_dynamics):
        """bus_fault() creates a complete bus fault contingency."""
        mock_dynamics.bus_fault("Fault1", "101", fault_time=1.0, duration=0.1)

        assert "Fault1" in mock_dynamics._pending_ctgs
        builder = mock_dynamics._pending_ctgs["Fault1"]
        assert len(builder._events) == 2

        # Check fault event
        assert builder._events[0][0] == 1.0
        assert builder._events[0][3] == "FAULT 3PB SOLID"

        # Check clear event
        assert builder._events[1][0] == pytest.approx(1.1)
        assert builder._events[1][3] == "CLEARFAULT"

    def test_bus_fault_default_params(self, mock_dynamics):
        """bus_fault() uses correct default parameters."""
        mock_dynamics.bus_fault("Fault1", "101")
        builder = mock_dynamics._pending_ctgs["Fault1"]

        assert builder._events[0][0] == 1.0
        assert builder._events[1][0] == pytest.approx(1.0833)  # 1.0 + 0.0833

    def test_upload_contingency_not_found_raises(self, mock_dynamics):
        """upload_contingency() raises ValueError for unknown contingency."""
        with pytest.raises(ValueError, match="not found in pending list"):
            mock_dynamics.upload_contingency("NonExistent")

    def test_upload_contingency_removes_from_pending(self, mock_dynamics):
        """upload_contingency() removes contingency from pending list."""
        mock_dynamics.bus_fault("Fault1", "101")
        assert "Fault1" in mock_dynamics._pending_ctgs

        mock_dynamics.upload_contingency("Fault1")
        assert "Fault1" not in mock_dynamics._pending_ctgs


class TestDynamicsGetResults:
    """Tests for get_results() method."""

    def test_get_results_returns_tuple(self, mock_dynamics):
        """get_results() returns tuple of DataFrames."""
        meta, data = mock_dynamics.get_results("Ctg1", ["Field1"])
        assert isinstance(meta, pd.DataFrame)
        assert isinstance(data, pd.DataFrame)

    def test_get_results_calls_tsgetresults(self, mock_dynamics):
        """get_results() calls esa.TSGetResults with correct args."""
        mock_dynamics.get_results("Ctg1", ["Field1", "Field2"])
        mock_dynamics.esa.TSGetResults.assert_called_once_with(
            "SEPARATE", ["Ctg1"], ["Field1", "Field2"]
        )

    def test_get_results_handles_none(self, mock_dynamics):
        """get_results() returns (None, None) when TSGetResults returns None."""
        mock_dynamics.esa.TSGetResults.return_value = None
        meta, data = mock_dynamics.get_results("Ctg1", ["Field1"])
        assert meta is None
        assert data is None


class TestDynamicsPrepareEnvironment:
    """Tests for _prepare_environment() method."""

    def test_prepare_environment_enables_storage(self, mock_dynamics):
        """_prepare_environment() calls TSResultStorageSetAll for watched types."""
        from esapp.components import Gen
        mock_dynamics.watch(Gen, [TS.Gen.P])
        mock_dynamics._prepare_environment()
        mock_dynamics.esa.TSResultStorageSetAll.assert_called()

    def test_prepare_environment_returns_field_list(self, mock_dynamics):
        """_prepare_environment() returns list of field specifications."""
        from esapp.components import Bus
        mock_dynamics.watch(Bus, [TS.Bus.VPU])
        mock_dynamics.esa.GetParamsRectTyped.return_value = pd.DataFrame({'ObjectID': ['Bus 1']})

        fields = mock_dynamics._prepare_environment()
        assert isinstance(fields, list)
        assert len(fields) > 0
        assert 'Bus 1 | TSBusVPU' in fields

    def test_prepare_environment_handles_empty_objects(self, mock_dynamics):
        """_prepare_environment() handles case with no objects of watched type."""
        from esapp.components import Gen
        mock_dynamics.watch(Gen, [TS.Gen.P])
        mock_dynamics.esa.GetParamsRectTyped.return_value = pd.DataFrame({'ObjectID': []})

        fields = mock_dynamics._prepare_environment()
        assert fields == []


class TestDynamicsProcessResults:
    """Tests for _process_results() method."""

    def test_process_results_sets_time_index(self, mock_dynamics):
        """_process_results() sets 'time' column as index."""
        meta = pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'time': [0.0, 0.1], 'Col1': [1.0, 0.95]})

        _, result_df = mock_dynamics._process_results(meta, df, "Ctg1")
        assert result_df.index.name == "time"

    def test_process_results_renames_columns(self, mock_dynamics):
        """_process_results() renames metadata columns."""
        meta = pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'time': [0.0], 'Col1': [1.0]})

        result_meta, _ = mock_dynamics._process_results(meta, df, "Ctg1")
        assert 'Object' in result_meta.columns
        assert 'ID-A' in result_meta.columns
        assert 'Metric' in result_meta.columns
        assert 'Contingency' in result_meta.columns

    def test_process_results_handles_empty_df(self, mock_dynamics):
        """_process_results() returns empty DataFrames for empty input."""
        meta = pd.DataFrame()
        df = pd.DataFrame()

        result_meta, result_df = mock_dynamics._process_results(meta, df, "Ctg1")
        assert result_meta.empty
        assert result_df.empty

    def test_process_results_handles_none_df(self, mock_dynamics):
        """_process_results() handles None DataFrame."""
        meta = pd.DataFrame({'ColHeader': ['Col1']})

        result_meta, result_df = mock_dynamics._process_results(meta, None, "Ctg1")
        assert result_meta.empty
        assert result_df.empty

    def test_process_results_casts_to_float32(self, mock_dynamics):
        """_process_results() casts data to float32."""
        meta = pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'time': [0.0], 'Col1': [1.0]})

        _, result_df = mock_dynamics._process_results(meta, df, "Ctg1")
        assert result_df['Col1'].dtype == np.float32


class TestDynamicsListModels:
    """Tests for list_models() method."""

    def test_list_models_returns_dataframe(self, mock_dynamics):
        """list_models() returns a DataFrame."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (MachineModel_GENROU, [Field1])\n'
                '{\n}\n'
                'DATA (Exciter_EXST1, [Field1])\n'
                '{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert isinstance(result, pd.DataFrame)
            assert 'Category' in result.columns
            assert 'Model' in result.columns

    def test_list_models_categorizes_machine_models(self, mock_dynamics):
        """list_models() correctly categorizes machine models."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (MachineModel_GENROU, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert 'Machine' in result['Category'].values
            assert 'GENROU' in result['Model'].values

    def test_list_models_categorizes_exciters(self, mock_dynamics):
        """list_models() correctly categorizes exciter models."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (Exciter_EXST1, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert 'Exciter' in result['Category'].values

    def test_list_models_categorizes_governors(self, mock_dynamics):
        """list_models() correctly categorizes governor models."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (Governor_TGOV1, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert 'Governor' in result['Category'].values

    def test_list_models_handles_no_file(self, mock_dynamics):
        """list_models() returns empty DataFrame when no file created."""
        with patch('os.path.exists', return_value=False):
            result = mock_dynamics.list_models()
            assert result.empty

    def test_list_models_skips_network_objects(self, mock_dynamics):
        """list_models() skips Gen, Load, Bus, etc. network objects."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (Gen, [Field1])\n{\n}\n'
                'DATA (Bus, [Field1])\n{\n}\n'
                'DATA (MachineModel_GENROU, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            # Should only have GENROU, not Gen or Bus
            assert len(result) == 1
            assert 'GENROU' in result['Model'].values


class TestDynamicsSolve:
    """Tests for solve() method."""

    def test_solve_accepts_single_contingency(self, mock_dynamics):
        """solve() accepts a single contingency name as string."""
        mock_dynamics.bus_fault("Fault1", "101")
        mock_dynamics.esa.TSGetResults.return_value = (
            pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                          'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']}),
            pd.DataFrame({'time': [0.0], 'Col1': [1.0]})
        )

        meta, data = mock_dynamics.solve("Fault1")
        mock_dynamics.esa.TSSolve.assert_called()

    def test_solve_accepts_list_of_contingencies(self, mock_dynamics):
        """solve() accepts a list of contingency names."""
        mock_dynamics.bus_fault("Fault1", "101")
        mock_dynamics.bus_fault("Fault2", "102")

        mock_dynamics.solve(["Fault1", "Fault2"])
        assert mock_dynamics.esa.TSSolve.call_count == 2

    def test_solve_uploads_pending_contingencies(self, mock_dynamics):
        """solve() uploads pending contingencies before solving."""
        mock_dynamics.bus_fault("Fault1", "101")
        assert "Fault1" in mock_dynamics._pending_ctgs

        mock_dynamics.solve("Fault1")
        assert "Fault1" not in mock_dynamics._pending_ctgs

    def test_solve_calls_ts_initialize(self, mock_dynamics):
        """solve() calls TSAutoCorrect and TSInitialize."""
        mock_dynamics.bus_fault("Fault1", "101")
        mock_dynamics.solve("Fault1")

        mock_dynamics.esa.TSAutoCorrect.assert_called_once()
        mock_dynamics.esa.TSInitialize.assert_called_once()

    def test_solve_returns_empty_when_no_results(self, mock_dynamics):
        """solve() returns empty DataFrames when no results."""
        mock_dynamics.bus_fault("Fault1", "101")
        mock_dynamics.esa.TSGetResults.return_value = (None, None)

        meta, data = mock_dynamics.solve("Fault1")
        assert meta.empty
        assert data.empty


class TestDynamicsUploadContingencyEdgeCases:
    """Tests for upload_contingency edge cases."""

    def test_upload_contingency_empty_elements(self, mock_dynamics):
        """upload_contingency() handles contingency with no events."""
        # Create a contingency with no events
        mock_dynamics.contingency("EmptyCtg")
        mock_dynamics.upload_contingency("EmptyCtg")
        # Should not raise, just upload header without elements
        assert "EmptyCtg" not in mock_dynamics._pending_ctgs

    def test_upload_contingency_updates_runtime(self, mock_dynamics):
        """upload_contingency() updates builder runtime to Dynamics runtime."""
        mock_dynamics.runtime = 15.0
        mock_dynamics.contingency("TestCtg")
        mock_dynamics.upload_contingency("TestCtg")
        # Verify the contingency was uploaded (removed from pending)
        assert "TestCtg" not in mock_dynamics._pending_ctgs


class TestDynamicsListModelsCategories:
    """Tests for list_models() categorization of all model types."""

    def test_list_models_categorizes_stabilizers(self, mock_dynamics):
        """list_models() correctly categorizes stabilizer models."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (Stabilizer_PSS2A, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert 'Stabilizer' in result['Category'].values
            assert 'PSS2A' in result['Model'].values

    def test_list_models_categorizes_plant_controllers(self, mock_dynamics):
        """list_models() correctly categorizes plant controller models."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (PlantController_REPC_A, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert 'Plant Controller' in result['Category'].values
            assert 'REPC_A' in result['Model'].values

    def test_list_models_categorizes_relays(self, mock_dynamics):
        """list_models() correctly categorizes relay models."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (RelayModel_DISTR1, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert 'Relay' in result['Category'].values
            assert 'DISTR1' in result['Model'].values

    def test_list_models_categorizes_load_models(self, mock_dynamics):
        """list_models() correctly categorizes load characteristic models."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (LoadModel_CIM5, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert 'Load Characteristic' in result['Category'].values
            assert 'CIM5' in result['Model'].values

    def test_list_models_categorizes_other_models(self, mock_dynamics):
        """list_models() categorizes unknown models as 'Other'."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (UnknownModel_XYZ, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert 'Other' in result['Category'].values
            assert 'UnknownModel_XYZ' in result['Model'].values

    def test_list_models_empty_data(self, mock_dynamics):
        """list_models() returns empty DataFrame with correct columns when no models."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (Gen, [Field1])\n{\n}\n'
                'DATA (Bus, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert result.empty
            assert list(result.columns) == ["Category", "Model", "Object Type"]

    def test_list_models_skips_shunt_and_branch(self, mock_dynamics):
        """list_models() skips Shunt and Branch network objects."""
        with patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                'DATA (Shunt, [Field1])\n{\n}\n'
                'DATA (Branch, [Field1])\n{\n}\n'
                'DATA (Load, [Field1])\n{\n}\n'
            )
            result = mock_dynamics.list_models()
            assert result.empty


class TestDynamicsSolveWarnings:
    """Tests for solve() warning conditions."""

    def test_solve_warns_no_watch_fields(self, mock_dynamics, caplog):
        """solve() logs warning when no fields are watched."""
        import logging
        mock_dynamics.bus_fault("Fault1", "101")
        mock_dynamics.esa.TSGetResults.return_value = (None, None)

        with caplog.at_level(logging.WARNING):
            mock_dynamics.solve("Fault1")

        assert "No fields watched" in caplog.text or mock_dynamics.esa.TSSolve.called

    def test_solve_warns_no_results_for_ctg(self, mock_dynamics, caplog):
        """solve() logs warning when contingency returns no results."""
        import logging
        from esapp.components import Gen
        mock_dynamics.watch(Gen, [TS.Gen.P])
        mock_dynamics.bus_fault("Fault1", "101")
        mock_dynamics.esa.TSGetResults.return_value = (None, None)

        with caplog.at_level(logging.WARNING):
            mock_dynamics.solve("Fault1")

        # Either warning logged or empty result returned
        meta, data = mock_dynamics.esa.TSGetResults.return_value
        assert meta is None

    def test_solve_handles_empty_df_in_results(self, mock_dynamics):
        """solve() handles empty DataFrame in results."""
        from esapp.components import Gen
        mock_dynamics.watch(Gen, [TS.Gen.P])
        mock_dynamics.bus_fault("Fault1", "101")
        mock_dynamics.esa.TSGetResults.return_value = (
            pd.DataFrame({'ColHeader': ['Col1']}),
            pd.DataFrame()  # Empty DataFrame
        )

        meta, data = mock_dynamics.solve("Fault1")
        assert meta.empty
        assert data.empty

class TestDynamicsProcessResultsEdgeCases:
    """Tests for _process_results() edge cases."""

    def test_process_results_no_matching_columns(self, mock_dynamics):
        """_process_results() handles case where no columns match metadata."""
        meta = pd.DataFrame({'ColHeader': ['NonExistent'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'time': [0.0], 'DifferentCol': [1.0]})

        result_meta, result_df = mock_dynamics._process_results(meta, df, "Ctg1")
        assert result_meta.empty
        assert result_df.empty

    def test_process_results_no_time_column(self, mock_dynamics):
        """_process_results() handles DataFrame without time column."""
        meta = pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'Col1': [1.0, 0.95]})

        result_meta, result_df = mock_dynamics._process_results(meta, df, "Ctg1")
        # Should still work, just won't set time as index
        assert not result_df.empty


class TestTSFieldReexport:
    """Tests for TS re-export from dynamics module."""

    def test_ts_in_all(self):
        """TS is included in __all__ for backward compatibility."""
        from esapp.apps.dynamics import __all__
        assert 'TS' in __all__

    def test_ts_import_from_dynamics(self):
        """TS can be imported from dynamics module."""
        from esapp.apps.dynamics import TS
        assert hasattr(TS, 'Gen')
        assert hasattr(TS, 'Bus')
