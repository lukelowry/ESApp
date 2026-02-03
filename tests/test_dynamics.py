"""
Unit tests for transient stability utilities.

These are **unit tests** that do NOT require PowerWorld Simulator. All
PowerWorld interactions are mocked. They test the pure-Python transient
stability utilities: ContingencyBuilder, SimAction (esapp.utils.contingency),
TSWatch, get_ts_results, process_ts_results (esapp.utils.dynamics), and
PowerWorld.ts_solve (esapp.workbench).

USAGE:
    pytest tests/test_dynamics.py -v
"""
import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from esapp.utils.contingency import ContingencyBuilder, SimAction
from esapp.utils.dynamics import TSWatch, get_ts_results, process_ts_results
from esapp.components import TS


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
# TSWatch Tests
# =============================================================================

class TestTSWatch:
    """Tests for the TSWatch utility class."""

    def test_init_empty(self):
        """TSWatch initializes with empty watch fields."""
        tsw = TSWatch()
        assert tsw.fields == {}

    def test_watch_stores_fields(self):
        """watch() stores field names for object type."""
        from esapp.components import Gen
        tsw = TSWatch()
        tsw.watch(Gen, [TS.Gen.P, TS.Gen.W])
        assert Gen in tsw.fields
        assert tsw.fields[Gen] == ['TSGenP', 'TSGenW']

    def test_watch_returns_self(self):
        """watch() returns self for method chaining."""
        from esapp.components import Bus
        tsw = TSWatch()
        result = tsw.watch(Bus, [TS.Bus.VPU])
        assert result is tsw

    def test_watch_converts_tsfield_to_string(self):
        """watch() converts TSField objects to their string names."""
        from esapp.components import Gen
        tsw = TSWatch()
        tsw.watch(Gen, [TS.Gen.Delta])
        assert tsw.fields[Gen] == ['TSGenDelta']

    def test_prepare_enables_storage(self):
        """prepare() calls TSResultStorageSetAll for watched types."""
        from esapp.components import Gen
        tsw = TSWatch()
        tsw.watch(Gen, [TS.Gen.P])

        mock_wb = MagicMock()
        mock_wb.__getitem__ = MagicMock(return_value=pd.DataFrame({'ObjectID': ['Gen 1']}))

        tsw.prepare(mock_wb)
        mock_wb.esa.TSResultStorageSetAll.assert_called()

    def test_prepare_returns_field_list(self):
        """prepare() returns list of field specifications."""
        from esapp.components import Bus
        tsw = TSWatch()
        tsw.watch(Bus, [TS.Bus.VPU])

        mock_wb = MagicMock()
        mock_wb.__getitem__ = MagicMock(return_value=pd.DataFrame({'ObjectID': ['Bus 1']}))

        fields = tsw.prepare(mock_wb)
        assert isinstance(fields, list)
        assert len(fields) > 0
        assert 'Bus 1 | TSBusVPU' in fields

    def test_prepare_handles_empty_objects(self):
        """prepare() handles case with no objects of watched type."""
        from esapp.components import Gen
        tsw = TSWatch()
        tsw.watch(Gen, [TS.Gen.P])

        mock_wb = MagicMock()
        mock_wb.__getitem__ = MagicMock(return_value=pd.DataFrame({'ObjectID': []}))

        fields = tsw.prepare(mock_wb)
        assert fields == []


# =============================================================================
# get_ts_results Tests
# =============================================================================

class TestGetTSResults:
    """Tests for the get_ts_results() function."""

    def test_returns_tuple(self):
        """get_ts_results() returns tuple of DataFrames."""
        mock_esa = MagicMock()
        mock_esa.TSGetResults.return_value = (
            pd.DataFrame({'ColHeader': ['Bus 1 | TSBusVPU']}),
            pd.DataFrame({'time': [0.0, 0.1], 'Bus 1 | TSBusVPU': [1.0, 0.95]})
        )
        meta, data = get_ts_results(mock_esa, "Ctg1", ["Field1"])
        assert isinstance(meta, pd.DataFrame)
        assert isinstance(data, pd.DataFrame)

    def test_calls_tsgetresults(self):
        """get_ts_results() calls esa.TSGetResults with correct args."""
        mock_esa = MagicMock()
        mock_esa.TSGetResults.return_value = (pd.DataFrame(), pd.DataFrame())
        get_ts_results(mock_esa, "Ctg1", ["Field1", "Field2"])
        mock_esa.TSGetResults.assert_called_once_with(
            "SEPARATE", ["Ctg1"], ["Field1", "Field2"]
        )

    def test_handles_none(self):
        """get_ts_results() returns (None, None) when TSGetResults returns None."""
        mock_esa = MagicMock()
        mock_esa.TSGetResults.return_value = None
        meta, data = get_ts_results(mock_esa, "Ctg1", ["Field1"])
        assert meta is None
        assert data is None


# =============================================================================
# process_ts_results Tests
# =============================================================================

class TestProcessTSResults:
    """Tests for the process_ts_results() function."""

    def test_sets_time_index(self):
        """process_ts_results() sets 'time' column as index."""
        meta = pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'time': [0.0, 0.1], 'Col1': [1.0, 0.95]})

        _, result_df = process_ts_results(meta, df, "Ctg1")
        assert result_df.index.name == "time"

    def test_renames_columns(self):
        """process_ts_results() renames metadata columns."""
        meta = pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'time': [0.0], 'Col1': [1.0]})

        result_meta, _ = process_ts_results(meta, df, "Ctg1")
        assert 'Object' in result_meta.columns
        assert 'ID-A' in result_meta.columns
        assert 'Metric' in result_meta.columns
        assert 'Contingency' in result_meta.columns

    def test_handles_empty_df(self):
        """process_ts_results() returns empty DataFrames for empty input."""
        meta = pd.DataFrame()
        df = pd.DataFrame()

        result_meta, result_df = process_ts_results(meta, df, "Ctg1")
        assert result_meta.empty
        assert result_df.empty

    def test_handles_none_df(self):
        """process_ts_results() handles None DataFrame."""
        meta = pd.DataFrame({'ColHeader': ['Col1']})

        result_meta, result_df = process_ts_results(meta, None, "Ctg1")
        assert result_meta.empty
        assert result_df.empty

    def test_casts_to_float32(self):
        """process_ts_results() casts data to float32."""
        meta = pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'time': [0.0], 'Col1': [1.0]})

        _, result_df = process_ts_results(meta, df, "Ctg1")
        assert result_df['Col1'].dtype == np.float32

    def test_no_matching_columns(self):
        """process_ts_results() handles case where no columns match metadata."""
        meta = pd.DataFrame({'ColHeader': ['NonExistent'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'time': [0.0], 'DifferentCol': [1.0]})

        result_meta, result_df = process_ts_results(meta, df, "Ctg1")
        assert result_meta.empty
        assert result_df.empty

    def test_no_time_column(self):
        """process_ts_results() handles DataFrame without time column."""
        meta = pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                             'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']})
        df = pd.DataFrame({'Col1': [1.0, 0.95]})

        result_meta, result_df = process_ts_results(meta, df, "Ctg1")
        assert not result_df.empty


# =============================================================================
# PowerWorld.ts_solve Tests
# =============================================================================

class TestTSSolve:
    """Tests for PowerWorld.ts_solve() method."""

    @pytest.fixture
    def mock_wb(self):
        """Create a mock workbench with ESA for ts_solve testing."""
        from esapp.workbench import PowerWorld

        pw = object.__new__(PowerWorld)
        pw.esa = MagicMock()
        pw.esa.TSAutoCorrect.return_value = None
        pw.esa.TSInitialize.return_value = None
        pw.esa.TSSolve.return_value = None
        pw.esa.TSGetResults.return_value = (
            pd.DataFrame({'ColHeader': ['Col1'], 'ObjectType': ['Bus'],
                          'PrimaryKey': ['1'], 'SecondaryKey': [None], 'VariableName': ['VPU']}),
            pd.DataFrame({'time': [0.0, 0.1], 'Col1': [1.0, 0.95]})
        )
        return pw

    def test_accepts_single_contingency(self, mock_wb):
        """ts_solve() accepts a single contingency name as string."""
        meta, data = mock_wb.ts_solve("Fault1", ["Col1"])
        mock_wb.esa.TSSolve.assert_called_once_with("Fault1")

    def test_accepts_list_of_contingencies(self, mock_wb):
        """ts_solve() accepts a list of contingency names."""
        mock_wb.ts_solve(["Fault1", "Fault2"], ["Col1"])
        assert mock_wb.esa.TSSolve.call_count == 2

    def test_calls_ts_initialize(self, mock_wb):
        """ts_solve() calls TSAutoCorrect and TSInitialize."""
        mock_wb.ts_solve("Fault1", ["Col1"])
        mock_wb.esa.TSAutoCorrect.assert_called_once()
        mock_wb.esa.TSInitialize.assert_called_once()

    def test_returns_empty_when_no_results(self, mock_wb):
        """ts_solve() returns empty DataFrames when no results."""
        mock_wb.esa.TSGetResults.return_value = (None, None)
        meta, data = mock_wb.ts_solve("Fault1", ["Col1"])
        assert meta.empty
        assert data.empty

    def test_handles_empty_df_in_results(self, mock_wb):
        """ts_solve() handles empty DataFrame in results."""
        mock_wb.esa.TSGetResults.return_value = (
            pd.DataFrame({'ColHeader': ['Col1']}),
            pd.DataFrame()
        )
        meta, data = mock_wb.ts_solve("Fault1", ["Col1"])
        assert meta.empty
        assert data.empty

    def test_warns_no_fields(self, mock_wb, caplog):
        """ts_solve() logs warning when no fields are provided."""
        import logging
        mock_wb.esa.TSGetResults.return_value = (None, None)
        with caplog.at_level(logging.WARNING):
            mock_wb.ts_solve("Fault1", [])
        assert "No fields provided" in caplog.text


# =============================================================================
# TS Component Import Tests
# =============================================================================

class TestTSImport:
    """Tests for TS import from esapp.components."""

    def test_ts_has_gen(self):
        """TS has Gen attribute."""
        assert hasattr(TS, 'Gen')

    def test_ts_has_bus(self):
        """TS has Bus attribute."""
        assert hasattr(TS, 'Bus')
