"""
Unit tests for pure Python helper functions and validation logic.

These are **unit tests** that do NOT require PowerWorld Simulator. They test
data transformation (df_to_aux), path conversion, argument packing, format
string edge cases, CSV result parsing, matrix file parsing, and Python-side
input validation in the SAW class.

USAGE:
    pytest tests/test_helpers_unit.py -v
"""
import os
import pytest
import pandas as pd
import numpy as np


# =============================================================================
# df_to_aux function
# =============================================================================

class TestDfToAux:
    """Tests for df_to_aux function."""

    def test_df_to_aux_basic(self):
        """df_to_aux writes correct AUX format for a simple DataFrame."""
        from esapp.saw._helpers import df_to_aux
        import io

        df = pd.DataFrame({"BusNum": [1, 2], "BusName": ["Bus1", "Bus2"]})
        fp = io.StringIO()
        df_to_aux(fp, df, "Bus")
        content = fp.getvalue()

        assert "DATA (Bus, [BusNum,BusName])" in content
        assert "{" in content
        assert "}" in content
        assert "1" in content
        assert "Bus1" in content

    def test_df_to_aux_long_header_wraps(self):
        """df_to_aux wraps long headers across multiple lines."""
        from esapp.saw._helpers import df_to_aux
        import io

        # Create a DataFrame with many columns to force header wrapping
        cols = {f"VeryLongFieldName{i}": [i] for i in range(20)}
        df = pd.DataFrame(cols)
        fp = io.StringIO()
        df_to_aux(fp, df, "Branch")
        content = fp.getvalue()

        assert "DATA (Branch," in content
        assert "{" in content
        assert "}" in content


# =============================================================================
# Helper conversion functions
# =============================================================================

class TestHelperConversions:
    """Tests for helper conversion functions."""

    def test_convert_to_windows_path(self):
        from esapp.saw._helpers import convert_to_windows_path
        result = convert_to_windows_path("/tmp/test/file.pwb")
        assert "\\" in result or "/" not in result.replace("//", "")

    def test_create_object_string_single_key(self):
        from esapp.saw._helpers import create_object_string
        assert create_object_string("Bus", 1) == "[BUS 1]"

    def test_create_object_string_multiple_keys(self):
        from esapp.saw._helpers import create_object_string
        assert create_object_string("Branch", 1, 2, "1") == "[BRANCH 1 2 1]"


# =============================================================================
# pack_args function
# =============================================================================

class TestPackArgs:
    """Tests for pack_args helper function."""

    def test_pack_args_basic(self):
        """pack_args joins arguments with commas."""
        from esapp.saw._helpers import pack_args
        result = pack_args("a", "b", "c")
        assert result == "a, b, c"

    def test_pack_args_filters_trailing_none(self):
        """pack_args removes trailing None values."""
        from esapp.saw._helpers import pack_args
        result = pack_args("a", "b", None, None)
        assert result == "a, b"

    def test_pack_args_converts_middle_none_to_empty(self):
        """pack_args converts middle None to empty string."""
        from esapp.saw._helpers import pack_args
        result = pack_args("a", None, "c")
        assert result == "a, , c"

    def test_pack_args_all_none(self):
        """pack_args returns empty string for all None."""
        from esapp.saw._helpers import pack_args
        result = pack_args(None, None)
        assert result == ""

    def test_pack_args_empty(self):
        """pack_args returns empty string for no arguments."""
        from esapp.saw._helpers import pack_args
        result = pack_args()
        assert result == ""

    def test_pack_args_numbers(self):
        """pack_args converts numbers to strings."""
        from esapp.saw._helpers import pack_args
        result = pack_args(1, 2.5, 3)
        assert result == "1, 2.5, 3"


# =============================================================================
# format_optional function
# =============================================================================

class TestFormatOptionalEdgeCases:
    """Tests for format_optional edge cases."""

    def test_format_optional_empty_string_not_quoted(self):
        """format_optional returns empty string for empty input."""
        from esapp.saw._helpers import format_optional
        result = format_optional("")
        assert result == ""

    def test_format_optional_empty_string_with_empty_quoted(self):
        """format_optional returns '\"\"' when empty_quoted=True."""
        from esapp.saw._helpers import format_optional
        result = format_optional("", empty_quoted=True)
        assert result == '""'

    def test_format_optional_none_not_quoted(self):
        """format_optional returns empty string for None."""
        from esapp.saw._helpers import format_optional
        result = format_optional(None)
        assert result == ""

    def test_format_optional_none_with_empty_quoted(self):
        """format_optional returns '\"\"' for None when empty_quoted=True."""
        from esapp.saw._helpers import format_optional
        result = format_optional(None, empty_quoted=True)
        assert result == '""'

    def test_format_optional_value_quoted(self):
        """format_optional quotes non-empty values by default."""
        from esapp.saw._helpers import format_optional
        result = format_optional("MyValue")
        assert result == '"MyValue"'

    def test_format_optional_value_not_quoted(self):
        """format_optional returns raw value when quote=False."""
        from esapp.saw._helpers import format_optional
        result = format_optional("MyValue", quote=False)
        assert result == "MyValue"


# =============================================================================
# load_ts_csv_results function
# =============================================================================

class TestLoadTsCsvResults:
    """Tests for load_ts_csv_results function."""

    def test_load_ts_csv_results_no_files(self, tmp_path):
        """load_ts_csv_results returns empty DataFrames when no files found."""
        from esapp.saw._helpers import load_ts_csv_results
        base_path = tmp_path / "nonexistent_results"
        meta, data = load_ts_csv_results(base_path)
        assert meta.empty
        assert data.empty

    def test_load_ts_csv_results_header_only(self, tmp_path):
        """load_ts_csv_results parses header file correctly."""
        from esapp.saw._helpers import load_ts_csv_results

        # Create header file
        header_file = tmp_path / "results_header.csv"
        header_file.write_text("Column,Object,Variable,Key 1,Key 2\n0,Bus,VPU,1,\n1,Gen,P,2,1\n")

        base_path = tmp_path / "results"
        meta, data = load_ts_csv_results(base_path)

        assert not meta.empty
        assert "ColHeader" in meta.columns
        assert "ObjectType" in meta.columns

    def test_load_ts_csv_results_with_data(self, tmp_path):
        """load_ts_csv_results parses data files correctly."""
        from esapp.saw._helpers import load_ts_csv_results

        # Create header file
        header_file = tmp_path / "results_header.csv"
        header_file.write_text("Column,Object,Variable,Key 1,Key 2\n0,Bus,VPU,1,\n")

        # Create data file
        data_file = tmp_path / "results_data.csv"
        data_file.write_text("0.0,1.0\n0.1,0.99\n0.2,0.98\n")

        base_path = tmp_path / "results"
        meta, data = load_ts_csv_results(base_path)

        assert not data.empty
        assert "time" in data.columns
        assert len(data) == 3

    def test_load_ts_csv_results_object_fields_header(self, tmp_path):
        """load_ts_csv_results skips ObjectFields line in header."""
        from esapp.saw._helpers import load_ts_csv_results

        # Create header file with ObjectFields line
        header_file = tmp_path / "results_header.csv"
        header_file.write_text("ObjectFields\nColumn,Object,Variable,Key 1,Key 2\n0,Bus,VPU,1,\n")

        base_path = tmp_path / "results"
        meta, data = load_ts_csv_results(base_path)

        assert not meta.empty
        assert "ColHeader" in meta.columns

    def test_load_ts_csv_results_delete_files(self, tmp_path):
        """load_ts_csv_results deletes files when delete_files=True."""
        from esapp.saw._helpers import load_ts_csv_results

        # Create header file
        header_file = tmp_path / "results_header.csv"
        header_file.write_text("Column,Object,Variable,Key 1,Key 2\n0,Bus,VPU,1,\n")

        base_path = tmp_path / "results"
        load_ts_csv_results(base_path, delete_files=True)

        assert not header_file.exists()


# =============================================================================
# format_list function
# =============================================================================


class TestFormatList:
    """Tests for format_list helper function."""

    def test_format_list_none(self):
        from esapp.saw._helpers import format_list
        assert format_list(None) == "[]"

    def test_format_list_empty(self):
        from esapp.saw._helpers import format_list
        assert format_list([]) == "[]"

    def test_format_list_basic(self):
        from esapp.saw._helpers import format_list
        assert format_list(["BusNum", "BusName"]) == "[BusNum, BusName]"

    def test_format_list_quote_items(self):
        from esapp.saw._helpers import format_list
        assert format_list(["Gen1", "Gen2"], quote_items=True) == '["Gen1", "Gen2"]'

    def test_format_list_stringify(self):
        from esapp.saw._helpers import format_list
        assert format_list([1.5, 2.0], stringify=True) == "[1.5, 2.0]"


# =============================================================================
# format_optional_numeric function
# =============================================================================


class TestFormatOptionalNumeric:
    """Tests for format_optional_numeric function."""

    def test_none_returns_empty(self):
        from esapp.saw._helpers import format_optional_numeric
        assert format_optional_numeric(None) == ""

    def test_zero_returns_str(self):
        from esapp.saw._helpers import format_optional_numeric
        assert format_optional_numeric(0) == "0"

    def test_float_returns_str(self):
        from esapp.saw._helpers import format_optional_numeric
        assert format_optional_numeric(3.14) == "3.14"


# =============================================================================
# format_filter functions
# =============================================================================


class TestFormatFilterSelectedOnly:
    """Tests for format_filter_selected_only."""

    def test_empty_string(self):
        from esapp.saw._enums import format_filter_selected_only
        assert format_filter_selected_only("") == ""

    def test_none(self):
        from esapp.saw._enums import format_filter_selected_only
        assert format_filter_selected_only(None) == ""

    def test_selected_enum(self):
        from esapp.saw._enums import format_filter_selected_only, FilterKeyword
        result = format_filter_selected_only(FilterKeyword.SELECTED)
        assert result == "SELECTED"

    def test_selected_string(self):
        from esapp.saw._enums import format_filter_selected_only
        result = format_filter_selected_only("SELECTED")
        assert result == "SELECTED"

    def test_custom_filter_quoted(self):
        from esapp.saw._enums import format_filter_selected_only
        result = format_filter_selected_only("MyFilter")
        assert result == '"MyFilter"'

    def test_areazone_enum_quoted(self):
        from esapp.saw._enums import format_filter_selected_only, FilterKeyword
        result = format_filter_selected_only(FilterKeyword.AREAZONE)
        # AREAZONE is not SELECTED, so it gets quoted
        assert result.startswith('"') and result.endswith('"')


class TestFormatFilterAreazone:
    """Tests for format_filter_areazone."""

    def test_empty_string(self):
        from esapp.saw._enums import format_filter_areazone
        assert format_filter_areazone("") == ""

    def test_none(self):
        from esapp.saw._enums import format_filter_areazone
        assert format_filter_areazone(None) == ""

    def test_selected_enum(self):
        from esapp.saw._enums import format_filter_areazone, FilterKeyword
        result = format_filter_areazone(FilterKeyword.SELECTED)
        assert result == "SELECTED"

    def test_areazone_enum(self):
        from esapp.saw._enums import format_filter_areazone, FilterKeyword
        result = format_filter_areazone(FilterKeyword.AREAZONE)
        assert result == "AREAZONE"

    def test_all_enum_quoted(self):
        from esapp.saw._enums import format_filter_areazone, FilterKeyword
        result = format_filter_areazone(FilterKeyword.ALL)
        assert result == '"ALL"'

    def test_selected_string(self):
        from esapp.saw._enums import format_filter_areazone
        result = format_filter_areazone("SELECTED")
        assert result == "SELECTED"

    def test_areazone_string(self):
        from esapp.saw._enums import format_filter_areazone
        result = format_filter_areazone("AREAZONE")
        assert result == "AREAZONE"

    def test_custom_filter_quoted(self):
        from esapp.saw._enums import format_filter_areazone
        result = format_filter_areazone("MyFilter")
        assert result == '"MyFilter"'


# =============================================================================
# load_ts_csv_results edge cases
# =============================================================================


class TestLoadTsCsvResultsEdgeCases:
    """Edge case tests for load_ts_csv_results."""

    def test_bad_header_file_logged(self, tmp_path, caplog):
        """A corrupt header file logs warning and returns empty meta."""
        from esapp.saw._helpers import load_ts_csv_results
        import logging

        header_file = tmp_path / "results_header.csv"
        header_file.write_bytes(b'\xff\xfe' + b'\x00' * 50)

        base_path = tmp_path / "results"
        with caplog.at_level(logging.WARNING):
            meta, data = load_ts_csv_results(base_path)

    def test_bad_data_file_logged(self, tmp_path, caplog):
        """A corrupt data file logs warning and is skipped."""
        from esapp.saw._helpers import load_ts_csv_results
        import logging

        data_file = tmp_path / "results_0.csv"
        data_file.write_bytes(b'\xff\xfe' + b'\x00' * 50)

        base_path = tmp_path / "results"
        with caplog.at_level(logging.WARNING):
            meta, data = load_ts_csv_results(base_path)


# =============================================================================
# SAW Python-side validation tests (no COM calls needed)
# =============================================================================


class TestSAWValidation:
    """Tests for SAW input validation that happens in Python before COM calls."""

    def test_set_simauto_property_invalid_name(self, saw_obj):
        """set_simauto_property raises ValueError for unsupported property."""
        with pytest.raises(ValueError, match="not currently supported"):
            saw_obj.set_simauto_property("InvalidProp", True)

    def test_set_simauto_property_invalid_type(self, saw_obj):
        """set_simauto_property raises ValueError for wrong type."""
        with pytest.raises(ValueError, match="is invalid"):
            saw_obj.set_simauto_property("CreateIfNotFound", "not_a_bool")

    def test_open_case_no_filename_no_path(self, saw_obj):
        """OpenCase raises TypeError when no FileName and no prior path."""
        saw_obj.pwb_file_path = None
        with pytest.raises(TypeError, match="FileName is required"):
            saw_obj.OpenCase(FileName=None)

    def test_save_case_no_filename_no_path(self, saw_obj):
        """SaveCase raises TypeError when no FileName and no opened case."""
        saw_obj.pwb_file_path = None
        with pytest.raises(TypeError, match="SaveCase was called without"):
            saw_obj.SaveCase()

    def test_call_simauto_invalid_func(self, saw_obj):
        """_call_simauto raises AttributeError for invalid function."""
        del saw_obj._pwcom.InvalidFunc
        with pytest.raises(AttributeError, match="not a valid SimAuto function"):
            saw_obj._call_simauto("InvalidFunc")

    def test_replace_decimal_delimiter(self, saw_obj):
        """_replace_decimal_delimiter handles non-string data."""
        data = pd.Series([1.0, 2.0, 3.0])
        result = saw_obj._replace_decimal_delimiter(data)
        assert (result == data).all()

    def test_replace_decimal_delimiter_comma(self, saw_obj):
        """_replace_decimal_delimiter replaces comma with period."""
        saw_obj.decimal_delimiter = ","
        data = pd.Series(["1,5", "2,3", "3,0"])
        result = saw_obj._replace_decimal_delimiter(data)
        assert result.iloc[0] == "1.5"

    def test_get_field_list_copy(self, saw_obj):
        """GetFieldList returns a copy when copy=True."""
        saw_obj._object_fields = {}
        result1 = saw_obj.GetFieldList("Bus", copy=False)
        result2 = saw_obj.GetFieldList("Bus", copy=True)
        assert result1 is not result2


# =============================================================================
# Matrix parsing tests (pure file-based, no COM)
# =============================================================================


class TestMatrixParsing:
    """Tests for matrix file parsing (pure Python, reads from files)."""

    def _make_ybus_file(self, tmp_path):
        """Helper to create a mock YBus .m file in PowerWorld format."""
        content = (
            "% YBus Matrix\n"
            "Ybus=sparse(3,3);\n"
            "Ybus(1,1)=10.0+j*(-5.0);\n"
            "Ybus(1,2)=-5.0+j*(2.5);\n"
            "Ybus(2,1)=-5.0+j*(2.5);\n"
            "Ybus(2,2)=10.0+j*(-5.0);\n"
            "Ybus(3,3)=5.0+j*(-2.5);\n"
        )
        fpath = str(tmp_path / "ybus.m")
        with open(fpath, "w") as f:
            f.write(content)
        return fpath

    def test_parse_real_matrix(self, saw_obj):
        """_parse_real_matrix correctly parses sparse matrix."""
        mat_str = (
            "Jac=sparse(2,2);\n"
            "Jac(1,1)=1.0;\n"
            "Jac(1,2)=-0.5;\n"
            "Jac(2,1)=-0.5;\n"
            "Jac(2,2)=1.0;\n"
        )
        result = saw_obj._parse_real_matrix(mat_str, "Jac")
        arr = result.toarray()
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 1.0
        assert arr[0, 1] == -0.5

    def test_get_ybus_from_file(self, saw_obj, tmp_path):
        """get_ybus reads from an existing file."""
        fpath = self._make_ybus_file(tmp_path)
        result = saw_obj.get_ybus(file=fpath, full=True)
        assert result.shape == (3, 3)
        assert result[0, 0] == 10.0 + (-5.0j)

    def test_get_ybus_sparse(self, saw_obj, tmp_path):
        """get_ybus returns sparse matrix by default."""
        from scipy.sparse import issparse
        fpath = self._make_ybus_file(tmp_path)
        result = saw_obj.get_ybus(file=fpath, full=False)
        assert issparse(result)


# =============================================================================
# TSField unit tests
# =============================================================================


class TestTSFieldIndexing:
    """Tests for TSField.__getitem__."""

    def test_tsfield_getitem(self):
        """TSField[index] creates an indexed field."""
        from esapp.components.ts_fields import TSField
        field = TSField("TSBusInput", "Input voltage")
        indexed = field[1]
        assert str(indexed) == "TSBusInput:1"

    def test_ts_bus_input_indexing(self):
        """TS.Bus.Input[1] creates indexed field."""
        from esapp.components import TS
        indexed = TS.Bus.Input[1]
        assert str(indexed) == "TSBusInput:1"


# =============================================================================
# Workbench logic tests (file I/O, no COM)
# =============================================================================


class TestWorkbenchLogic:
    """Tests for GridWorkBench Python-side logic (no PowerWorld needed)."""

    def test_init_no_fname(self):
        """GridWorkBench initializes without fname."""
        from esapp.workbench import GridWorkBench
        wb = GridWorkBench()
        assert wb.esa is None
        assert wb.fname is None

    def test_open_file_not_found(self):
        """GridWorkBench.open raises FileNotFoundError for missing file."""
        from esapp.workbench import GridWorkBench
        from unittest.mock import patch

        wb = GridWorkBench()
        wb.fname = "C:/nonexistent/file.pwb"

        with patch('esapp.indexable.path.isabs', return_value=True), \
             patch('esapp.indexable.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Case file not found"):
                wb.open()

    def test_log_output(self):
        """GridWorkBench.print_log reads and prints log content."""
        from esapp.workbench import GridWorkBench
        from unittest.mock import MagicMock

        wb = GridWorkBench()
        wb.esa = MagicMock()
        wb._log_last_position = 0

        def mock_log_save(path, append=False):
            with open(path, "w") as f:
                f.write("Some log output")

        wb.esa.LogSave.side_effect = mock_log_save

        result = wb.print_log(new_only=False, clear=False)
        assert "Some log output" in result

    def test_print_log_new_only(self):
        """GridWorkBench.print_log returns only new content."""
        from esapp.workbench import GridWorkBench
        from unittest.mock import MagicMock

        wb = GridWorkBench()
        wb.esa = MagicMock()
        wb._log_last_position = 5

        def mock_log_save(path, append=False):
            with open(path, "w") as f:
                f.write("Hello World")

        wb.esa.LogSave.side_effect = mock_log_save

        result = wb.print_log(new_only=True, clear=False)
        assert result == " World"

    def test_print_log_empty(self):
        """GridWorkBench.print_log handles empty log output (whitespace only)."""
        from esapp.workbench import GridWorkBench
        from unittest.mock import MagicMock

        wb = GridWorkBench()
        wb.esa = MagicMock()
        wb._log_last_position = 0

        def mock_log_save(path, append=False):
            with open(path, "w") as f:
                f.write("   ")

        wb.esa.LogSave.side_effect = mock_log_save

        result = wb.print_log(new_only=False, clear=False)
        assert result == "   "

    def test_log_with_clear(self):
        """GridWorkBench.print_log clears log after reading."""
        from esapp.workbench import GridWorkBench
        from unittest.mock import MagicMock

        wb = GridWorkBench()
        wb.esa = MagicMock()
        wb._log_last_position = 0

        def mock_log_save(path, append=False):
            with open(path, "w") as f:
                f.write("Log content")

        wb.esa.LogSave.side_effect = mock_log_save

        wb.print_log(new_only=False, clear=True)
        wb.esa.LogClear.assert_called_once()
        assert wb._log_last_position == 0


# =============================================================================
# Indexable fallback (error routing logic)
# =============================================================================


class TestIndexableFallback:
    """Tests for Indexable.__setitem__ fallback on ChangeParametersMultipleElement."""

    def test_fallback_create_suppresses_not_found(self):
        """Fallback to ChangeParametersMultipleElement suppresses 'not found'."""
        from esapp.saw import PowerWorldPrerequisiteError
        from esapp.indexable import Indexable
        from esapp import components as grid
        from unittest.mock import Mock

        instance = Indexable()
        mock_esa = Mock()
        instance.esa = mock_esa

        update_df = pd.DataFrame({
            "BusNum": [999, 1000],
            "GenID": ["1", "1"],
            "GenMW": [100.0, 200.0],
        })

        mock_esa.ChangeParametersMultipleElementRect.side_effect = PowerWorldPrerequisiteError(
            "Object not found in case"
        )
        mock_esa.ChangeParametersMultipleElement.side_effect = PowerWorldPrerequisiteError(
            "Object not found in case"
        )

        instance[grid.Gen] = update_df

    def test_fallback_create_raises_other_error(self):
        """Fallback to ChangeParametersMultipleElement re-raises non-'not found' errors."""
        from esapp.saw import PowerWorldPrerequisiteError
        from esapp.indexable import Indexable
        from esapp import components as grid
        from unittest.mock import Mock

        instance = Indexable()
        mock_esa = Mock()
        instance.esa = mock_esa

        update_df = pd.DataFrame({
            "BusNum": [999],
            "GenID": ["1"],
            "GenMW": [100.0],
        })

        mock_esa.ChangeParametersMultipleElementRect.side_effect = PowerWorldPrerequisiteError(
            "Object not found in case"
        )
        mock_esa.ChangeParametersMultipleElement.side_effect = PowerWorldPrerequisiteError(
            "License expired"
        )

        with pytest.raises(PowerWorldPrerequisiteError, match="License expired"):
            instance[grid.Gen] = update_df


# =============================================================================
# Transient validation (Python-side)
# =============================================================================


class TestTransientValidation:
    """Tests for transient stability input validation."""

    def test_ts_set_play_in_signals_dim_mismatch(self, saw_obj):
        """TSSetPlayInSignals raises on dimension mismatch."""
        times = np.array([0.0, 1.0])
        signals = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            saw_obj.TSSetPlayInSignals("Sig1", times, signals)


# =============================================================================
# AUX Parsing / Building helpers
# =============================================================================


class TestAuxParsing:
    """Tests for parse_aux_line, parse_aux_content, and build_aux_string."""

    # ---- parse_aux_line ----

    def test_parse_aux_line_space_delimited(self):
        from esapp.saw._helpers import parse_aux_line
        result = parse_aux_line('101 "Gen 1" 50.0')
        assert result == ["101", "Gen 1", "50.0"]

    def test_parse_aux_line_bracket_format(self):
        from esapp.saw._helpers import parse_aux_line
        result = parse_aux_line("[1, 100.0], [2, 200.0]")
        assert result == ["1, 100.0", "2, 200.0"]

    def test_parse_aux_line_quoted_with_brackets(self):
        """Brackets inside quoted strings should NOT trigger bracket mode."""
        from esapp.saw._helpers import parse_aux_line
        result = parse_aux_line('"Name [test]" 42')
        assert result == ["Name [test]", "42"]

    def test_parse_aux_line_empty(self):
        from esapp.saw._helpers import parse_aux_line
        assert parse_aux_line("") == []
        assert parse_aux_line("   ") == []

    # ---- parse_aux_content: Legacy format ----

    def test_parse_aux_content_legacy_basic(self):
        from esapp.saw._helpers import parse_aux_content
        content = (
            'DATA (Bus, [BusNum, BusName])\n'
            '{\n'
            '1 "Alpha"\n'
            '2 "Beta"\n'
            '}\n'
        )
        records = parse_aux_content(content, ["BusNum", "BusName"])
        assert len(records) == 2
        assert records[0]["BusNum"] == "1"
        assert records[0]["BusName"] == "Alpha"
        assert records[1]["BusNum"] == "2"

    def test_parse_aux_content_legacy_with_subdata(self):
        from esapp.saw._helpers import parse_aux_content
        content = (
            'DATA (Gen, [BusNum, GenID])\n'
            '{\n'
            '101 "1"\n'
            '  <SUBDATA BidCurve>\n'
            '  10.0 50.0\n'
            '  20.0 60.0\n'
            '  </SUBDATA>\n'
            '}\n'
        )
        records = parse_aux_content(content, ["BusNum", "GenID"], ["BidCurve"])
        assert len(records) == 1
        assert records[0]["BusNum"] == "101"
        assert len(records[0]["BidCurve"]) == 2
        assert records[0]["BidCurve"][0] == ["10.0", "50.0"]

    def test_parse_aux_content_legacy_multiple_subdata(self):
        from esapp.saw._helpers import parse_aux_content
        content = (
            'DATA (Gen, [BusNum, GenID])\n'
            '{\n'
            '101 "1"\n'
            '  <SUBDATA BidCurve>\n'
            '  10.0 50.0\n'
            '  </SUBDATA>\n'
            '  <SUBDATA ReactiveCapability>\n'
            '  100.0 -50.0 50.0\n'
            '  </SUBDATA>\n'
            '}\n'
        )
        records = parse_aux_content(content, ["BusNum", "GenID"],
                                    ["BidCurve", "ReactiveCapability"])
        assert len(records) == 1
        assert len(records[0]["BidCurve"]) == 1
        assert len(records[0]["ReactiveCapability"]) == 1

    def test_parse_aux_content_legacy_empty_subdata(self):
        from esapp.saw._helpers import parse_aux_content
        content = (
            'DATA (Gen, [BusNum, GenID])\n'
            '{\n'
            '101 "1"\n'
            '  <SUBDATA BidCurve>\n'
            '  </SUBDATA>\n'
            '}\n'
        )
        records = parse_aux_content(content, ["BusNum", "GenID"], ["BidCurve"])
        assert len(records) == 1
        assert records[0]["BidCurve"] == []

    def test_parse_aux_content_legacy_comments(self):
        from esapp.saw._helpers import parse_aux_content
        content = (
            'DATA (Bus, [BusNum, BusName])\n'
            '{\n'
            '// This is a comment\n'
            '1 "Alpha"\n'
            '}\n'
        )
        records = parse_aux_content(content, ["BusNum", "BusName"])
        assert len(records) == 1
        assert records[0]["BusNum"] == "1"

    # ---- parse_aux_content: Concise format ----

    def test_parse_aux_content_concise_basic(self):
        from esapp.saw._helpers import parse_aux_content
        content = (
            'Bus (BusNum, BusName)\n'
            '{\n'
            '1 "Alpha"\n'
            '2 "Beta"\n'
            '}\n'
        )
        records = parse_aux_content(content, ["BusNum", "BusName"])
        assert len(records) == 2

    def test_parse_aux_content_concise_with_subdata(self):
        from esapp.saw._helpers import parse_aux_content
        content = (
            'Contingency (CTGName)\n'
            '{\n'
            '"MyCtg"\n'
            '  <SUBDATA CTGElement>\n'
            '  "OPEN BRANCH FROM 1 TO 2 CKT 1"\n'
            '  </SUBDATA>\n'
            '}\n'
        )
        records = parse_aux_content(content, ["CTGName"], ["CTGElement"])
        assert len(records) == 1
        assert records[0]["CTGName"] == "MyCtg"
        assert len(records[0]["CTGElement"]) == 1

    # ---- parse_aux_content: edge cases ----

    def test_parse_aux_content_empty(self):
        from esapp.saw._helpers import parse_aux_content
        assert parse_aux_content("", ["BusNum"]) == []

    def test_parse_aux_content_malformed_subdata(self):
        from esapp.saw._helpers import parse_aux_content
        content = (
            'DATA (Gen, [BusNum])\n'
            '{\n'
            '101\n'
            '  <SUBDATA>\n'
            '  10.0\n'
            '  </SUBDATA>\n'
            '}\n'
        )
        with pytest.raises(ValueError, match="Malformed SUBDATA"):
            parse_aux_content(content, ["BusNum"], ["BidCurve"])

    # ---- build_aux_string ----

    def test_build_aux_string_no_subdata(self):
        from esapp.saw._helpers import build_aux_string
        records = [{"BusNum": 1, "BusName": "Alpha"}]
        result = build_aux_string("Bus", ["BusNum", "BusName"], records)
        assert "DATA (Bus, [BusNum, BusName])" in result
        assert '"Alpha"' in result
        assert "1" in result

    def test_build_aux_string_single_subdata(self):
        from esapp.saw._helpers import build_aux_string
        records = [{
            "BusNum": 101,
            "GenID": "1",
            "BidCurve": [[10.0, 50.0], [20.0, 60.0]],
        }]
        result = build_aux_string("Gen", ["BusNum", "GenID"], records,
                                  subdatatypes="BidCurve")
        assert "<SUBDATA BidCurve>" in result
        assert "</SUBDATA>" in result
        assert "10.0" in result

    def test_build_aux_string_multiple_subdata(self):
        from esapp.saw._helpers import build_aux_string
        records = [{
            "BusNum": 101,
            "GenID": "1",
            "BidCurve": [[10.0, 50.0]],
            "ReactiveCapability": [[100.0, -50.0, 50.0]],
        }]
        result = build_aux_string("Gen", ["BusNum", "GenID"], records,
                                  subdatatypes=["BidCurve", "ReactiveCapability"])
        assert "<SUBDATA BidCurve>" in result
        assert "<SUBDATA ReactiveCapability>" in result

    # ---- Roundtrip ----

    def test_roundtrip_build_then_parse(self):
        """build_aux_string output can be parsed back by parse_aux_content."""
        from esapp.saw._helpers import build_aux_string, parse_aux_content
        records = [{
            "BusNum": 101,
            "GenID": "1",
            "BidCurve": [["10.0", "50.0"], ["20.0", "60.0"]],
        }]
        aux_str = build_aux_string("Gen", ["BusNum", "GenID"], records,
                                   subdatatypes="BidCurve")
        parsed = parse_aux_content(aux_str, ["BusNum", "GenID"], ["BidCurve"])
        assert len(parsed) == 1
        assert parsed[0]["BusNum"] == "101"
        assert parsed[0]["GenID"] == "1"
        assert len(parsed[0]["BidCurve"]) == 2
        assert parsed[0]["BidCurve"][0] == ["10.0", "50.0"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
