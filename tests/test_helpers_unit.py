"""
Unit tests for pure Python helper functions in the SAW module.

These tests do NOT require PowerWorld - they test pure Python functions
that handle data transformation, formatting, and file parsing.

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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
