"""
Online integration tests for GridWB components using IndexTool.
This script connects to a live PowerWorld session and verifies that
data can be retrieved for all defined GObject subclasses.

Usage:
    python test_online_components.py "C:\\Path\\To\\Case.pwb"
"""

import os
import sys
import pytest
import inspect
import pandas as pd
import tempfile

# Ensure gridwb can be imported if running from tests directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from gridwb.indexable import Indexable
    from gridwb import grid
    from gridwb.grid import GObject
    from gridwb.saw import PowerWorldError, COMError
except ImportError:
    print("Error: Could not import gridwb packages. Please ensure the package is in your Python path.")
    sys.exit(1)


@pytest.fixture(scope="module")
def io_instance():
    case_path = os.environ.get("SAW_TEST_CASE")
    if not case_path or not os.path.exists(case_path):
        pytest.skip("SAW_TEST_CASE environment variable not set or file not found.")

    print(f"\nConnecting to PowerWorld with case: {case_path}")
    io = Indexable(case_path)
    io.open()
    yield io
    print("\nClosing case...")
    if hasattr(io, 'esa'):
        io.esa.CloseCase()


def get_gobject_subclasses():
    """Helper to discover all GObject subclasses in the components module."""
    return [
        obj for _, obj in inspect.getmembers(grid, inspect.isclass)
        if issubclass(obj, GObject) and obj is not GObject
    ]


@pytest.mark.parametrize("component_class", get_gobject_subclasses())
class TestOnlineComponents:
    
    def test_read_keys(self, io_instance, component_class):
        """
        Test that IndexTool can read key fields for the component.
        This verifies that the object type string is correct and objects can be identified.
        """
        try:
            df = io_instance[component_class]
        except (PowerWorldError, COMError) as e:
            self._check_if_supported(io_instance, component_class, e)
        except Exception as e:
            pytest.fail(f"Failed to read keys for {component_class.__name__} ({component_class.TYPE}): {e}")
            
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            # If dataframe is not empty, check columns
            if not df.empty:
                for key in component_class.keys:
                    assert key in df.columns, f"Key field '{key}' missing in DataFrame for {component_class.__name__}"

    def test_read_all_fields(self, io_instance, component_class):
        """
        Test that IndexTool can read ALL defined fields for the component.
        This verifies that all field names defined in the GObject subclass are valid in the connected PowerWorld version.
        """
        try:
            # Request all fields using the slice syntax
            df = io_instance[component_class, :]
        except (PowerWorldError, COMError) as e:
            self._check_if_supported(io_instance, component_class, e)
        except Exception as e:
            pytest.fail(f"Failed to read all fields for {component_class.__name__} ({component_class.TYPE}): {e}")

        if df is not None:
            assert isinstance(df, pd.DataFrame)

    def _check_if_supported(self, io, component_class, original_error):
        """
        Second layer test: Try to save object fields to determine if the object type 
        is supported by the connected PowerWorld version.
        """
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
        
        is_supported = False
        try:
            fields = component_class.keys if component_class.keys else ["ALL"]
            io.esa.SaveObjectFields(tmp_path, component_class.TYPE, fields)
            is_supported = True
        except PowerWorldError:
            is_supported = False
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        if is_supported:
            err_msg = str(original_error)
            if hasattr(original_error, '__cause__') and original_error.__cause__:
                err_msg += " " + str(original_error.__cause__)

            if "cannot be retrieved through SimAuto" in err_msg:
                pytest.skip(f"Object type {component_class.TYPE} is supported by PowerWorld but not accessible via SimAuto.")
            if "memory resources" in err_msg:
                pytest.skip(f"Object type {component_class.TYPE} has too many fields to retrieve via SimAuto (Memory Error).")
            pytest.fail(f"Object type {component_class.TYPE} is supported (SaveObjectFields worked) but failed to read data: {original_error}")
        else:
            pytest.skip(f"Object type {component_class.TYPE} not recognized by PowerWorld version.")


if __name__ == "__main__":
    # Run pytest on this file
    sys.exit(pytest.main(["-v", __file__]))