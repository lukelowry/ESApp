"""
Global fixtures for the ESA++ test suite.

This module provides reusable test fixtures for both offline (mocked) and online
(integration) testing of the ESA++ library. Fixtures are scoped appropriately to
balance test isolation with performance.
"""
import pytest
import os
from typing import Optional, Iterator, Callable, TYPE_CHECKING
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

if TYPE_CHECKING:
    from esapp.saw import SAW
    from esapp.workbench import GridWorkBench

try:
    from esapp.saw import SAW
    from esapp.workbench import GridWorkBench
except ImportError:
    # This allows tests to be collected even if esapp is not installed,
    # though online tests will be skipped.
    SAW = None  # type: ignore
    GridWorkBench = None  # type: ignore


def _get_test_case_path():
    """
    Get the test case path from configuration.
    
    Priority order:
    1. Environment variable SAW_TEST_CASE
    2. config_test.py file
    3. None (skip online tests)
    """
    # First check environment variable
    env_path = os.environ.get("SAW_TEST_CASE")
    if env_path:
        return env_path
    
    # Try to load from config_test.py
    try:
        import config_test
        if hasattr(config_test, 'SAW_TEST_CASE'):
            return config_test.SAW_TEST_CASE
    except ImportError:
        pass
    
    return None

@pytest.fixture(scope="session")
def saw_session():
    """
    Session-scoped fixture to manage a single PowerWorld Simulator instance
    for the entire test run.
    
    This fixture connects to PowerWorld once at the start of the test session
    and reuses the connection for all tests, improving performance. The connection
    is automatically closed at the end of the session.
    
    Configuration:
        Set case path in config_test.py or via SAW_TEST_CASE environment variable.
    
    Yields
    ------
    SAW
        An initialized SAW instance connected to the test case.
        
    Raises
    ------
    pytest.skip
        If esapp is not installed or test case is not configured.
    """
    if SAW is None:
        pytest.skip("esapp library not found.")

    case_path = _get_test_case_path()
    if not case_path:
        pytest.skip("SAW test case not configured. Set path in tests/config_test.py or SAW_TEST_CASE environment variable.")
    
    if not os.path.exists(case_path):
        pytest.skip(f"SAW test case file not found: {case_path}")

    print(f"\n[Session Setup] Connecting to PowerWorld with case: {case_path}")
    saw = None
    try:
        saw = SAW(case_path, early_bind=True)
        yield saw
    finally:
        print("\n[Session Teardown] Closing case and exiting PowerWorld...")
        if saw is not None:
            try:
                saw.exit()
            except Exception as e:
                print(f"Warning: Error during SAW cleanup: {e}")


@pytest.fixture(scope="function")
def saw_obj():
    """
    Provides a function-scoped, mocked SAW object for offline unit tests.
    
    This fixture patches the low-level COM dispatch calls to prevent any
    actual connection to PowerWorld, allowing tests to run without requiring
    PowerWorld Simulator to be installed or a valid case file.
    
    The mock is configured with default return values for common SAW operations,
    but can be customized within individual tests as needed.
    
    Yields
    ------
    SAW
        A SAW instance with mocked COM interface, suitable for testing without
        PowerWorld connectivity.
        
    Examples
    --------
    >>> def test_something(saw_obj):
    ...     # Customize mock behavior for this specific test
    ...     saw_obj._pwcom.RunScriptCommand.return_value = ("Success",)
    ...     result = saw_obj.RunScriptCommand("TestCommand;")
    ...     assert result is not None
    """
    with patch("win32com.client.dynamic.Dispatch") as mock_dispatch, \
         patch("win32com.client.gencache.EnsureDispatch", create=True) as mock_ensure_dispatch, \
         patch("tempfile.NamedTemporaryFile") as mock_tempfile, \
         patch("os.unlink"):
        
        mock_pwcom = MagicMock()
        mock_dispatch.return_value = mock_pwcom
        mock_ensure_dispatch.return_value = mock_pwcom

        # Mock the temp file used in SAW.__init__
        mock_ntf = Mock()
        mock_ntf.name = "dummy_temp.axd"
        mock_tempfile.return_value = mock_ntf

        # --- Mock return values for calls made during SAW.__init__ ---
        # And set default "success" return values for other common methods.
        # A successful call with no data should return ('',).
        mock_pwcom.RunScriptCommand.return_value = ("",)
        mock_pwcom.ChangeParametersSingleElement.return_value = ("",)
        mock_pwcom.ProcessAuxFile.return_value = ("",)
        mock_pwcom.SaveCase.return_value = ("",)
        mock_pwcom.CloseCase.return_value = ("",)
        mock_pwcom.GetCaseHeader.return_value = ("",)
        mock_pwcom.ChangeParametersMultipleElementRect.return_value = ("",)
        mock_pwcom.GetParametersMultipleElement.return_value = ("", [[1, 2], ["Bus1", "Bus2"]])

        mock_pwcom.OpenCase.return_value = ("",)  # Simulate successful case opening
        mock_pwcom.GetParametersSingleElement.return_value = ("", ("23", "Jan 01 2023"))
        field_list_data = [
            ["*1*", "BusNum", "Integer", "Bus Number", "Bus Number"],
            ["*2*", "BusName", "String", "Bus Name", "Bus Name"],
        ]
        mock_pwcom.GetFieldList.return_value = ("", field_list_data)

        # Limit object field lookup to speed up test setup
        saw_instance = SAW(FileName="dummy.pwb")

        # Attach the mock for easy access in tests and reset it to clear __init__ calls
        saw_instance._pwcom = mock_pwcom

        yield saw_instance


# -------------------------------------------------------------------------
# Additional Utility Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """
    Provides a temporary directory for test file operations.
    
    The directory is automatically cleaned up after each test.
    
    Parameters
    ----------
    tmp_path : Path
        Pytest's built-in temporary path fixture.
        
    Returns
    -------
    Path
        Path to a temporary directory unique to the test.
    """
    return tmp_path


@pytest.fixture
def sample_dataframe():
    """
    Provides a sample pandas DataFrame for testing data operations.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with sample bus data.
    """
    import pandas as pd
    return pd.DataFrame({
        "BusNum": [1, 2, 3],
        "BusName": ["Bus1", "Bus2", "Bus3"],
        "BusPUVolt": [1.0, 0.98, 1.02],
        "BusAngle": [0.0, -2.5, 1.8]
    })


@pytest.fixture
def mock_power_flow_results(saw_obj):
    """
    Configures the mock SAW object to return realistic power flow results.
    
    This fixture sets up mock return values for common power flow queries,
    useful for testing workflows that depend on power flow results.
    
    Parameters
    ----------
    saw_obj : SAW
        The mocked SAW fixture.
        
    Returns
    -------
    SAW
        The configured SAW object with power flow mock data.
    """
    import pandas as pd
    
    bus_data = pd.DataFrame({
        "BusNum": [1, 2, 3, 4, 5],
        "BusName": ["Bus1", "Bus2", "Bus3", "Bus4", "Bus5"],
        "BusPUVolt": [1.05, 1.02, 0.98, 1.01, 0.99],
        "BusAngle": [0.0, -2.1, -5.3, -3.2, -4.5],
        "BusNetMW": [100.0, -50.0, -30.0, -20.0, 0.0],
        "BusNetMVR": [50.0, -20.0, -15.0, -10.0, -5.0]
    })
    
    def get_params_side_effect(obj_type, fields, *args, **kwargs):
        if obj_type.lower() == "bus":
            return bus_data[fields]
        return pd.DataFrame()
    
    saw_obj._pwcom.GetParametersMultipleElement.side_effect = None
    saw_obj.GetParametersMultipleElement = Mock(side_effect=get_params_side_effect)
    
    return saw_obj


@pytest.fixture
def reset_mock_calls(saw_obj):
    """
    Fixture that resets mock call counts before each test.
    
    This ensures that tests don't interfere with each other's assertions
    about mock call counts. Only used in unit tests with saw_obj.
    
    Parameters
    ----------
    saw_obj : SAW
        The mocked SAW fixture.
        
    Note
    ----
    This is NOT autouse - tests must explicitly request it if needed.
    """
    yield
    if hasattr(saw_obj, '_pwcom'):
        saw_obj._pwcom.reset_mock()


@pytest.fixture
def temp_file():
    """
    Provides a factory for creating temporary files that are automatically cleaned up.
    
    This fixture creates temporary files with specified suffixes and ensures they are
    removed after the test completes, even if the test fails.
    
    Returns
    -------
    callable
        A function that takes a suffix (e.g., '.pwb', '.csv') and returns a temp file path.
        
    Examples
    --------
    >>> def test_save(temp_file):
    ...     tmp_pwb = temp_file('.pwb')
    ...     save_case(tmp_pwb)
    ...     assert os.path.exists(tmp_pwb)
    """
    import tempfile
    import os
    files = []

    def _create(suffix):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.close()
        files.append(tf.name)
        return tf.name

    yield _create
    
    # Cleanup all created temp files
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


# -------------------------------------------------------------------------
# Pytest Configuration Hooks
# -------------------------------------------------------------------------

def pytest_configure(config):
    """Add custom markers for test organization."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests requiring PowerWorld")
    config.addinivalue_line("markers", "unit: marks tests as unit tests with mocked dependencies")
    config.addinivalue_line("markers", "requires_case: marks tests that require a valid PowerWorld case file")


# -------------------------------------------------------------------------
# Assertion Helpers
# -------------------------------------------------------------------------

def assert_dataframe_valid(df, expected_columns=None, min_rows=1, name="DataFrame"):
    """
    Assert that a DataFrame is valid and has expected structure.
    
    Parameters
    ----------
    df : pd.DataFrame or None
        The DataFrame to validate.
    expected_columns : list, optional
        List of column names that must be present.
    min_rows : int, optional
        Minimum number of rows expected. Default is 1.
    name : str, optional
        Name of the DataFrame for error messages.
        
    Raises
    ------
    AssertionError
        If validation fails.
    """
    import pandas as pd
    
    assert df is not None, f"{name} is None"
    assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"
    assert len(df) >= min_rows, f"{name} has {len(df)} rows, expected at least {min_rows}"
    
    if expected_columns:
        for col in expected_columns:
            assert col in df.columns, f"{name} missing expected column: {col}"


def assert_voltage_reasonable(voltage, min_pu=0.5, max_pu=1.5):
    """
    Assert that a voltage value is within reasonable bounds.
    
    Parameters
    ----------
    voltage : float or array-like
        Voltage value(s) in per-unit.
    min_pu : float
        Minimum acceptable voltage (default 0.5 pu).
    max_pu : float
        Maximum acceptable voltage (default 1.5 pu).
    """
    import numpy as np
    voltage_arr = np.atleast_1d(voltage)
    assert np.all(voltage_arr >= min_pu), f"Voltage below {min_pu} pu: {voltage_arr.min()}"
    assert np.all(voltage_arr <= max_pu), f"Voltage above {max_pu} pu: {voltage_arr.max()}"


def assert_matrix_valid(matrix, expected_shape=None, is_sparse=True):
    """
    Assert that a matrix is valid.
    
    Parameters
    ----------
    matrix : sparse matrix or ndarray
        The matrix to validate.
    expected_shape : tuple, optional
        Expected (rows, cols) shape.
    is_sparse : bool, optional
        Whether matrix should be sparse.
    """
    assert matrix is not None, "Matrix is None"
    
    if is_sparse:
        assert hasattr(matrix, "toarray"), "Matrix is not sparse"
    
    if expected_shape:
        assert matrix.shape == expected_shape, f"Matrix shape {matrix.shape} != expected {expected_shape}"


# -------------------------------------------------------------------------
# Shared Test Utilities
# -------------------------------------------------------------------------

def get_all_gobject_subclasses():
    """
    Recursively finds all non-abstract, testable GObject subclasses.
    
    This utility is used by parametrized tests to discover all component types
    defined in the grid module.
    
    Returns
    -------
    list[Type[GObject]]
        List of all GObject subclass types that have a _TYPE attribute.
    """
    try:
        from esapp import grid
    except ImportError:
        return []
    
    all_subclasses = []
    q = list(grid.GObject.__subclasses__())
    visited = set(q)
    while q:
        cls = q.pop(0)
        # A concrete, testable GObject subclass must have a _TYPE attribute
        if hasattr(cls, '_TYPE'):
            all_subclasses.append(cls)

        for subclass in cls.__subclasses__():
            if subclass not in visited:
                visited.add(subclass)
                q.append(subclass)
    return all_subclasses


def get_sample_gobject_subclasses():
    """
    Returns a representative sample of GObject subclasses for testing.
    
    This reduces test execution time by testing a diverse sample instead of
    all component types. Use this for tests where behavior is identical across
    all components (e.g., mocked unit tests).
    
    Returns
    -------
    list[Type[GObject]]
        Sample of GObject subclasses representing different categories.
    """
    try:
        from esapp import grid
        all_classes = get_all_gobject_subclasses()
        
        if not all_classes:
            # Return empty list if no classes found - parametrize will skip tests
            import warnings
            warnings.warn("No GObject subclasses found. Tests will be skipped.")
            return []
        
        # Prioritize commonly used components and diverse categories
        priority_types = ['Bus', 'Gen', 'Load', 'Branch', 'Shunt', 'Area', 'Zone', 
                         'Contingency', 'Interface', 'InjectionGroup']
        
        sample = []
        for type_name in priority_types:
            for cls in all_classes:
                if hasattr(cls, 'TYPE') and cls.TYPE == type_name:
                    sample.append(cls)
                    break
        
        # If we don't have enough, add more randomly (with seed for reproducibility)
        import random
        random.seed(42)  # Deterministic sampling for consistent test discovery
        remaining = [c for c in all_classes if c not in sample]
        if remaining and len(sample) < 15:
            sample.extend(random.sample(remaining, min(5, len(remaining))))
        
        return sample
    except ImportError as e:
        import warnings
        warnings.warn(f"Failed to import esapp.grid: {e}")
        return []
    except Exception as e:
        import warnings
        warnings.warn(f"Error getting GObject subclasses: {e}")
        return []


# -------------------------------------------------------------------------
# Additional Mocked SAW Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def saw_with_error_responses():
    """
    Provides a SAW object configured to return errors for testing error handling.
    
    Use this fixture to test error paths and exception handling in code
    that calls SAW methods.
    
    Yields
    ------
    SAW
        A SAW instance configured to return error responses.
    """
    with patch("win32com.client.dynamic.Dispatch") as mock_dispatch, \
         patch("win32com.client.gencache.EnsureDispatch", create=True), \
         patch("tempfile.NamedTemporaryFile") as mock_tempfile, \
         patch("os.unlink"):
        
        mock_pwcom = MagicMock()
        mock_dispatch.return_value = mock_pwcom
        
        mock_ntf = MagicMock()
        mock_ntf.name = "dummy_temp.axd"
        mock_tempfile.return_value = mock_ntf
        
        # Setup minimal init requirements
        mock_pwcom.OpenCase.return_value = ("",)
        mock_pwcom.GetParametersSingleElement.return_value = ("", ("23", "Jan 01 2023"))
        mock_pwcom.GetFieldList.return_value = ("", [])
        
        # Setup error responses for testing
        mock_pwcom.RunScriptCommand.return_value = ("Error: Test error message",)
        mock_pwcom.GetParametersMultipleElement.return_value = (
            "Error: Object not found", None
        )
        mock_pwcom.ChangeParametersSingleElement.return_value = (
            "Error: Could not modify object",
        )
        
        saw_instance = SAW(FileName="dummy.pwb")
        saw_instance._pwcom = mock_pwcom
        
        yield saw_instance


@pytest.fixture
def saw_empty_case():
    """
    Provides a SAW object configured to return empty data sets.
    
    Use this fixture to test handling of empty results (no buses, no generators, etc.)
    
    Yields
    ------
    SAW
        A SAW instance configured to return empty data.
    """
    with patch("win32com.client.dynamic.Dispatch") as mock_dispatch, \
         patch("win32com.client.gencache.EnsureDispatch", create=True), \
         patch("tempfile.NamedTemporaryFile") as mock_tempfile, \
         patch("os.unlink"):
        
        mock_pwcom = MagicMock()
        mock_dispatch.return_value = mock_pwcom
        
        mock_ntf = MagicMock()
        mock_ntf.name = "dummy_temp.axd"
        mock_tempfile.return_value = mock_ntf
        
        mock_pwcom.OpenCase.return_value = ("",)
        mock_pwcom.GetParametersSingleElement.return_value = ("", ("23", "Jan 01 2023"))
        mock_pwcom.GetFieldList.return_value = ("", [])
        
        # Return empty data
        mock_pwcom.RunScriptCommand.return_value = ("",)
        mock_pwcom.GetParametersMultipleElement.return_value = ("", None)
        
        saw_instance = SAW(FileName="dummy.pwb")
        saw_instance._pwcom = mock_pwcom
        
        yield saw_instance


@pytest.fixture
def workbench_mocked(saw_obj, sample_dataframe):
    """
    Provides a mocked GridWorkBench for testing workbench functionality.
    
    This fixture creates a workbench with a mocked SAW backend,
    useful for testing workbench operations without PowerWorld.
    
    Parameters
    ----------
    saw_obj : SAW
        The mocked SAW fixture.
    sample_dataframe : pd.DataFrame
        Sample data for bus results.
        
    Yields
    ------
    GridWorkBench or Mock
        A mocked workbench object, or Mock if GridWorkBench is unavailable.
    """
    if GridWorkBench is None:
        # Return a mock if workbench not available
        mock_wb = MagicMock()
        mock_wb.saw = saw_obj
        yield mock_wb
    else:
        with patch.object(GridWorkBench, '__init__', lambda self, *args, **kwargs: None):
            wb = GridWorkBench.__new__(GridWorkBench)
            wb.saw = saw_obj
            wb._case_path = "dummy.pwb"
            yield wb


def pytest_collection_modifyitems(config, items):
    """
    Auto-mark tests based on their location and naming.
    
    This automatically applies markers to tests based on their module,
    reducing boilerplate marker declarations.
    """
    for item in items:
        # Mark integration tests (files starting with test_integration_)
        if "test_integration_" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.requires_case)
        # Mark unit tests (all other test files)
        elif "test_" in item.nodeid and "test_integration_" not in item.nodeid:
            item.add_marker(pytest.mark.unit)