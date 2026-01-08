"""
Global fixtures for the ESA++ test suite.
"""
import pytest, os
from unittest.mock import Mock, patch, MagicMock

try:
    from esapp.saw import SAW
    from esapp.workbench import GridWorkBench
except ImportError:
    # This allows tests to be collected even if esapp is not installed,
    # though online tests will be skipped.
    SAW = None
    GridWorkBench = None

@pytest.fixture(scope="session")
def saw_session():
    """
    Session-scoped fixture to manage a single PowerWorld Simulator instance
    for the entire test run.
    """
    if SAW is None:
        pytest.skip("esapp library not found.")

    case_path = os.environ.get("SAW_TEST_CASE")
    if not case_path or not os.path.exists(case_path):
        pytest.skip("SAW_TEST_CASE environment variable not set or file not found.")

    print(f"\n[Session Setup] Connecting to PowerWorld with case: {case_path}")
    saw = SAW(case_path, early_bind=True)
    yield saw
    print("\n[Session Teardown] Closing case and exiting PowerWorld...")
    saw.exit()


@pytest.fixture(scope="function")
def saw_obj():
    """
    Provides a function-scoped, mocked SAW object for offline unit tests.
    This fixture patches the low-level COM dispatch calls to prevent any
    actual connection to PowerWorld.
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