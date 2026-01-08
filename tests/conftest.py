import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def saw_obj():
    """
    Provides a real SAW instance with a mocked low-level COM object.
    This allows testing of the SAW methods without a live PowerWorld connection.
    """
    with patch("win32com.client.dynamic.Dispatch") as mock_dispatch, patch(
        "win32com.client.gencache.EnsureDispatch", create=True
    ) as mock_ensure_dispatch, patch("tempfile.NamedTemporaryFile") as mock_tempfile, patch("os.unlink"):
        mock_pwcom = Mock()
        mock_dispatch.return_value = mock_pwcom
        mock_ensure_dispatch.return_value = mock_pwcom

        # Configure mock_tempfile to return an object with a string name
        # to satisfy Path(ntf.name) calls in SAW.__init__
        mock_ntf = Mock()
        mock_ntf.name = "dummy_temp.axd"
        mock_tempfile.return_value = mock_ntf

        # --- Set default "success" return values for common methods ---
        # This prevents `TypeError: 'Mock' object is not subscriptable` in tests
        # that call a method which returns a tuple (error_string, data).
        # A successful call with no data returns ('',).
        mock_pwcom.RunScriptCommand.return_value = ("",)
        mock_pwcom.ChangeParametersSingleElement.return_value = ("",)
        mock_pwcom.ProcessAuxFile.return_value = ("",)
        mock_pwcom.GetParametersMultipleElement.return_value = ("",)
        mock_pwcom.TSGetContingencyResults.return_value = ("",)
        mock_pwcom.OpenCase.return_value = ("",)
        mock_pwcom.SaveCase.return_value = ("",)
        mock_pwcom.CloseCase.return_value = ("",)
        mock_pwcom.GetCaseHeader.return_value = ("",)

        # --- Mock return values for calls made during SAW.__init__ ---
        mock_pwcom.GetParametersSingleElement.return_value = ("", ("23", "Jan 01 2023"))

        field_list_data = [
            ["*1*", "BusNum", "Integer", "Bus Number", "Bus Number"],
            ["*2*", "BusName", "String", "Bus Name", "Bus Name"],
        ]
        mock_pwcom.GetFieldList.return_value = ("", field_list_data)

        from esapp.saw import SAW

        # Limit object field lookup to speed up test setup
        saw_instance = SAW(FileName="dummy.pwb", object_field_lookup=("bus",))

        # Attach the mock for easy access in tests and reset it to clear __init__ calls
        saw_instance._pwcom = mock_pwcom
        # Reset the mock's call history, but *do not* reset its return_value or side_effect.
        # This preserves the default_return_value and any specific return_values set above
        # for methods called during SAW.__init__ or as general defaults.
        saw_instance._pwcom.reset_mock(return_value=False, side_effect=False)

        yield saw_instance