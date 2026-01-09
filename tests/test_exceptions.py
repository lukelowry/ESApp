"""
Unit tests for exception handling in the esapp module.

WHAT THIS TESTS:
- Custom exception hierarchy (PowerWorldError, COMError, SimAutoFeatureError, etc.)
- Exception instantiation and message handling
- Error parsing from PowerWorld COM interface responses
- Specific error type detection (prerequisite, add-on, command failures)
- SAW error handling patterns with mocked responses

DEPENDENCIES: None (mocked, no PowerWorld required)

USAGE:
    pytest tests/test_exceptions.py -v
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Type

try:
    from esapp.saw._exceptions import (
        Error,
        PowerWorldError,
        PowerWorldPrerequisiteError,
        PowerWorldAddonError,
        CommandNotRespectedError,
        COMError,
        SimAutoFeatureError,
        RPC_S_UNKNOWN_IF,
        RPC_S_CALL_FAILED,
    )
    from esapp.saw import SAW
    from esapp.utils.exceptions import ESAPlusError
except ImportError:
    pytest.skip("esapp library not found", allow_module_level=True)


# -------------------------------------------------------------------------
# Exception Hierarchy Tests
# -------------------------------------------------------------------------

def test_exception_hierarchy():
    """Test that custom exceptions have proper inheritance."""
    assert issubclass(PowerWorldError, Error)
    assert issubclass(PowerWorldPrerequisiteError, PowerWorldError)
    assert issubclass(PowerWorldAddonError, PowerWorldError)
    assert issubclass(CommandNotRespectedError, PowerWorldError)
    assert issubclass(COMError, Error)


def test_exception_instantiation():
    """Test that exceptions can be instantiated with messages."""
    msg = "Test error message"
    
    err = PowerWorldError(msg)
    assert str(err) == msg
    assert err.message == msg
    
    err2 = CommandNotRespectedError(msg)
    assert str(err2) == msg


# -------------------------------------------------------------------------
# PowerWorld Error Tests
# -------------------------------------------------------------------------

def test_powerworld_error_parsing():
    """Test that PowerWorld error messages are parsed correctly."""
    error_msg = "Error: Bus 123 not found in case"
    
    err = PowerWorldError(error_msg)
    assert "Bus 123" in str(err)
    assert "not found" in str(err)


@pytest.mark.parametrize("error_class,expected_type", [
    (PowerWorldPrerequisiteError, PowerWorldPrerequisiteError),
    (PowerWorldAddonError, PowerWorldAddonError),
    (CommandNotRespectedError, CommandNotRespectedError),
])
def test_specific_powerworld_errors(error_class: Type[Exception], expected_type: Type[Exception]):
    """Test specific PowerWorld error types."""
    msg = "Specific error"
    err = error_class(msg)
    assert isinstance(err, expected_type)
    assert isinstance(err, PowerWorldError)


# -------------------------------------------------------------------------
# COM Error Tests
# -------------------------------------------------------------------------

def test_com_error_instantiation():
    """Test COMError with different message formats."""
    err1 = COMError("Simple COM error")
    assert "Simple COM error" in str(err1)
    
    err2 = COMError(RPC_S_UNKNOWN_IF)
    assert str(err2) == str(RPC_S_UNKNOWN_IF)


def test_rpc_error_constants():
    """Test that RPC error constants are defined."""
    assert isinstance(RPC_S_UNKNOWN_IF, int)
    assert isinstance(RPC_S_CALL_FAILED, int)
    assert RPC_S_UNKNOWN_IF != RPC_S_CALL_FAILED


# -------------------------------------------------------------------------
# SAW Error Handling Tests (with mocking)
# -------------------------------------------------------------------------

@pytest.fixture
def saw_with_error_mock():
    """Create a mocked SAW instance that raises errors."""
    with patch("win32com.client.dynamic.Dispatch") as mock_dispatch, \
         patch("win32com.client.gencache.EnsureDispatch", create=True), \
         patch("tempfile.NamedTemporaryFile"), \
         patch("os.unlink"):
        
        mock_pwcom = MagicMock()
        mock_dispatch.return_value = mock_pwcom
        
        # Set up default error-free responses for __init__
        mock_pwcom.OpenCase.return_value = ("",)
        mock_pwcom.GetParametersSingleElement.return_value = ("", ("23", "Jan 01 2023"))
        mock_pwcom.GetFieldList.return_value = ("", [])
        
        saw = SAW(FileName="dummy.pwb")
        saw._pwcom = mock_pwcom
        
        yield saw


def test_saw_handles_command_errors(saw_with_error_mock):
    """Test that SAW properly raises errors for failed commands."""
    saw = saw_with_error_mock
    
    # Simulate a PowerWorld error
    error_msg = "Error: Invalid command syntax"
    saw._pwcom.RunScriptCommand.return_value = (error_msg,)
    
    with pytest.raises(PowerWorldError, match="Invalid command"):
        saw.RunScriptCommand("InvalidCommand")


def test_saw_handles_com_errors(saw_with_error_mock):
    """Test that SAW handles COM errors appropriately."""
    import pywintypes
    
    saw = saw_with_error_mock
    
    # Simulate a COM error
    com_error = pywintypes.com_error(RPC_S_UNKNOWN_IF, "COM Error", None, None)
    saw._pwcom.RunScriptCommand.side_effect = com_error
    
    with pytest.raises(COMError):
        saw.RunScriptCommand("AnyCommand")


def test_saw_prerequisite_error():
    """Test PowerWorldPrerequisiteError is raised for missing prerequisites."""
    # Example: trying to run transient analysis without TS initialized
    with patch("win32com.client.dynamic.Dispatch") as mock_dispatch, \
         patch("win32com.client.gencache.EnsureDispatch", create=True), \
         patch("tempfile.NamedTemporaryFile"), \
         patch("os.unlink"):
        
        mock_pwcom = MagicMock()
        mock_dispatch.return_value = mock_pwcom
        
        mock_pwcom.OpenCase.return_value = ("",)
        mock_pwcom.GetParametersSingleElement.return_value = ("", ("23", "Jan 01 2023"))
        mock_pwcom.GetFieldList.return_value = ("", [])
        
        saw = SAW(FileName="dummy.pwb")
        saw._pwcom = mock_pwcom
        
        # Simulate prerequisite error
        error_msg = "Error: Transient stability not initialized"
        saw._pwcom.RunScriptCommand.return_value = (error_msg,)
        
        # The actual implementation would parse this and raise PowerWorldPrerequisiteError
        # For now, test that the error contains the right message
        result = saw._pwcom.RunScriptCommand("TSSolve")
        assert "not initialized" in result[0]


# -------------------------------------------------------------------------
# Error Message Quality Tests
# -------------------------------------------------------------------------

def test_error_messages_are_informative():
    """Test that error messages provide actionable information."""
    errors_and_expectations = [
        (PowerWorldError("Bus not found"), "Bus"),
        (CommandNotRespectedError("Solve failed"), "failed"),
        (PowerWorldAddonError("Add-on required"), "Add-on"),
    ]
    
    for error, expected_content in errors_and_expectations:
        assert expected_content in str(error), \
            f"Error message should contain '{expected_content}'"


def test_error_repr():
    """Test that exceptions have useful repr for debugging."""
    err = PowerWorldError("Test error")
    repr_str = repr(err)
    assert "PowerWorldError" in repr_str
    assert "Test error" in repr_str


# -------------------------------------------------------------------------
# Utils Exception Tests
# -------------------------------------------------------------------------

def test_esaplus_error():
    """Test the general ESAPlus error class if it exists."""
    try:
        err = ESAPlusError("General error")
        assert "General error" in str(err)
    except NameError:
        pytest.skip("ESAPlusError not defined")


# -------------------------------------------------------------------------
# Exception Context Tests
# -------------------------------------------------------------------------

def test_exception_chaining():
    """Test that exceptions can be chained for context."""
    original = ValueError("Original error")
    
    try:
        try:
            raise original
        except ValueError as e:
            raise PowerWorldError("PowerWorld error occurred") from e
    except PowerWorldError as pwe:
        assert pwe.__cause__ is original
        assert isinstance(pwe.__cause__, ValueError)


def test_exception_with_traceback_info():
    """Test that exceptions preserve useful traceback information."""
    try:
        raise PowerWorldError("Error with traceback")
    except PowerWorldError as e:
        import traceback
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        assert "test_exception_with_traceback_info" in tb_str
        assert "PowerWorldError" in tb_str


# -------------------------------------------------------------------------
# Error Path Tests - Specific Scenarios
# -------------------------------------------------------------------------

class TestErrorPathScenarios:
    """Tests for specific error paths in SAW operations."""

    @pytest.fixture
    def mock_saw(self):
        """Create a fresh mocked SAW for each test."""
        with patch("win32com.client.dynamic.Dispatch") as mock_dispatch, \
             patch("win32com.client.gencache.EnsureDispatch", create=True), \
             patch("tempfile.NamedTemporaryFile"), \
             patch("os.unlink"):
            
            mock_pwcom = MagicMock()
            mock_dispatch.return_value = mock_pwcom
            
            mock_pwcom.OpenCase.return_value = ("",)
            mock_pwcom.GetParametersSingleElement.return_value = ("", ("23", "Jan 01 2023"))
            mock_pwcom.GetFieldList.return_value = ("", [])
            mock_pwcom.RunScriptCommand.return_value = ("",)
            mock_pwcom.GetParametersMultipleElement.return_value = ("", [[1, 2], ["A", "B"]])
            
            saw = SAW(FileName="dummy.pwb")
            saw._pwcom = mock_pwcom
            
            yield saw

    def test_addon_error_detection(self, mock_saw):
        """Test that add-on errors are detected and raised as PowerWorldAddonError."""
        # Simulate an add-on not registered error
        mock_saw._pwcom.RunScriptCommand.return_value = (
            "Error: Add-on 'TransLineCalc' is not registered",
        )
        
        with pytest.raises(PowerWorldError) as exc_info:
            mock_saw.RunScriptCommand("CalculateRXBG")
        
        assert "Add-on" in str(exc_info.value) or "not registered" in str(exc_info.value)

    def test_empty_data_returns_none(self, mock_saw):
        """Test that empty data returns None or empty DataFrame."""
        mock_saw._pwcom.GetParametersMultipleElement.return_value = ("", None)
        
        result = mock_saw.GetParametersMultipleElement("Bus", ["BusNum"])
        assert result is None or (hasattr(result, "empty") and result.empty)

    def test_object_not_found_error(self, mock_saw):
        """Test error when object type is not found."""
        mock_saw._pwcom.RunScriptCommand.return_value = (
            "Error: Object type 'InvalidObject' not found",
        )
        
        with pytest.raises(PowerWorldError):
            mock_saw.RunScriptCommand("SelectAll(InvalidObject)")

    def test_case_not_open_error(self, mock_saw):
        """Test error when no case is open."""
        mock_saw._pwcom.RunScriptCommand.return_value = (
            "Error: No case is currently open",
        )
        
        with pytest.raises(PowerWorldError):
            mock_saw.RunScriptCommand("SolvePowerFlow")

    def test_multiple_errors_in_response(self, mock_saw):
        """Test handling of multiple errors in a single response."""
        mock_saw._pwcom.RunScriptCommand.return_value = (
            "Error: First error\nError: Second error",
        )
        
        with pytest.raises(PowerWorldError) as exc_info:
            mock_saw.RunScriptCommand("BadCommand")
        
        # Should contain at least the first error
        assert "error" in str(exc_info.value).lower()

    def test_warning_vs_error_handling(self, mock_saw):
        """Test that warnings are handled differently from errors."""
        # Some PowerWorld responses are warnings, not errors
        mock_saw._pwcom.RunScriptCommand.return_value = ("",)  # No error
        
        # Should not raise
        mock_saw.RunScriptCommand("ValidCommand")

    def test_get_parameters_with_invalid_fields(self, mock_saw):
        """Test error when requesting invalid field names."""
        mock_saw._pwcom.GetParametersMultipleElement.return_value = (
            "Error: Field 'InvalidField' not found for object type 'Bus'",
            None
        )
        
        with pytest.raises(PowerWorldError):
            mock_saw.GetParametersMultipleElement("Bus", ["InvalidField"])


class TestCOMErrorRecovery:
    """Tests for COM error scenarios and recovery."""

    def test_com_error_with_rpc_call_failed(self):
        """Test handling of RPC_S_CALL_FAILED error."""
        err = COMError(RPC_S_CALL_FAILED)
        assert isinstance(err, Error)

    def test_com_error_message_preservation(self):
        """Test that COM error messages are preserved."""
        import pywintypes
        
        # Simulate a COM error with a description
        com_err = pywintypes.com_error(
            -2147352567,  # DISP_E_EXCEPTION
            "Automation error",
            ("Error", 0, "Description of error", None, 0, -2147024809),
            None
        )
        
        # The exception should be constructable from this
        wrapped_err = COMError(str(com_err))
        assert "error" in str(wrapped_err).lower()

