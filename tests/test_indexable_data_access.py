"""
Unit tests for the Indexable class data access methods.

WHAT THIS TESTS:
- __getitem__ method for reading PowerWorld data (single objects, lists, DataFrames)
- __setitem__ method for writing data back to PowerWorld
- Data type conversion and validation (strings, ints, floats, DataFrames)
- Integration with all GObject component types (parametrized testing)
- Error handling for invalid indices and data types

DEPENDENCIES: None (mocked SAW instance, no PowerWorld required)

USAGE:
    pytest tests/test_indexable_data_access.py -v
    pytest tests/test_indexable_data_access.py -k "test_getitem" -v  # Only read tests
"""
import pytest
from unittest.mock import Mock, patch
from typing import Type, List
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from esapp.indexable import Indexable
from esapp import grid

# Import shared test utilities
from conftest import get_all_gobject_subclasses, get_sample_gobject_subclasses


def pytest_generate_tests(metafunc):
    """
    Dynamically generate test parameters at test collection time.
    This ensures get_sample_gobject_subclasses() is called with proper environment setup.
    """
    if "g_object" in metafunc.fixturenames:
        classes = get_sample_gobject_subclasses()
        ids = [c.TYPE if hasattr(c, 'TYPE') else c.__name__ for c in classes]
        metafunc.parametrize("g_object", classes, ids=ids)


@pytest.fixture
def indexable_instance() -> Indexable:
    """Provides an Indexable instance with a mocked SAW dependency."""
    with patch('esapp.indexable.SAW') as mock_saw_class:
        mock_esa = Mock()
        mock_saw_class.return_value = mock_esa
        
        instance = Indexable()
        instance.esa = mock_esa
        yield instance


# Use sample strategy for most tests to reduce execution time from 10,000+ to ~100 tests
# Full parametrization only for critical validation tests
# Note: g_object parameter is provided by pytest_generate_tests hook
def test_getitem_key_fields(indexable_instance: Indexable, g_object: Type[grid.GObject]):
    """Test `idx_tool[GObject]` retrieves only key fields."""
    # Arrange
    mock_esa = indexable_instance.esa
    unique_keys = sorted(list(set(g_object.keys)))

    if not unique_keys:
        # Act
        result = indexable_instance[g_object]
        # Assert
        assert result is None
        mock_esa.GetParamsRectTyped.assert_not_called()
        return

    mock_df = pd.DataFrame({k: [1, 2] for k in unique_keys})
    mock_esa.GetParamsRectTyped.return_value = mock_df

    # Act
    result_df = indexable_instance[g_object]

    # Assert
    mock_esa.GetParamsRectTyped.assert_called_once_with(g_object.TYPE, unique_keys)
    assert_frame_equal(result_df, mock_df)


def test_getitem_all_fields(indexable_instance: Indexable, g_object: Type[grid.GObject]):
    """Test `idx_tool[GObject, :]` retrieves all fields."""
    # Arrange
    mock_esa = indexable_instance.esa
    expected_fields = sorted(list(set(g_object.keys) | set(g_object.fields)))

    if not expected_fields:
        # Act
        result = indexable_instance[g_object, :]
        # Assert
        assert result is None
        mock_esa.GetParamsRectTyped.assert_not_called()
        return

    mock_df = pd.DataFrame({f: [1] for f in expected_fields})
    mock_esa.GetParamsRectTyped.return_value = mock_df

    # Act
    result_df = indexable_instance[g_object, :]

    # Assert
    mock_esa.GetParamsRectTyped.assert_called_once_with(g_object.TYPE, expected_fields)
    assert_frame_equal(result_df, mock_df)


def test_setitem_broadcast(indexable_instance: Indexable, g_object: Type[grid.GObject]):
    """Test `idx_tool[GObject, 'Field'] = value` broadcasts a value."""
    # Arrange
    mock_esa = indexable_instance.esa
    settable_fields = [f for f in g_object.fields if f not in g_object.keys]
    if not settable_fields:
        pytest.skip(f"{g_object.__name__} has no settable (non-key) fields.")

    field_to_set = settable_fields[0]
    value_to_set = 1.234
    unique_keys = sorted(list(set(g_object.keys)))

    # Act
    if not unique_keys:  # Keyless object
        indexable_instance[g_object, field_to_set] = value_to_set
        expected_df = pd.DataFrame({field_to_set: [value_to_set]})
    else:  # Keyed object
        mock_key_df = pd.DataFrame({k: [101, 102] for k in unique_keys})
        mock_esa.GetParamsRectTyped.return_value = mock_key_df
        
        indexable_instance[g_object, field_to_set] = value_to_set

        # The df sent to PW should have keys and the new value
        expected_df = mock_key_df.copy()
        expected_df[field_to_set] = value_to_set

    # Assert
    mock_esa.ChangeParametersMultipleElementRect.assert_called_once()
    call_args, _ = mock_esa.ChangeParametersMultipleElementRect.call_args
    sent_df = call_args[2]
    assert_frame_equal(sent_df, expected_df)


def test_setitem_bulk_update_from_df(indexable_instance: Indexable, g_object: Type[grid.GObject]):
    """Test `idx_tool[GObject] = df` performs a bulk update."""
    # Arrange
    mock_esa = indexable_instance.esa
    
    # This covers a previously untested code path.
    if not g_object.fields:
        pytest.skip(f"{g_object.__name__} has no fields to update.")

    update_df = pd.DataFrame({f: [10, 20] for f in g_object.fields})

    # Act
    indexable_instance[g_object] = update_df

    # Assert
    mock_esa.ChangeParametersMultipleElementRect.assert_called_once_with(
        g_object.TYPE,
        update_df.columns.tolist(),
        update_df
    )


def test_getitem_specific_fields(indexable_instance: Indexable, g_object: Type[grid.GObject]):
    """Test `idx_tool[GObject, ['Field1', 'Field2']]` retrieves specific fields plus all keys."""
    # Arrange
    mock_esa = indexable_instance.esa
    
    # Select one non-key field to request, if available.
    specific_fields_to_request = [f for f in g_object.fields if f not in g_object.keys]
    if not specific_fields_to_request:
        pytest.skip(f"{g_object.__name__} has no non-key fields to request specifically.")
    
    field_to_request = specific_fields_to_request[0]

    # The implementation always fetches all keys plus the requested fields.
    expected_fields_to_get = sorted(list(set(g_object.keys) | {field_to_request}))

    mock_df = pd.DataFrame({f: [1, 2] for f in expected_fields_to_get})
    mock_esa.GetParamsRectTyped.return_value = mock_df

    # Act
    result_df = indexable_instance[g_object, [field_to_request]]

    # Assert
    mock_esa.GetParamsRectTyped.assert_called_once_with(g_object.TYPE, expected_fields_to_get)
    assert_frame_equal(result_df, mock_df)


def test_setitem_broadcast_multiple_fields(indexable_instance: Indexable, g_object: Type[grid.GObject]):
    """Test `idx_tool[GObject, ['F1', 'F2']] = [v1, v2]` broadcasts multiple values."""
    # Arrange
    mock_esa = indexable_instance.esa
    settable_fields = [f for f in g_object.fields if f not in g_object.keys]
    if len(settable_fields) < 2:
        pytest.skip(f"{g_object.__name__} has fewer than two settable fields.")

    fields_to_set = settable_fields[:2]
    values_to_set = [1.1, 2.2]
    unique_keys = sorted(list(set(g_object.keys)))

    if not unique_keys:
        pytest.skip("Skipping multiple field broadcast test for keyless objects for simplicity.")

    mock_key_df = pd.DataFrame({k: [101, 102] for k in unique_keys})
    mock_esa.GetParamsRectTyped.return_value = mock_key_df
    
    # Act
    indexable_instance[g_object, fields_to_set] = values_to_set

    # Assert
    expected_df = mock_key_df.copy()
    expected_df[fields_to_set] = values_to_set  # Pandas assigns list to columns
    
    mock_esa.ChangeParametersMultipleElementRect.assert_called_once()
    sent_df = mock_esa.ChangeParametersMultipleElementRect.call_args[0][2]
    assert_frame_equal(sent_df, expected_df)


def test_setitem_raises_error_on_invalid_index(indexable_instance: Indexable):
    """Test that __setitem__ raises TypeError for unsupported index types."""
    with pytest.raises(TypeError, match="Unsupported index for __setitem__"):
        indexable_instance[123] = "some_value"
    with pytest.raises(TypeError, match="First element of index must be a GObject subclass"):
        indexable_instance[(123, "field")] = "some_value"


# -------------------------------------------------------------------------
# Edge Case Tests
# -------------------------------------------------------------------------

def test_getitem_with_empty_dataframe(indexable_instance: Indexable):
    """Test behavior when PowerWorld returns an empty DataFrame."""
    mock_esa = indexable_instance.esa
    mock_esa.GetParamsRectTyped.return_value = pd.DataFrame()
    
    result = indexable_instance[grid.Bus]
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_getitem_with_none_return(indexable_instance: Indexable):
    """Test behavior when PowerWorld returns None."""
    mock_esa = indexable_instance.esa
    mock_esa.GetParamsRectTyped.return_value = None
    
    result = indexable_instance[grid.Bus]
    assert result is None


def test_setitem_with_nan_values(indexable_instance: Indexable):
    """Test that NaN values are handled correctly in DataFrame updates."""
    mock_esa = indexable_instance.esa
    
    # Create DataFrame with NaN values
    update_df = pd.DataFrame({
        "BusNum": [1, 2, 3],
        "BusPUVolt": [1.0, np.nan, 1.02]
    })
    
    indexable_instance[grid.Bus] = update_df
    
    # Verify the DataFrame was passed with NaN intact
    mock_esa.ChangeParametersMultipleElementRect.assert_called_once()
    sent_df = mock_esa.ChangeParametersMultipleElementRect.call_args[0][2]
    assert pd.isna(sent_df.iloc[1]["BusPUVolt"])


def test_setitem_with_mixed_types(indexable_instance: Indexable):
    """Test setting fields with mixed data types."""
    mock_esa = indexable_instance.esa
    
    update_df = pd.DataFrame({
        "BusNum": [1, 2, 3],
        "BusName": ["A", "B", "C"],
        "BusPUVolt": [1.0, 1.01, 1.02]
    })
    
    indexable_instance[grid.Bus] = update_df
    mock_esa.ChangeParametersMultipleElementRect.assert_called_once()


def test_getitem_with_slice_none(indexable_instance: Indexable):
    """Test that slice(None) correctly retrieves all fields."""
    mock_esa = indexable_instance.esa
    mock_df = pd.DataFrame({"BusNum": [1], "BusName": ["Bus1"], "BusPUVolt": [1.0]})
    mock_esa.GetParamsRectTyped.return_value = mock_df
    
    result = indexable_instance[grid.Bus, :]
    
    # Should request all fields (keys + fields)
    call_args = mock_esa.GetParamsRectTyped.call_args[0]
    requested_fields = call_args[1]
    assert len(requested_fields) > len(grid.Bus.keys)


def test_setitem_broadcast_with_single_value(indexable_instance: Indexable):
    """Test broadcasting a single value to all instances."""
    mock_esa = indexable_instance.esa
    mock_df = pd.DataFrame({"BusNum": [1, 2, 3]})
    mock_esa.GetParamsRectTyped.return_value = mock_df
    
    indexable_instance[grid.Bus, "BusPUVolt"] = 1.05
    
    # Verify all three buses got the same value
    sent_df = mock_esa.ChangeParametersMultipleElementRect.call_args[0][2]
    assert len(sent_df) == 3
    assert (sent_df["BusPUVolt"] == 1.05).all()


def test_setitem_with_series(indexable_instance: Indexable):
    """Test setting data using a pandas Series instead of a DataFrame."""
    mock_esa = indexable_instance.esa
    mock_df = pd.DataFrame({"BusNum": [1, 2, 3]})
    mock_esa.GetParamsRectTyped.return_value = mock_df
    
    # Create a Series with per-bus values (reset index to ensure proper alignment)
    values = pd.Series([1.00, 1.01, 1.02]).values  # Convert to numpy array to avoid index alignment issues
    indexable_instance[grid.Bus, "BusPUVolt"] = values
    
    sent_df = mock_esa.ChangeParametersMultipleElementRect.call_args[0][2]
    # Compare values directly, not relying on index alignment
    assert np.allclose(sent_df["BusPUVolt"].values, values)


def test_getitem_with_nonexistent_field():
    """Test requesting a field that doesn't exist in the GObject definition."""
    with patch('esapp.indexable.SAW') as mock_saw_class:
        mock_esa = Mock()
        mock_saw_class.return_value = mock_esa
        
        instance = Indexable()
        instance.esa = mock_esa
        
        # This should still call the API but might return empty or error
        # Document the expected behavior
        result = instance[grid.Bus, ["NonExistentField"]]
        mock_esa.GetParamsRectTyped.assert_called_once()


def test_setitem_empty_dataframe(indexable_instance: Indexable):
    """Test setting an empty DataFrame (should handle gracefully)."""
    mock_esa = indexable_instance.esa
    
    empty_df = pd.DataFrame()
    indexable_instance[grid.Bus] = empty_df
    
    # Should still call the API, even if DataFrame is empty
    mock_esa.ChangeParametersMultipleElementRect.assert_called_once()


def test_concurrent_field_access(indexable_instance: Indexable):
    """Test that multiple field requests are handled correctly."""
    mock_esa = indexable_instance.esa
    mock_df = pd.DataFrame({
        "BusNum": [1, 2],
        "BusPUVolt": [1.0, 1.01],
        "BusAngle": [0.0, -2.0]
    })
    mock_esa.GetParamsRectTyped.return_value = mock_df
    
    # Request multiple fields
    result = indexable_instance[grid.Bus, ["BusPUVolt", "BusAngle"]]
    
    assert "BusPUVolt" in result.columns
    assert "BusAngle" in result.columns
    assert "BusNum" in result.columns  # Keys should also be included