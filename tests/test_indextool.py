"""
Unit tests for the Indexable class, using pytest for clarity and robustness.
Tests focus on the __getitem__ and __setitem__ methods for data I/O.
"""
import pytest
from unittest.mock import Mock, patch
from typing import Type, List
import pandas as pd
from pandas.testing import assert_frame_equal

from gridwb.indexable import Indexable
from gridwb import grid


def get_all_gobject_subclasses() -> List[Type[grid.GObject]]:
    """Recursively finds all non-abstract, testable GObject subclasses."""
    all_subclasses = []
    q = list(grid.GObject.__subclasses__())
    visited = set(q)
    while q:
        cls = q.pop(0)
        # A concrete, testable GObject subclass must have a _TYPE attribute
        # and at least one field defined.
        if hasattr(cls, '_TYPE'):
            all_subclasses.append(cls)

        for subclass in cls.__subclasses__():
            if subclass not in visited:
                visited.add(subclass)
                q.append(subclass)
    return all_subclasses


@pytest.fixture
def indexable_instance() -> Indexable:
    """Provides an Indexable instance with a mocked SAW dependency."""
    with patch('gridwb.indexable.SAW') as mock_saw_class:
        mock_esa = Mock()
        mock_saw_class.return_value = mock_esa
        
        instance = Indexable()
        instance.esa = mock_esa
        yield instance


@pytest.mark.parametrize("g_object", get_all_gobject_subclasses())
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


@pytest.mark.parametrize("g_object", get_all_gobject_subclasses())
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


@pytest.mark.parametrize("g_object", get_all_gobject_subclasses())
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


@pytest.mark.parametrize("g_object", get_all_gobject_subclasses())
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


@pytest.mark.parametrize("g_object", get_all_gobject_subclasses())
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


@pytest.mark.parametrize("g_object", get_all_gobject_subclasses())
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