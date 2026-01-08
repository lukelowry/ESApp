import pytest
import inspect
from enum import Flag

from gridwb import grid

# --- Fixtures ---

@pytest.fixture(scope="module")
def test_gobject_class():
    """A simple GObject subclass for testing purposes."""
    class TestGObject(grid.GObject):
        ID = ("id", int, grid.FieldPriority.PRIMARY)
        NAME = ("name", str, grid.FieldPriority.SECONDARY | grid.FieldPriority.REQUIRED)
        VALUE = ("value", float, grid.FieldPriority.OPTIONAL | grid.FieldPriority.EDITABLE)
        DUPLICATE_KEY = ("duplicate_key", str, grid.FieldPriority.PRIMARY | grid.FieldPriority.SECONDARY)
        ObjectString = "TestGObject"
    return TestGObject

# --- Tests for FieldPriority ---

def test_fieldpriority_is_flag():
    """Ensures FieldPriority is a Flag enum, allowing bitwise operations."""
    assert issubclass(grid.FieldPriority, Flag)

def test_fieldpriority_combinations():
    """Tests bitwise combinations of FieldPriority flags."""
    primary_required = grid.FieldPriority.PRIMARY | grid.FieldPriority.REQUIRED
    assert grid.FieldPriority.PRIMARY in primary_required
    assert grid.FieldPriority.REQUIRED in primary_required
    assert grid.FieldPriority.SECONDARY not in primary_required

# --- Tests for GObject ---

def test_gobject_type_is_set(test_gobject_class):
    """Tests that the _TYPE class attribute is correctly set from ObjectString."""
    assert test_gobject_class.TYPE == "TestGObject"

def test_gobject_with_no_type():
    """Tests GObject subclass without an ObjectString."""
    class NoTypeObject(grid.GObject):
        FIELD = ("field", str, grid.FieldPriority.OPTIONAL)
    
    assert NoTypeObject.TYPE == 'NO_OBJECT_NAME'

def test_gobject_fields_are_collected(test_gobject_class):
    """Tests that all field names are collected in the .fields property."""
    expected_fields = ['id', 'name', 'value', 'duplicate_key']
    assert test_gobject_class.fields == expected_fields

def test_gobject_keys_are_collected(test_gobject_class):
    """
    Tests that PRIMARY fields are collected in the .keys property.
    """
    expected_keys = ['id', 'duplicate_key']
    assert test_gobject_class.keys == expected_keys

@pytest.mark.parametrize("member, expected_value", [
    ("ID", (1, 'id', int, grid.FieldPriority.PRIMARY)),
    ("NAME", (2, 'name', str, grid.FieldPriority.SECONDARY | grid.FieldPriority.REQUIRED)),
    ("VALUE", (3, 'value', float, grid.FieldPriority.OPTIONAL | grid.FieldPriority.EDITABLE)),
    ("DUPLICATE_KEY", (4, 'duplicate_key', str, grid.FieldPriority.PRIMARY | grid.FieldPriority.SECONDARY)),
    ("ObjectString", 5)
])
def test_gobject_member_values(test_gobject_class, member, expected_value):
    """Tests the underlying .value of each enum member."""
    assert getattr(test_gobject_class, member).value == expected_value

def test_gobject_str_representation(test_gobject_class):
    """Tests the __str__ representation of a GObject member."""
    assert str(test_gobject_class.NAME) == "Field String: name"

# --- Parametrized tests for all GObject subclasses in components.py ---

def get_gobject_subclasses():
    """Helper to discover all GObject subclasses in the components module."""
    return [
        obj for _, obj in inspect.getmembers(grid, inspect.isclass)
        if issubclass(obj, grid.GObject) and obj is not grid.GObject
    ]

@pytest.mark.parametrize("g_object_class", get_gobject_subclasses())
def test_real_gobject_subclass_is_well_formed(g_object_class):
    """
    Performs basic sanity checks on all GObject subclasses found in components.py.
    This ensures that the metaprogramming has worked as expected for all defined objects.
    """
    assert g_object_class.TYPE != 'NO_OBJECT_NAME', f"{g_object_class.__name__} is missing an ObjectString."
    assert isinstance(g_object_class.TYPE, str)
    assert hasattr(g_object_class, '_FIELDS'), f"{g_object_class.__name__} is missing _FIELDS."
    assert isinstance(g_object_class.fields, list)
    assert hasattr(g_object_class, '_KEYS'), f"{g_object_class.__name__} is missing _KEYS."
    assert isinstance(g_object_class.keys, list)
    assert set(g_object_class.keys).issubset(set(g_object_class.fields)), \
        f"Not all keys in {g_object_class.__name__} are in its fields list."

    if len(g_object_class.keys) != len(set(g_object_class.keys)):
        pytest.warning(
            f"{g_object_class.__name__} has duplicate keys. "
            f"This is likely due to fields being marked as both PRIMARY and SECONDARY. "
            f"Keys: {g_object_class.keys}"
        )