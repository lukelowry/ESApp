"""
Unit tests for GObject and FieldPriority from esapp.grid module.

WHAT THIS TESTS:
- FieldPriority flag enum functionality and bitwise operations
- GObject metaclass behavior and field collection
- Component class generation from PowerWorld object definitions
- Field type validation across all component types (Bus, Gen, Load, etc.)
- Docstring presence and name collision detection

DEPENDENCIES: None (mocked, no PowerWorld required)

USAGE:
    pytest tests/test_grid_components.py -v
"""
import pytest
import inspect
from enum import Flag
from typing import Type, List

from esapp import grid

# Import shared test utility
from conftest import get_all_gobject_subclasses

# --- Fixtures ---

@pytest.fixture(scope="module")
def test_gobject_class() -> Type[grid.GObject]:
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


def test_gobject_editable_fields_are_collected(test_gobject_class):
    """Tests that EDITABLE fields are collected in the .editable property."""
    expected_editable = ['value']
    assert test_gobject_class.editable == expected_editable


def test_gobject_secondary_fields_are_collected(test_gobject_class):
    """Tests that SECONDARY fields are collected in the .secondary property."""
    # NAME is SECONDARY, DUPLICATE_KEY is both PRIMARY and SECONDARY
    expected_secondary = ['name', 'duplicate_key']
    assert test_gobject_class.secondary == expected_secondary


def test_gobject_identifiers(test_gobject_class):
    """Tests that identifiers returns primary + secondary keys."""
    # id and duplicate_key are PRIMARY, name and duplicate_key are SECONDARY
    expected_identifiers = {'id', 'name', 'duplicate_key'}
    assert test_gobject_class.identifiers == expected_identifiers


def test_gobject_settable_fields(test_gobject_class):
    """Tests that settable returns identifiers (primary + secondary) + editable fields."""
    # identifiers: id, name, duplicate_key; editable: value
    expected_settable = {'id', 'name', 'duplicate_key', 'value'}
    assert test_gobject_class.settable == expected_settable


def test_gobject_is_editable(test_gobject_class):
    """Tests is_editable() helper method."""
    assert test_gobject_class.is_editable('value') is True
    assert test_gobject_class.is_editable('id') is False
    assert test_gobject_class.is_editable('name') is False
    assert test_gobject_class.is_editable('nonexistent') is False


def test_gobject_is_settable(test_gobject_class):
    """Tests is_settable() helper method."""
    assert test_gobject_class.is_settable('value') is True  # Editable
    assert test_gobject_class.is_settable('id') is True  # Primary key
    assert test_gobject_class.is_settable('duplicate_key') is True  # Primary + Secondary key
    assert test_gobject_class.is_settable('name') is True  # Secondary key (identifier)
    assert test_gobject_class.is_settable('nonexistent') is False

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

def test_gobject_str_representation(test_gobject_class: Type[grid.GObject]):
    """Tests the __str__ representation of a GObject member."""
    assert str(test_gobject_class.NAME) == "name"


def test_gobject_field_access_by_name(test_gobject_class: Type[grid.GObject]):
    """Tests that fields can be accessed by their string name using getattr."""
    id_field = getattr(test_gobject_class, "ID")
    assert id_field.value[1] == "id"
    assert id_field.value[2] == int


def test_gobject_duplicate_field_names():
    """Tests that duplicate field names raise an error or are handled gracefully."""
    # This test documents expected behavior when duplicate field names are defined
    try:
        class DuplicateFields(grid.GObject):
            FIELD1 = ("same_name", int, grid.FieldPriority.PRIMARY)
            FIELD2 = ("same_name", str, grid.FieldPriority.SECONDARY)
            ObjectString = "DuplicateTest"
        # If no error, check that both are in fields list
        assert "same_name" in DuplicateFields.fields
    except (ValueError, TypeError) as e:
        # Document that this is expected to fail
        pytest.skip(f"Duplicate field names not allowed: {e}")


def test_gobject_empty_object():
    """Tests GObject subclass with no fields."""
    class EmptyObject(grid.GObject):
        ObjectString = "EmptyObject"

    assert EmptyObject.TYPE == "EmptyObject"
    assert EmptyObject.fields == []
    assert EmptyObject.keys == []
    assert EmptyObject.editable == []
    assert EmptyObject.settable == set()

# --- Parametrized tests for all GObject subclasses in components.py ---

@pytest.mark.parametrize("g_object_class", get_all_gobject_subclasses())
def test_real_gobject_subclass_is_well_formed(g_object_class: Type[grid.GObject]):
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
    assert hasattr(g_object_class, '_EDITABLE'), f"{g_object_class.__name__} is missing _EDITABLE."
    assert isinstance(g_object_class.editable, list)
    assert set(g_object_class.keys).issubset(set(g_object_class.fields)), \
        f"Not all keys in {g_object_class.__name__} are in its fields list."
    assert set(g_object_class.editable).issubset(set(g_object_class.fields)), \
        f"Not all editable fields in {g_object_class.__name__} are in its fields list."
    assert set(g_object_class.secondary).issubset(set(g_object_class.fields)), \
        f"Not all secondary fields in {g_object_class.__name__} are in its fields list."

    # Verify identifiers is the union of keys and secondary
    expected_identifiers = set(g_object_class.keys) | set(g_object_class.secondary)
    assert g_object_class.identifiers == expected_identifiers, \
        f"Identifiers mismatch in {g_object_class.__name__}"

    # Verify settable is the union of identifiers and editable
    expected_settable = expected_identifiers | set(g_object_class.editable)
    assert g_object_class.settable == expected_settable, \
        f"Settable mismatch in {g_object_class.__name__}"


@pytest.mark.parametrize("g_object_class", get_all_gobject_subclasses())
def test_gobject_field_types(g_object_class: Type[grid.GObject]):
    """
    Tests that all fields in real GObject subclasses have valid Python types.
    """
    valid_types = (int, float, str, bool, type(None))
    for member in g_object_class:
        if hasattr(member.value, '__len__') and len(member.value) >= 3:
            field_type = member.value[2]
            assert field_type in valid_types, \
                f"{g_object_class.__name__}.{member.name} has invalid type: {field_type}"


def test_documentation_coverage_summary():
    """
    Reports overall field documentation coverage across all GObject subclasses.
    This is an informational test that doesn't fail - it summarizes docstring coverage.
    """
    all_classes = get_all_gobject_subclasses()
    documented = 0
    undocumented = []

    for cls in all_classes:
        members = list(cls)
        if not members:
            continue
        if any(m.__doc__ and m.__doc__.strip() for m in members):
            documented += 1
        else:
            undocumented.append(cls.__name__)

    total = len(all_classes)
    coverage = documented / total if total > 0 else 0

    # Print summary (visible with pytest -v or pytest -s)
    print(f"\n{'='*60}")
    print(f"GObject Documentation Coverage: {coverage:.1%} ({documented}/{total} components)")
    print(f"{'='*60}")


@pytest.mark.parametrize("g_object_class", get_all_gobject_subclasses())
def test_gobject_no_name_collisions(g_object_class: Type[grid.GObject]):
    """
    Tests that field names don't collide with Python keywords or common methods.
    """
    reserved_names = {'class', 'def', 'if', 'else', 'for', 'while', 'return', 
                      'import', 'from', 'type', 'fields', 'keys', 'TYPE'}
    
    for field_name in g_object_class.fields:
        assert field_name.lower() not in reserved_names, \
            f"{g_object_class.__name__} has field '{field_name}' that collides with reserved name"