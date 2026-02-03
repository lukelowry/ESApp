"""
Unit tests for the GObject base class and FieldPriority flag.

These are **unit tests** that do NOT require PowerWorld Simulator. They test
GObject schema construction, field access, string representation, and
bitwise FieldPriority flag operations.

USAGE:
    pytest tests/test_gobject.py -v
"""
import pytest

from esapp.components.gobject import GObject, FieldPriority
from esapp import components as grid


class TestGObjectReprStr:
    """Test __repr__ and __str__ methods."""

    def test_str_type_defining_member(self):
        """str() on type-defining member returns the member name."""
        # ObjectString is the type-defining member with an int value (not a tuple)
        type_member = grid.Bus.ObjectString
        assert isinstance(type_member._value_, int)
        assert str(type_member) == "ObjectString"

    def test_repr_type_defining_member(self):
        """repr() on type-defining member shows TYPE info."""
        type_member = grid.Bus.ObjectString
        result = repr(type_member)
        assert "Bus.ObjectString" in result
        assert "TYPE=" in result

    def test_str_field_member(self):
        """str() on field member returns the PowerWorld field name."""
        # Get a field member (not the type-defining member)
        for member in grid.Bus:
            if member.name != "_":
                field_str = str(member)
                # Field members have a tuple value where _value_[1] is the field name
                assert isinstance(field_str, str)
                assert field_str  # Not empty
                break

    def test_repr_field_member(self):
        """repr() on field member shows field info."""
        for member in grid.Bus:
            if member.name != "_":
                result = repr(member)
                assert "Field=" in result
                break


class TestFieldPriority:
    """Test FieldPriority flag combinations."""

    def test_primary_flag(self):
        """PRIMARY flag can be checked with bitwise AND."""
        priority = FieldPriority.PRIMARY
        assert priority & FieldPriority.PRIMARY == FieldPriority.PRIMARY

    def test_combined_flags(self):
        """Flags can be combined and checked individually."""
        priority = FieldPriority.PRIMARY | FieldPriority.EDITABLE
        assert priority & FieldPriority.PRIMARY == FieldPriority.PRIMARY
        assert priority & FieldPriority.EDITABLE == FieldPriority.EDITABLE
        assert priority & FieldPriority.SECONDARY != FieldPriority.SECONDARY


class TestGObjectProperties:
    """Test GObject class properties."""

    def test_keys_property(self):
        """keys property returns primary key fields."""
        assert isinstance(grid.Bus.keys(), list)
        assert "BusNum" in grid.Bus.keys()

    def test_fields_property(self):
        """fields property returns all field names."""
        assert isinstance(grid.Bus.fields(), list)
        assert len(grid.Bus.fields()) > 0

    def test_secondary_property(self):
        """secondary property returns secondary identifier fields."""
        # Gen has secondary identifiers (alternate keys, base values)
        assert isinstance(grid.Gen.secondary(), list)
        # BusName_NomVolt is a secondary identifier (alternate key)
        assert "BusName_NomVolt" in grid.Gen.secondary()

    def test_editable_property(self):
        """editable property returns editable fields."""
        assert isinstance(grid.Bus.editable(), list)

    def test_identifiers_property(self):
        """identifiers includes both primary and secondary keys."""
        identifiers = grid.Gen.identifiers()
        assert isinstance(identifiers, set)
        assert "BusNum" in identifiers  # Primary key
        assert "GenID" in identifiers   # Primary key (composite)

    def test_settable_property(self):
        """settable includes identifiers and editable fields."""
        settable = grid.Bus.settable()
        assert isinstance(settable, set)

    def test_is_editable_method(self):
        """is_editable correctly identifies editable fields."""
        editable_fields = grid.Bus.editable()
        if editable_fields:
            assert grid.Bus.is_editable(editable_fields[0]) is True
        # Non-existent field should return False
        assert grid.Bus.is_editable("NonExistentField") is False

    def test_is_settable_method(self):
        """is_settable correctly identifies settable fields."""
        # Key fields are settable
        key = grid.Bus.keys()[0]
        assert grid.Bus.is_settable(key) is True
        # Non-existent field should return False
        assert grid.Bus.is_settable("NonExistentField") is False

    def test_type_property(self):
        """TYPE property returns the PowerWorld object type string."""
        assert grid.Bus.TYPE() == "Bus"
        assert grid.Gen.TYPE() == "Gen"
        assert grid.Load.TYPE() == "Load"


class TestGObjectWithoutType:
    """Test edge cases with GObject base class."""

    def test_base_gobject_type_default(self):
        """Base GObject class returns default TYPE when not set."""
        # GObject itself has no _TYPE attribute, should return default
        assert GObject.TYPE() == "NO_OBJECT_NAME"

    def test_base_gobject_empty_keys(self):
        """Base GObject class returns empty keys list."""
        assert GObject.keys() == []

    def test_base_gobject_empty_fields(self):
        """Base GObject class returns empty fields list."""
        assert GObject.fields() == []
