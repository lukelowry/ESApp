Development
===========

This section covers the internal maintenance and extension of the ESA++ toolkit, specifically focusing on how the library maintains its structured representation of PowerWorld objects to facilitate ObjectField access.

Component Architecture
----------------------

ESA++ uses a sophisticated class generation system to represent all PowerWorld objects and their fields. 
This architecture provides:

Type Safety
    IDE autocompletion and type hints for all components
Automatic Synchronization
    Stays compatible with new PowerWorld versions automatically
Maintainability
    No manual class definitions needed
Documentation
    Docstrings auto-generated from PowerWorld metadata

The System
~~~~~~~~~~

The component system consists of:

1. **GObject Base Class** (``esapp/components/gobject.py``)

   An Enum-based foundation that dynamically builds component schemas from class definitions:

   - Uses custom ``__new__`` to parse member definitions at class creation time
   - Collects fields into ``_FIELDS``, ``_KEYS``, ``_SECONDARY``, and ``_EDITABLE`` lists
   - Provides class properties for accessing schema information (``keys``, ``fields``, ``editable``, etc.)
   - Supports composable ``FieldPriority`` flags (PRIMARY, SECONDARY, REQUIRED, OPTIONAL, EDITABLE)

2. **Field Definitions** (``esapp/components/grid.py``)

   Auto-generated classes defining all PowerWorld objects:

   .. code-block:: python

       class Bus(GObject):
           # Fields: (PowerWorld field name, data type, priority flags)
           BusNum = ("BusNum", int, FieldPriority.PRIMARY)
           """Number"""
           BusName = ("BusName", str, FieldPriority.SECONDARY | FieldPriority.REQUIRED | FieldPriority.EDITABLE)
           """Name"""
           AreaNum = ("AreaNum", int, FieldPriority.SECONDARY | FieldPriority.REQUIRED | FieldPriority.EDITABLE)
           """Area Num"""

3. **Transient Stability Fields** (``esapp/components/ts_fields.py``)

   Auto-generated constants for TS result field intellisense:

   .. code-block:: python

       from esapp.components import TS

       # IDE autocomplete for all TS fields
       TS.Gen.P       # Generator active power
       TS.Gen.W       # Generator rotor speed
       TS.Bus.VPU     # Bus voltage magnitude

4. **Generation Script** (``esapp/components/generate_components.py``)

   Python script that:
   - Parses PowerWorld field export (PWRaw format)
   - Generates ``grid.py`` with GObject subclasses for all object types
   - Generates ``ts_fields.py`` with TS field constants for IDE intellisense
   - Handles field name sanitization and priority assignment

5. **Indexable Mixin** (``esapp/indexable.py``)
   
   Translates Python indexing syntax into SimAuto calls:
   
   .. code-block:: python
   
       buses = wb[Bus, ["BusNum", "BusPUVolt"]]
   
   .. note::
      Translates to ``SimAuto.GetDataRaw("Bus", None, ["BusNum", "BusPUVolt"])``

Updating Component Definitions
-------------------------------

When PowerWorld adds new fields or a new version is released, component definitions must be updated.

Step 1: Export the Field List from PowerWorld
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Open PowerWorld Simulator
2. Navigate to the **Window** ribbon
3. Click **Export Case Object Fields**
4. Save the resulting tab-delimited text file (typically named Export.txt or similar)

Step 2: Prepare the Raw Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Rename the exported file to ``PWRaw``
2. Place it in the ``esapp/components/`` folder, overwriting the existing one

The PWRaw file format is tab-delimited with columns:

.. code-block:: text

    ObjectType    FieldName    DataType    KeyType    Description
    Bus           BusNum       Integer     PRIMARY_KEY    Bus number identifier
    Bus           BusName      String      OPTIONAL       Bus name
    Bus           BusPUVolt    Double      OPTIONAL       Bus voltage in per-unit
    ...

Step 3: Run the Generation Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the generation script from the project root:

.. code-block:: bash

    python esapp/components/generate_components.py

The script will:

1. Parse the PWRaw file
2. Generate Python class definitions
3. Assign field priorities based on PowerWorld metadata:

   - **PRIMARY**: Primary key field that identifies the object
   - **SECONDARY**: Alternate identifier field (e.g., names)
   - **REQUIRED**: Must be specified when creating new objects
   - **OPTIONAL**: Can be read/written but not required
   - **EDITABLE**: User-modifiable field

4. Create ``esapp/components/grid.py`` with all GObject component classes
5. Create ``esapp/components/ts_fields.py`` with TS field constants
6. Print progress to console including any warnings or excluded fields

Step 4: Verify the Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Check console output for errors or excluded objects/fields:

   .. code-block:: text

       Processed 150 object types with 5247 total fields
       Excluded: 12 fields due to naming conflicts
       Component definitions updated successfully

2. Run unit tests to verify component generation:

   .. code-block:: bash

       pytest tests/test_grid_components.py -v

3. Run integration tests if PowerWorld is available:

   .. code-block:: bash

       pytest tests/test_integration_workbench.py -v

Generation Script Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``generate_components.py`` script handles several important tasks:

**Field Name Sanitization**

Colons
    ``Bus:Num`` → ``Bus__Num`` (stored as ``Bus:Num`` internally)
Spaces
    ``Line Name`` → ``Line_Name``
Identifiers
    Converts to valid Python identifiers

**Priority Assignment**

PRIMARY
    Fields marked as primary keys in PWRaw (identifies the object)
SECONDARY
    Fields that serve as alternate identifiers (e.g., name fields)
REQUIRED
    Fields that must be specified when creating objects
OPTIONAL
    Fields that can be read/written but are not required
EDITABLE
    Fields that users can modify (combined with other flags)

**Conflict Resolution**
  - Fields with invalid names are excluded (rare)
  - Duplicate field names are logged
  - Output includes summary of excluded fields

**Component Class Generation**
  - Creates GObject subclass for each ObjectType in PWRaw
  - First member ``_`` defines the PowerWorld object type string
  - Subsequent members define fields as ``(PowerWorld name, data type, priority flags)``
  - The GObject ``__new__`` method dynamically populates ``_FIELDS``, ``_KEYS``, ``_SECONDARY``, ``_EDITABLE``

Example Generated Component
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A generated component class follows this pattern (excerpt from actual ``Bus`` class):

.. code-block:: python

    class Bus(GObject):
        # Fields: (PowerWorld field name, Python type, composable priority flags)
        BusNum = ("BusNum", int, FieldPriority.PRIMARY)
        """Number"""
        BusName_NomVolt = ("BusName_NomVolt", str, FieldPriority.SECONDARY)
        """Name_Nominal kV"""
        AreaNum = ("AreaNum", int, FieldPriority.SECONDARY | FieldPriority.REQUIRED | FieldPriority.EDITABLE)
        """Area Num"""
        BusName = ("BusName", str, FieldPriority.SECONDARY | FieldPriority.REQUIRED | FieldPriority.EDITABLE)
        """Name"""
        BusNomVolt = ("BusNomVolt", float, FieldPriority.SECONDARY | FieldPriority.REQUIRED | FieldPriority.EDITABLE)
        """The nominal kV voltage specified as part of the input file."""
        ZoneNum = ("ZoneNum", int, FieldPriority.SECONDARY | FieldPriority.REQUIRED | FieldPriority.EDITABLE)
        """Number of the Zone"""
        # ... many more fields

The GObject base class automatically collects these definitions and exposes them via class properties:

.. code-block:: python

    Bus.TYPE        # 'Bus' - PowerWorld object type string
    Bus.keys        # ['BusNum'] - primary key fields
    Bus.fields      # ['BusNum', 'BusName_NomVolt', 'AreaNum', 'BusName', ...] - all fields
    Bus.secondary   # ['BusName_NomVolt', 'AreaNum', 'BusName', 'BusNomVolt', 'ZoneNum', ...] - secondary identifier fields
    Bus.editable    # ['AreaNum', 'BusName', 'BusNomVolt', 'ZoneNum', ...] - user-modifiable fields
    Bus.identifiers # set of all identifier fields (keys + secondary)
    Bus.settable    # set of all settable fields (identifiers + editable)

Using Generated Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In user code, components are used for type-safe data access:

.. code-block:: python

    from esapp.components import Bus, Gen
    from esapp import TS

    # Access data using component classes
    data = wb[Bus, ["BusNum", "BusName", "BusPUVolt"]]

    # Use class attributes for field names (IDE autocomplete)
    data = wb[Bus, [Bus.BusNum, Bus.BusName, Bus.BusPUVolt]]

    # Check field properties
    Bus.is_editable('BusName')  # True
    Bus.is_settable('BusNum')   # True (it's a key)

    # For transient stability, use TS for field intellisense
    from esapp.components import TS
    wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])

.. note::
   IDE provides autocompletion for all fields when using class attributes. The ``TS`` class
   provides organized access to transient stability result fields.

Maintenance Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**After PowerWorld Upgrade:**
  - Export new field list immediately
  - Run generation script to update components
  - Run full test suite to verify compatibility
  - Commit updated components file to version control

**Periodic Cleanup:**
  - Monitor for excluded fields (usually rare)
  - Review field name sanitization for any issues
  - Update version requirements if significant changes occur

**Documentation:**
  - Keep README updated with supported PowerWorld versions
  - Document any known field limitations or quirks
  - Maintain changelog of compatibility updates

Extending ESA++
---------------

**Adding New Analysis Methods**

To add a new analysis capability to GridWorkBench:

1. Create a new mixin in ``esapp/saw/`` (e.g., ``custom_analysis.py``)
2. Implement method using SAW interface
3. Add mixin to SAW class in ``esapp/saw/saw.py``
4. Add convenience method to GridWorkBench if commonly used

**Adding Helper Functions**

New utility functions should go in:

- ``esapp/utils/`` for general utilities
- ``esapp/saw/_helpers.py`` for SAW-specific helpers
- ``esapp/apps/`` for domain-specific analysis

**Contributing Tests**

When adding features:

1. Add unit tests to ``tests/``
2. Add integration tests if PowerWorld interaction
3. Update test documentation
4. Run full test suite before submitting

API Stability
~~~~~~~~~~~~~

ESA++ maintains semantic versioning:

- **MAJOR**: Breaking changes to public API
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes

The public API includes:

- GridWorkBench class and all public methods
- Component classes in ``esapp.components``
- Exception types in ``esapp.saw.exceptions``

Internal APIs (subject to change):

- SAW mixin implementations
- Indexable internals
- GObject metaclass details

Generally, this is the only step required to keep ESA++ compatible with new PowerWorld releases regarding data access and modification.