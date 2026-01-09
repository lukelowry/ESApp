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

1. **GObject Base Class** (``esapp/gobject.py``)
   
   A metaclass-based foundation that provides:
   
   - Field definition collection from class attributes
   - Primary key information management
   - Field priority marking (required, optional)
   - Informative string representations

2. **Field Definitions** (``esapp/grid.py``)
   
   Auto-generated classes defining all PowerWorld objects:
   
   .. code-block:: python
   
       class Bus(GObject):
           """A power system bus/node"""
           BusNum = (FieldPriority.PRIMARY_KEY, np.int32)
           BusName = (FieldPriority.OPTIONAL, str)
           BusPUVolt = (FieldPriority.OPTIONAL, np.float64)

3. **Generation Script** (``esapp/dev/generate_components.py``)
   
   Python script that:
   - Parses PowerWorld field export (PWRaw format)
   - Generates component class definitions
   - Handles field name sanitization
   - Assigns priority levels automatically

4. **Indexable Mixin** (``esapp/indexable.py``)
   
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
2. Place it in the ``esapp/dev/`` folder, overwriting the existing one

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

    python esapp/dev/generate_components.py

The script will:

1. Parse the PWRaw file
2. Generate Python class definitions
3. Assign field priorities based on PowerWorld metadata:
   
   - **PRIMARY_KEY**: Component identifier (e.g., BusNum for Bus objects)
   - **REQUIRED**: Must be specified when creating new objects
   - **OPTIONAL**: Can be read/written but not required
   
4. Create ``esapp/grid.py`` with all component classes
5. Print progress to console including any warnings or excluded fields

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

PRIMARY_KEY
    Fields marked as KeyType="KEY" in PWRaw
REQUIRED
    Fields marked as KeyType="REQUIRED"
OPTIONAL
    Remaining fields

**Conflict Resolution**
  - Fields with invalid names are excluded (rare)
  - Duplicate field names are logged
  - Output includes summary of excluded fields

**Component Class Generation**
  - Creates class for each ObjectType in PWRaw
  - Adds docstring with description
  - Defines field tuple ``(priority, data_type)`` for each field
  - Adds special attributes like ``_object_type``, ``_fields``, ``_keys``

Example Generated Component
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A generated component class looks like:

.. code-block:: python

    class Bus(GObject):
        """A power system bus/node - represents a point of electrical connection"""
        
        BusNum = (FieldPriority.PRIMARY_KEY, np.int32)
        BusName = (FieldPriority.OPTIONAL, str)
        BusPUVolt = (FieldPriority.OPTIONAL, np.float64)
        BusAngle = (FieldPriority.OPTIONAL, np.float64)
        AreaNum = (FieldPriority.OPTIONAL, np.int32)
        ZoneNum = (FieldPriority.OPTIONAL, np.int32)
        
        _object_type = "Bus"
        _fields = ["BusNum", "BusName", "BusPUVolt", ...]
        _keys = ["BusNum"]

Using Generated Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In user code, components are used for type-safe data access:

.. code-block:: python

    from esapp.grid import Bus
    
    data = wb[Bus, [Bus.BusNum, Bus.BusName, Bus.BusPUVolt]]
    data = wb[Bus, ["BusNum", "BusName", "BusPUVolt"]]

.. note::
   IDE provides autocompletion for all fields when using class attributes.

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
- Component classes in ``esapp.grid``
- Exception types in ``esapp.saw.exceptions``

Internal APIs (subject to change):

- SAW mixin implementations
- Indexable internals
- GObject metaclass details

Generally, this is the only step required to keep ESA++ compatible with new PowerWorld releases regarding data access and modification.