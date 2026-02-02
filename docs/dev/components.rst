Component System
================

ESA++ auto-generates Python classes from PowerWorld's field metadata, providing type-safe
access to all SimAuto objects with IDE autocompletion.

Architecture Overview
---------------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``gobject.py``
     - Base class using Enum mechanics to build component schemas at class creation
   * - ``grid.py``
     - Auto-generated GObject subclasses for all PowerWorld object types
   * - ``ts_fields.py``
     - Auto-generated TS field constants for transient stability intellisense
   * - ``generate_components.py``
     - Script that parses PWRaw export and generates the above files
   * - ``indexable.py``
     - Translates ``wb[Bus, "field"]`` syntax into SimAuto calls

GObject Base Class
~~~~~~~~~~~~~~~~~~

Each component is an Enum subclass where members define fields. A single-argument
member (``ObjectString``) sets the PowerWorld object type, while 3-tuple members
define fields with ``(PowerWorld name, data type, priority flags)``:

.. code-block:: python

    class Bus(GObject):
        BusNum = ("BusNum", int, FieldPriority.PRIMARY)
        """Number"""
        BusName = ("BusName", str, FieldPriority.SECONDARY | FieldPriority.REQUIRED | FieldPriority.EDITABLE)
        """Name"""
        # ... more fields ...

        ObjectString = 'Bus'  # Sets Bus.TYPE — must be last member

The base class collects these into queryable properties:

.. code-block:: python

    Bus.TYPE        # 'Bus' - PowerWorld object type (from ObjectString)
    Bus.keys        # ['BusNum'] - primary key fields
    Bus.fields      # all field names
    Bus.secondary   # alternate identifier fields
    Bus.editable    # user-modifiable fields

Field Priority Flags
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Flag
     - Meaning
   * - PRIMARY
     - Primary key that uniquely identifies the object
   * - SECONDARY
     - Alternate identifier (e.g., name fields)
   * - REQUIRED
     - Must be specified when creating new objects
   * - OPTIONAL
     - Can be read/written but not required
   * - EDITABLE
     - User-modifiable field (combined with other flags)

Updating Components
-------------------

When PowerWorld releases new versions or adds fields, regenerate the component definitions.

**Step 1: Export Field List**

In PowerWorld Simulator: **Window** → **Export Case Object Fields** → Save as tab-delimited text.

**Step 2: Replace PWRaw**

Copy the exported file to ``esapp/components/PWRaw``, overwriting the existing one.

**Step 3: Run Generator**

.. code-block:: bash

    python esapp/components/generate_components.py

**Step 4: Verify**

.. code-block:: bash

    pytest tests/test_grid_components.py -v
    pytest tests/test_integration_workbench.py -v  # if PowerWorld available

The script sanitizes field names (``Bus:Num`` → ``Bus__Num``, spaces → underscores) and
logs any excluded fields due to naming conflicts.

Usage Examples
--------------

.. code-block:: python

    from esapp.components import Bus, Gen, TS

    # Data access with component classes
    data = wb[Bus, [Bus.BusNum, Bus.BusName, Bus.BusPUVolt]]

    # Check field properties
    Bus.is_editable('BusName')  # True
    Bus.is_settable('BusNum')   # True (it's a key)

    # Transient stability fields
    tsw = TSWatch()
    tsw.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])

Extending ESA++
---------------

**New Analysis Methods**

1. Create mixin in ``esapp/saw/`` (e.g., ``custom_analysis.py``)
2. Implement using SAW interface
3. Add mixin to SAW class in ``esapp/saw/saw.py``
4. Add convenience wrapper to GridWorkBench if commonly used

**New Utilities**

- General utilities → ``esapp/utils/``
- SAW-specific helpers → ``esapp/saw/_helpers.py``
- Domain analysis → ``esapp/utils/`` (GIC, Network, ContingencyBuilder)
- Example applications → ``examples/`` (Statics, Dynamics)

**Contributing Tests**

Add unit tests to ``tests/``, integration tests for PowerWorld interactions,
and run the full suite before submitting.

API Stability
-------------

ESA++ uses semantic versioning:

- **Public API** (stable): GridWorkBench, component classes, exception types
- **Internal API** (may change): SAW mixins, Indexable internals, GObject metaclass
