SimAuto Wrapper (SAW)
=====================

The ``SAW`` (SimAuto Wrapper) class provides complete access to PowerWorld's SimAuto API.
It is organized into functional mixins for power flow, contingencies, optimization, sensitivity,
transient stability, GIC, ATC, topology, and data management. Access SAW through ``wb.esa``
from ``GridWorkBench``.

SAW Class
---------

.. currentmodule:: esapp.saw

.. autoclass:: SAW
   :show-inheritance:
   :members:
   :inherited-members:
   :member-order: groupwise

Type-Safe Enumerations
----------------------

ESA++ provides enumeration types for common PowerWorld string parameters. Using these
enums instead of raw strings provides IDE autocomplete, type checking, and prevents typos.

.. code-block:: python

    from esapp.saw import SolverMethod, FilterKeyword, LinearMethod

    # Use enums for type-safe parameters
    saw.SolvePowerFlow(SolverMethod.RECTNEWT)
    saw.GetParametersMultipleElement("Bus", ["BusNum", "BusPUVolt"], FilterKeyword.ALL)

SolverMethod
~~~~~~~~~~~~

Power flow solution algorithms for the ``SolvePowerFlow`` command.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``RECTNEWT``
     - Rectangular Newton-Raphson (default)
   * - ``POLARNEWT``
     - Polar Newton-Raphson
   * - ``GAUSSSEIDEL``
     - Gauss-Seidel iterative method
   * - ``FASTDEC``
     - Fast Decoupled method
   * - ``ROBUST``
     - Robust solver for difficult cases
   * - ``DC``
     - DC power flow (linear approximation)

LinearMethod
~~~~~~~~~~~~

Linear calculation methods for sensitivity analysis (PTDF, LODF, shift factors).

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``DC``
     - DC linear method (most common default)
   * - ``AC``
     - AC linear method
   * - ``DCPS``
     - DC linear with post-solution adjustment

FilterKeyword
~~~~~~~~~~~~~

Special filter keywords passed unquoted to PowerWorld commands.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``ALL``
     - Select all objects of the type
   * - ``SELECTED``
     - Only objects currently selected in PowerWorld
   * - ``AREAZONE``
     - Objects in the active area/zone filter

YesNo
~~~~~

Boolean flag values for PowerWorld commands that use "YES"/"NO" strings.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``YES``
     - Affirmative / enable option
   * - ``NO``
     - Negative / disable option

Use ``YesNo.from_bool(value)`` to convert Python booleans.

FileFormat
~~~~~~~~~~

File format types for import/export operations.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``CSV``
     - Comma-separated values
   * - ``CSVCOLHEADER``
     - CSV with column headers
   * - ``CSVNOHEADER``
     - CSV without headers
   * - ``AUX``
     - PowerWorld auxiliary format
   * - ``AUXCSV``
     - Hybrid auxiliary/CSV format
   * - ``TAB``
     - Tab-separated format
   * - ``PTI``
     - PTI/PSS-E format
   * - ``TXT``
     - Text format
   * - ``PWB``
     - PowerWorld case format
   * - ``AXD``
     - Oneline diagram format
   * - ``GE``
     - GE EPC format
   * - ``CF``
     - Custom format
   * - ``UCTE``
     - UCTE format
   * - ``AREVAHDB``
     - AREVA HDB format
   * - ``OPENNETEMS``
     - OPENNET EMS format

ObjectType
~~~~~~~~~~

PowerWorld object type identifiers for filtering and operations.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``BUS``
     - Bus/node
   * - ``BRANCH``
     - Branch (line or transformer)
   * - ``GEN``
     - Generator
   * - ``LOAD``
     - Load
   * - ``SHUNT``
     - Shunt device
   * - ``AREA``
     - Control area
   * - ``ZONE``
     - Zone
   * - ``OWNER``
     - Owner
   * - ``INTERFACE``
     - Interface (flowgate)
   * - ``INJECTIONGROUP``
     - Injection group
   * - ``BUSSHUNT``
     - Bus shunt
   * - ``SUPERBUS``
     - Super bus (aggregated)
   * - ``TRANSFORMER``
     - Transformer specifically
   * - ``LINE``
     - Transmission line specifically
   * - ``SUPERAREA``
     - Super area (aggregated)

KeyFieldType
~~~~~~~~~~~~

Key field types for result output formatting.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``PRIMARY``
     - Primary key fields (e.g., BusNum)
   * - ``SECONDARY``
     - Secondary key fields (e.g., BusName)
   * - ``LABEL``
     - Label-based identification

IslandReference
~~~~~~~~~~~~~~~

Island reference options for sensitivity analysis.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``EXISTING``
     - Use existing island configuration
   * - ``NO``
     - No area reference

Other Enumerations
~~~~~~~~~~~~~~~~~~

Additional specialized enumerations for specific operations:

**InterfaceLimitSetting** - Interface limit configuration

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``AUTO``
     - Automatic limit calculation
   * - ``NONE``
     - No limit applied

**StarBusHandling** - Star bus handling for case append

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``NEAR``
     - Map to nearest bus (default)
   * - ``MAX``
     - Map to maximum impedance bus

**MultiSectionLineHandling** - Multi-section line handling for case append

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``MAINTAIN``
     - Maintain multisection line structure (default)
   * - ``EQUIVALENCE``
     - Convert to equivalent circuits

**OnelineLinkMode** - Oneline diagram linking modes

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``LABELS``
     - Link objects by labels (default)
   * - ``NUMBERS``
     - Link objects by numbers

**ShuntModel** - Shunt model types for line tapping

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``CAPACITANCE``
     - Capacitive shunt model
   * - ``INDUCTANCE``
     - Inductive shunt model

**BranchDeviceType** - Branch device types for bus splitting

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``Line``
     - Transmission line
   * - ``Breaker``
     - Circuit breaker

**TSGetResultsMode** - Transient stability results save mode

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``SINGLE``
     - Single combined output file
   * - ``SEPARATE``
     - Separate files per object
   * - ``JSIS``
     - JSIS format output

Helper Functions
~~~~~~~~~~~~~~~~

Functions for formatting filter parameters:

.. autofunction:: esapp.saw.format_filter

.. autofunction:: esapp.saw.format_filter_selected_only

.. autofunction:: esapp.saw.format_filter_areazone

Exceptions
----------

.. autoclass:: esapp.saw.PowerWorldError
   :show-inheritance:

.. autoclass:: esapp.saw.COMError
   :show-inheritance:

.. autoclass:: esapp.saw.CommandNotRespectedError
   :show-inheritance:

.. autoclass:: esapp.saw.SimAutoFeatureError
   :show-inheritance:

.. autoclass:: esapp.saw.PowerWorldPrerequisiteError
   :show-inheritance:

.. autoclass:: esapp.saw.PowerWorldAddonError
   :show-inheritance:
