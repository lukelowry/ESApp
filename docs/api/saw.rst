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

Power Flow & Analysis
~~~~~~~~~~~~~~~~~~~~~

**SolverMethod** - Power flow solution algorithms

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

**LinearMethod** - Sensitivity analysis methods (PTDF, LODF, shift factors)

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

**JacobianForm** - Jacobian matrix coordinate forms

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``RECTANGULAR``
     - AC Jacobian in Rectangular coordinates ("R")
   * - ``POLAR``
     - AC Jacobian in Polar coordinates ("P")
   * - ``DC``
     - B' matrix / DC approximation

Filtering & Selection
~~~~~~~~~~~~~~~~~~~~~

**FilterKeyword** - Special filter keywords (passed unquoted)

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

**YesNo** - Boolean flags for PowerWorld commands

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

**ObjectType** - PowerWorld object type identifiers

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

**KeyFieldType** - Result output key field types

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

File Operations
~~~~~~~~~~~~~~~

**FileFormat** - Import/export file formats

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

**ObjectIDHandling** - Contingency export object ID modes

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``NO``
     - Standard object references
   * - ``YES_MS_3W``
     - Include multi-section and 3-winding IDs

Topology & Network
~~~~~~~~~~~~~~~~~~

**BranchDistanceMeasure** - Distance metrics for topology analysis

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``REACTANCE``
     - Use reactance (X) as distance measure
   * - ``IMPEDANCE``
     - Use impedance magnitude (Z) as distance measure

**BranchFilterMode** - Branch filter modes for topology traversal

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``ALL``
     - All branches
   * - ``SELECTED``
     - Only selected branches
   * - ``CLOSED``
     - Only closed branches

**IslandReference** - Island reference options

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``EXISTING``
     - Use existing island configuration
   * - ``NO``
     - No area reference

Modification & Scaling
~~~~~~~~~~~~~~~~~~~~~~

**ScalingBasis** - Load/generation scaling basis

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``MW``
     - Absolute MW/MVAR values
   * - ``FACTOR``
     - Multiplier factor

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

Case Operations
~~~~~~~~~~~~~~~

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

Weather & Ratings
~~~~~~~~~~~~~~~~~

**RatingSetPrecedence** - Rating set precedence for weather-based ratings

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``NORMAL``
     - Use normal rating set
   * - ``CTG``
     - Use contingency rating set

**RatingSet** - Rating set identifiers (A-O, DEFAULT, NO)

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``DEFAULT``
     - Use default rating
   * - ``NO``
     - Don't update rating
   * - ``A`` - ``O``
     - Rating sets A through O

Transient Stability
~~~~~~~~~~~~~~~~~~~

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

Exception classes for handling PowerWorld and COM errors.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Exception
     - Description
   * - ``Error``
     - Base class for all ESA++ exceptions
   * - ``PowerWorldError``
     - Generic error from PowerWorld following a SimAuto call. Parses error messages to extract source and details.
   * - ``SimAutoFeatureError``
     - Raised when a SimAuto feature is not supported for the given object or context (e.g., object types that don't support ``GetParameters``)
   * - ``PowerWorldPrerequisiteError``
     - Raised when a command fails due to missing prerequisite data (e.g., no contingencies defined for ``CTGSolve``)
   * - ``PowerWorldAddonError``
     - Raised when a command requires an unlicensed PowerWorld add-on (e.g., TransLineCalc)
   * - ``COMError``
     - Raised when COM communication fails (SimAuto crash, unresponsive, or invalid function call)
   * - ``CommandNotRespectedError``
     - Raised when PowerWorld silently ignores a command (e.g., setting a value outside allowed limits)

Exception Hierarchy
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Exception
    └── Error (base for all ESA++ exceptions)
        ├── COMError
        └── PowerWorldError
            ├── SimAutoFeatureError
            ├── PowerWorldPrerequisiteError
            ├── PowerWorldAddonError
            └── CommandNotRespectedError

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError

    try:
        saw.CTGSolveAll()
    except PowerWorldPrerequisiteError:
        print("No contingencies defined - add contingencies first")
    except PowerWorldError as e:
        print(f"PowerWorld error: {e.message}")
        print(f"Source: {e.source}")
