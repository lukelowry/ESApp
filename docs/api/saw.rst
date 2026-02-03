SimAuto Wrapper (SAW)
=====================

The ``SAW`` (SimAuto Wrapper) class provides complete access to PowerWorld's SimAuto API.
It is organized into functional mixins for power flow, contingencies, optimization, sensitivity,
transient stability, GIC, ATC, topology, and data management. Access SAW through ``pw.esa``
from ``PowerWorld``.

.. currentmodule:: esapp.saw

.. autoclass:: SAW
   :show-inheritance:
   :no-members:

General Program Actions
-----------------------

.. autoclass:: esapp.saw.base.SAWBase
   :members:
   :noindex:

.. autoclass:: esapp.saw.general.GeneralMixin
   :members:
   :noindex:

Data Interaction
----------------

.. autoclass:: esapp.saw.data.DataMixin
   :members:
   :noindex:

Case Actions
------------

.. autoclass:: esapp.saw.case_actions.CaseActionsMixin
   :members:
   :noindex:

Modify Case Objects
-------------------

.. autoclass:: esapp.saw.modify.ModifyMixin
   :members:
   :noindex:

.. autoclass:: esapp.saw.topology.TopologyMixin
   :members:
   :noindex:

Power Flow
----------

.. autoclass:: esapp.saw.powerflow.PowerflowMixin
   :members:
   :noindex:

.. autoclass:: esapp.saw.matrices.MatrixMixin
   :members:
   :noindex:

Sensitivity Calculations
------------------------

.. autoclass:: esapp.saw.sensitivity.SensitivityMixin
   :members:
   :noindex:

Contingency Analysis
--------------------

.. autoclass:: esapp.saw.contingency.ContingencyMixin
   :members:
   :noindex:

Fault Analysis
--------------

.. autoclass:: esapp.saw.fault.FaultMixin
   :members:
   :noindex:

ATC (Available Transfer Capability)
------------------------------------

.. autoclass:: esapp.saw.atc.ATCMixin
   :members:
   :noindex:

GIC (Geomagnetically Induced Current)
--------------------------------------

.. autoclass:: esapp.saw.gic.GICMixin
   :members:
   :noindex:

OPF (Optimal Power Flow) and SCOPF
-----------------------------------

.. autoclass:: esapp.saw.opf.OPFMixin
   :members:
   :noindex:

PV Analysis
-----------

.. autoclass:: esapp.saw.pv.PVMixin
   :members:
   :noindex:

QV Analysis
-----------

.. autoclass:: esapp.saw.qv.QVMixin
   :members:
   :noindex:

Regions
-------

.. autoclass:: esapp.saw.regions.RegionsMixin
   :members:
   :noindex:

TS (Transient Stability)
------------------------

.. autoclass:: esapp.saw.transient.TransientMixin
   :members:
   :noindex:

Scheduled Actions
-----------------

.. autoclass:: esapp.saw.scheduled.ScheduledActionsMixin
   :members:
   :noindex:

Time Step Simulation
--------------------

.. autoclass:: esapp.saw.timestep.TimeStepMixin
   :members:
   :noindex:

Weather
-------

.. autoclass:: esapp.saw.weather.WeatherMixin
   :members:
   :noindex:

Type-Safe Enumerations
----------------------

ESA++ provides enumeration types for common PowerWorld string parameters. Using these
enums instead of raw strings provides IDE autocomplete, type checking, and prevents typos.

.. code-block:: python

    from esapp.saw import SolverMethod, FilterKeyword, LinearMethod

    # Use enums for type-safe parameters
    saw.SolvePowerFlow(SolverMethod.RECTNEWT)
    saw.GetParametersMultipleElement("Bus", ["BusNum", "BusPUVolt"], FilterKeyword.ALL)

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

**FieldListColumn** - Column names for ``GetFieldList`` results

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``KEY_FIELD``
     - Whether the field is a key field
   * - ``INTERNAL_FIELD_NAME``
     - PowerWorld internal field name
   * - ``FIELD_DATA_TYPE``
     - Data type of the field
   * - ``DESCRIPTION``
     - Human-readable description
   * - ``DISPLAY_NAME``
     - Display name in PowerWorld UI
   * - ``ENTERABLE``
     - Whether the field can be edited

**SpecificFieldListColumn** - Column names for ``GetSpecificFieldList`` results

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``VARIABLENAME_LOCATION``
     - Variable name with location
   * - ``FIELD``
     - Field identifier
   * - ``COLUMN_HEADER``
     - Column header label
   * - ``FIELD_DESCRIPTION``
     - Human-readable description
   * - ``ENTERABLE``
     - Whether the field can be edited

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
----------------

**Filter formatting:**

.. autofunction:: esapp.saw.format_filter

.. autofunction:: esapp.saw.format_filter_selected_only

.. autofunction:: esapp.saw.format_filter_areazone

**Data conversion:**

.. autofunction:: esapp.saw.df_to_aux

.. autofunction:: esapp.saw.create_object_string

.. autofunction:: esapp.saw.convert_to_windows_path

.. autofunction:: esapp.saw.convert_list_to_variant

.. autofunction:: esapp.saw.convert_df_to_variant

.. autofunction:: esapp.saw.convert_nested_list_to_variant

.. autofunction:: esapp.saw.get_temp_filepath

.. autofunction:: esapp.saw.format_list

.. autofunction:: esapp.saw.format_optional

.. autofunction:: esapp.saw.format_optional_numeric

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
   * - ``GridObjDNE``
     - Raised when a grid object data query fails (object does not exist in the case)
   * - ``FieldDataException``
     - Raised when there is an issue with field data retrieval or parsing
   * - ``AuxParseException``
     - Raised when parsing an auxiliary file fails
   * - ``ContainerDeletedException``
     - Raised when attempting to access a container that has been deleted
   * - ``PowerFlowException``
     - Base class for power flow solution errors
   * - ``BifurcationException``
     - Raised when voltage bifurcation is suspected during power flow
   * - ``DivergenceException``
     - Raised when the power flow solution diverges
   * - ``GeneratorLimitException``
     - Raised when a generator has exceeded a limit during power flow
   * - ``GICException``
     - Raised when a GIC analysis error occurs

.. code-block:: text

    Exception
    └── Error (base for all ESA++ exceptions)
        ├── COMError
        ├── GridObjDNE
        ├── FieldDataException
        ├── AuxParseException
        ├── ContainerDeletedException
        ├── GICException
        ├── PowerFlowException
        │   ├── BifurcationException
        │   ├── DivergenceException
        │   └── GeneratorLimitException
        └── PowerWorldError
            ├── SimAutoFeatureError
            ├── PowerWorldPrerequisiteError
            ├── PowerWorldAddonError
            └── CommandNotRespectedError

.. code-block:: python

    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError

    try:
        saw.CTGSolveAll()
    except PowerWorldPrerequisiteError:
        print("No contingencies defined - add contingencies first")
    except PowerWorldError as e:
        print(f"PowerWorld error: {e.message}")
        print(f"Source: {e.source}")
