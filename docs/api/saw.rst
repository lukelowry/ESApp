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

Power flow solution algorithms.

.. autoclass:: esapp.saw.SolverMethod
   :members:
   :undoc-members:

LinearMethod
~~~~~~~~~~~~

Linear calculation methods for sensitivity analysis (PTDF, LODF, etc.).

.. autoclass:: esapp.saw.LinearMethod
   :members:
   :undoc-members:

FilterKeyword
~~~~~~~~~~~~~

Special filter keywords passed unquoted to PowerWorld commands.

.. autoclass:: esapp.saw.FilterKeyword
   :members:
   :undoc-members:

YesNo
~~~~~

Boolean flag values for PowerWorld commands that use "YES"/"NO" strings.

.. autoclass:: esapp.saw.YesNo
   :members:
   :undoc-members:

FileFormat
~~~~~~~~~~

File format types for import/export operations.

.. autoclass:: esapp.saw.FileFormat
   :members:
   :undoc-members:

ObjectType
~~~~~~~~~~

PowerWorld object type identifiers for filtering and operations.

.. autoclass:: esapp.saw.ObjectType
   :members:
   :undoc-members:

KeyFieldType
~~~~~~~~~~~~

Key field types for result output formatting.

.. autoclass:: esapp.saw.KeyFieldType
   :members:
   :undoc-members:

IslandReference
~~~~~~~~~~~~~~~

Island reference options for sensitivity analysis.

.. autoclass:: esapp.saw.IslandReference
   :members:
   :undoc-members:

Other Enumerations
~~~~~~~~~~~~~~~~~~

Additional specialized enumerations:

.. autoclass:: esapp.saw.InterfaceLimitSetting
   :members:
   :undoc-members:

.. autoclass:: esapp.saw.StarBusHandling
   :members:
   :undoc-members:

.. autoclass:: esapp.saw.MultiSectionLineHandling
   :members:
   :undoc-members:

.. autoclass:: esapp.saw.OnelineLinkMode
   :members:
   :undoc-members:

.. autoclass:: esapp.saw.ShuntModel
   :members:
   :undoc-members:

.. autoclass:: esapp.saw.BranchDeviceType
   :members:
   :undoc-members:

.. autoclass:: esapp.saw.TSGetResultsMode
   :members:
   :undoc-members:

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
