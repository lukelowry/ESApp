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
