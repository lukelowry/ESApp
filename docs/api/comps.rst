Objects & Fields
================

The ``esapp.components`` module provides the base classes for defining grid component schemas
and transient stability field constants.

GObject Base Class
------------------

.. automodule:: esapp.components.gobject
   :members:

Transient Stability Fields
--------------------------

The ``TS`` class provides IDE intellisense for transient stability result fields, organized
by object type. Use these constants with ``TSWatch.watch()`` to specify which fields to record
during simulation.

.. code-block:: python

    from esapp import TS
    from esapp.components import Gen, Bus
    from esapp.utils import TSWatch

    tsw = TSWatch()

    # Watch generator fields
    tsw.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])

    # Watch bus fields
    tsw.watch(Bus, [TS.Bus.VPU, TS.Bus.Freq])

TS Field Reference
------------------

The following tables list all available transient stability field constants by category.
Access fields using ``TS.<Category>.<Field>`` syntax (e.g., ``TS.Gen.P``, ``TS.Bus.VPU``).

.. ts-field-list::

Available Grid Object Types
----------------------------

The following component types are available in ``esapp.components``.
Each class represents a PowerWorld object type that can be used with
the :class:`~esapp.PowerWorld` indexing syntax (e.g., ``pw[Bus, "BusNum"]``).

.. grid-component-list::
