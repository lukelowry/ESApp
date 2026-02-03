PowerWorld
=============

The ``PowerWorld`` is the high-level entry point for interacting with PowerWorld via ESA++. It wraps
SimAuto with a Pythonic interface for case management, data access, and analysis helpers.

.. currentmodule:: esapp.workbench

.. autoclass:: PowerWorld
   :members:

Descriptors
-----------

Lightweight descriptor classes that map Python attributes to PowerWorld option fields.
Used by ``PowerWorld`` for solver options and by ``GIC`` for GIC analysis options.

.. currentmodule:: esapp._descriptors

.. autoclass:: SolverOption
   :members:

.. autoclass:: GICOption
   :members:
