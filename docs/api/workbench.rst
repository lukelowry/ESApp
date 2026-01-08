GridWorkBench
=============

The ``GridWorkBench`` is the central orchestrator of the ESA++ toolkit. It manages the lifecycle of the 
PowerWorld Simulator instance, handles case loading/saving, and provides the primary interface for 
data access (via ``IndexTool``) and analysis (via ``Adapter`` and ``Apps``).

Sugar Functions
==================

The ``Adapter`` (accessed via ``wb.func``) provides a high-level, Pythonic interface to complex PowerWorld operations. 
It abstracts away the verbose SimAuto script syntax into clean, one-liner methods for common engineering tasks 
like contingency analysis, fault studies, and system scaling.

.. currentmodule:: gridwb

.. autoclass:: gridwb.GridWorkBench
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __getitem__, __setitem__

.. automodule:: gridwb.indextool
   :members:
   :special-members: __getitem__, __setitem__