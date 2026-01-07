GridWorkBench
=============

The ``GridWorkBench`` is the central orchestrator of the ESA++ toolkit. It manages the lifecycle of the 
PowerWorld Simulator instance, handles case loading/saving, and provides the primary interface for 
data access (via ``IndexTool``) and analysis (via ``Adapter`` and ``Apps``).

.. currentmodule:: gridwb.workbench

.. autoclass:: gridwb.workbench.GridWorkBench
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __getitem__, __setitem__

.. automodule:: gridwb.indextool
   :members:
   :special-members: __getitem__, __setitem__