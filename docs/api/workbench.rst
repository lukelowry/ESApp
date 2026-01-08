GridWorkBench
=============
.. currentmodule:: esapp.workbench

The ``GridWorkBench`` is the central orchestrator of the ESA++ toolkit. It manages the lifecycle of the 
PowerWorld Simulator instance, handles case loading/saving, and provides the primary interface for 
data access (via ``Indexable``) and analysis .

The ``Indexable`` is the core engine of ESA++. It enables the intuitive indexing syntax (e.g., ``wb[Bus, :]``) 
by translating Pythonic slices and keys into optimized SimAuto data requests. It handles both data retrieval 
and bulk updates, returning results as native Pandas DataFrames.

.. autoclass:: GridWorkBench
   :members:
