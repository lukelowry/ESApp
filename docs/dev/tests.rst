Testing Suite
=============

One suite covers everything: fast unit tests that run without PowerWorld and integration tests that exercise
live Simulator cases. Configure once, run anywhere.

Test Map
--------

**Unit Tests** (No PowerWorld required)

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Coverage
   * - test_gobject.py
     - GObject base class, FieldPriority flags, repr/str methods
   * - test_grid_components.py
     - Field collection, key/editable/settable classification, all generated components
   * - test_indexing.py
     - Indexable class data access (``wb[GObject, "field"]`` syntax), broadcast, bulk update
   * - test_helpers_unit.py
     - SAW helper functions: df_to_aux, path conversion, formatting utilities
   * - test_dynamics.py
     - Dynamics module: ContingencyBuilder, SimAction enum (mocked)

**Integration Tests** (Requires PowerWorld)

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Coverage
   * - test_integration_saw_powerworld.py
     - Core SAW operations, file I/O, data retrieval against live cases
   * - test_integration_workbench.py
     - GridWorkBench data access, indexing on live case
   * - test_integration_powerflow.py
     - Power flow solutions, matrices (Ybus, Jacobian), PTDF/LODF sensitivity
   * - test_integration_contingency.py
     - Contingency auto-insertion, solving, cloning, OTDF calculations
   * - test_integration_analysis.py
     - GIC analysis, ATC analysis, transient stability, time step simulation
   * - test_integration_extended.py
     - Scheduled actions, weather, oneline, OPF, extended method coverage

Configure Integration Tests
---------------------------

1. Copy ``tests/config_test.example.py`` to ``tests/config_test.py``
2. Set an absolute path to a PowerWorld case: ``SAW_TEST_CASE = r"C:\Path\To\Your\Case.pwb"``
3. Keep the file alongside the tests; pytest will auto-detect it

Running Tests
-------------

.. code-block:: bash

    # Full suite
    pytest tests/

    # Unit only (skip PowerWorld)
    pytest tests/ -m "not integration"

    # Integration only
    pytest tests/ -m integration

    # Specific file
    pytest tests/test_grid_components.py -v

    # With coverage
    pytest tests/ --cov=esapp --cov-report=html

VS Code
-------

Open the Testing view (beaker icon); tests are discovered automatically. Run by file or class, and
debug individual tests from the UI.

Troubleshooting
---------------

- **PowerWorld not found**: Ensure ``tests/config_test.py`` exists with a valid case path
- **Integration tests slow**: Run ``pytest -m "not integration"`` for unit-only
- **Import errors**: Install in editable mode with ``pip install -e .``
