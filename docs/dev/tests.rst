Testing Suite
=============

One suite covers everything: fast unit tests that run without PowerWorld and integration tests that exercise
live Simulator cases. Configure once, run anywhere.

Test map
--------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - File
     - What it covers
     - PowerWorld?
   * - test_grid_components.py
     - Component definitions, field metadata, GObject behavior
     - No
   * - test_exceptions.py
     - Exception hierarchy and messaging
     - No
   * - test_indexable_data_access.py
     - Indexing reads/writes on mock data
     - No
   * - test_saw_core_methods.py
     - SAW core calls with mocked COM responses
     - No
   * - test_integration_saw_powerworld.py
     - Power flow, contingencies, file ops against real cases
     - **Yes**
   * - test_integration_workbench.py
     - GridWorkBench data access on a live case
     - **Yes**

Configure integration tests (one-time)
--------------------------------------

1. Copy ``tests/config_test.example.py`` to ``tests/config_test.py``
2. Set an absolute path to a PowerWorld case: ``SAW_TEST_CASE = r"C:\Path\To\Your\Case.pwb"``
3. Keep the file alongside the tests; pytest will auto-detect it

How to run
----------

.. code-block:: bash

    # Full suite
    pytest tests/

    # Unit only (skip PowerWorld)
    pytest tests/ -m "not online"

    # Specific file
    pytest tests/test_grid_components.py -v

    # With coverage
    pytest tests/ --cov=esapp --cov-report=html

VS Code
-------

Open the Testing view (beaker icon); tests are discovered automatically. You can run by file or class, and
debug individual tests from the UI.

Troubleshooting
---------------

- PowerWorld not found: ensure ``tests/config_test.py`` exists and the path is correct
- Online tests slow: run ``pytest -m "not online"`` for unit-only
- Import errors: install in editable mode ``pip install -e .``