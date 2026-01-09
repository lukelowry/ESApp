Testing Suite
=============

ESA++ maintains a comprehensive testing suite with both unit tests (using mocks) and integration tests 
(connecting to real PowerWorld instances).

Test Organization
-----------------

**Unit Tests** (No PowerWorld Required)
  Fast tests using mocked dependencies. Verify component generation, data access, exceptions, and core functionality.

**Integration Tests** (Requires PowerWorld)
  Validate against live PowerWorld: case loading, power flow, contingency analysis, and data operations.

Test Files
----------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - File
     - Purpose
     - Requires PowerWorld?
   * - test_grid_components.py
     - Component class generation and GObject metaclass
     - No
   * - test_exceptions.py
     - Exception hierarchy and error handling
     - No
   * - test_indexable_data_access.py
     - Data retrieval/modification via indexing
     - No
   * - test_saw_core_methods.py
     - SAW methods with mocked COM interface
     - No
   * - test_integration_saw_powerworld.py
     - Real PowerWorld power flow and analysis
     - **Yes**
   * - test_integration_workbench.py
     - GridWorkBench with live PowerWorld data
     - **Yes**

Setup for Integration Tests
----------------------------

To run tests requiring PowerWorld:

1. Copy ``tests/config_test.example.py`` to ``tests/config_test.py``
2. Edit and set: ``SAW_TEST_CASE = r"C:\Path\To\Your\Case.pwb"``
3. Run tests normally - they auto-detect the configuration

Running Tests
-------------

.. code-block:: bash

    # All tests
    pytest tests/

    # Unit tests only (no PowerWorld)
    pytest tests/ -m "not online"

    # Specific file
    pytest tests/test_grid_components.py -v

    # With coverage
    pytest tests/ --cov=esapp --cov-report=html

**VS Code:** Open Testing view (beaker icon), tests appear automatically.

Troubleshooting
---------------

- **PowerWorld not found:** Verify ``tests/config_test.py`` exists with valid case path
- **Slow tests:** Use ``pytest -m "not online"`` to skip integration tests
- **Import errors:** Install with ``pip install -e .``