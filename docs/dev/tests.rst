Testing Suite
=============

ESA++ maintains a comprehensive testing suite with both unit tests (using mocks) and integration tests 
(connecting to real PowerWorld instances). This ensures reliability and compatibility across different 
scenarios and PowerWorld versions.

Test Organization
-----------------

The test suite is divided into two categories:

**Unit Tests** (No PowerWorld Required)
  These tests use mocked dependencies and can run without PowerWorld Simulator installed. They verify:
  
  - Component class generation and structure
  - Data access interface (indexing syntax)
  - Exception handling and error messages
  - Core library functionality
  
  Fast to run and ideal for continuous integration pipelines.

**Integration Tests** (Requires PowerWorld)
  These tests connect to a live PowerWorld Simulator instance and validate:
  
  - Real PowerWorld case loading and manipulation
  - Power flow solutions and convergence
  - Contingency analysis results
  - Data retrieval accuracy
  - File I/O operations
  
  Slower but provide the most comprehensive validation.

Test Files
----------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - File
     - Purpose
     - Requires PowerWorld?
   * - test_grid_components.py
     - Component class generation, field definitions, and GObject metaclass
     - No
   * - test_exceptions.py
     - Custom exception hierarchy and error handling patterns
     - No
   * - test_indexable_data_access.py
     - Data retrieval and modification via indexing syntax
     - No
   * - test_saw_core_methods.py
     - SAW class methods and COM interface wrapping
     - No
   * - test_integration_saw_powerworld.py
     - Real PowerWorld interactions: power flow, contingencies, file ops
     - **Yes**
   * - test_integration_workbench.py
     - GridWorkBench component access with live PowerWorld data
     - **Yes**

Unit Tests
----------

**test_grid_components.py**
  Tests the component system that enables type-safe access to PowerWorld objects:
  
  - GObject metaclass functionality
  - Field priority flag combinations
  - Component class generation
  - Field inheritance and override behavior
  - Docstring availability
  - Field type validation
  
  Example test:

  .. code-block:: python
  
      def test_gobject_fields_are_collected(test_gobject_class):
          """Verify that fields are correctly collected from class definition"""
          assert hasattr(test_gobject_class, '_fields')
          assert len(test_gobject_class._fields) > 0

**test_exceptions.py**
  Tests error handling when PowerWorld encounters problems:
  
  - PowerWorldError exception hierarchy
  - COMError for interface problems
  - CommandNotRespectedError for rejected operations
  - SimAutoFeatureError for unsupported features
  - Custom error message formatting
  
**test_indexable_data_access.py**
  Tests the core data access interface:
  
  - Reading single and multiple fields
  - Retrieving all fields with ``:`` operator
  - Bulk data retrieval into DataFrames
  - Broadcasting scalar values to all components
  - Bulk updates from DataFrames
  - DataFrame indexing and filtering integration

**test_saw_core_methods.py**
  Tests low-level SAW functionality without real PowerWorld:
  
  - Case file operations (open, save)
  - Parameter setting and retrieval
  - Device enumeration
  - Script execution
  - COM error handling

Integration Tests
-----------------

**test_integration_saw_powerworld.py**
  Validates SAW methods against a real PowerWorld case:
  
  - Case loading and saving
  - Power flow solutions with different algorithms
  - Convergence verification
  - Parameter reading and modification
  - Device enumeration (buses, branches, etc.)
  - Export operations (CSV, Excel)
  - Contingency analysis
  - Field list validation
  
  Example test:

  .. code-block:: python
  
      def test_powerflow_solve(self, saw_instance):
          """Verify power flow solution converges"""
          success = saw_instance.SolvePowerFlow()
          assert success

**test_integration_workbench.py**
  Tests GridWorkBench functionality with real PowerWorld:
  
  - Case file loading
  - Component data retrieval (buses, generators, loads)
  - Data modification and saving
  - Fixture setup for component access

Setup for PowerWorld Tests
---------------------------

To run integration tests with PowerWorld:

1. **Copy Configuration Template**
   
   .. code-block:: bash
   
       copy tests/config_test.example.py tests/config_test.py

2. **Edit Configuration**
   
   Open ``tests/config_test.py`` and set the case file path:
   
   .. code-block:: python
   
       SAW_TEST_CASE = r"C:\Path\To\Your\Case.pwb"

3. **Run Tests**
   
   Tests will automatically detect and use the configured case file.

Running Tests
-------------

**Using pytest**

Run all tests:

.. code-block:: bash

    pytest tests/

Run only unit tests (no PowerWorld needed):

.. code-block:: bash

    pytest tests/ -m "not online"

Run only a specific test file:

.. code-block:: bash

    pytest tests/test_grid_components.py -v

Run with coverage report:

.. code-block:: bash

    pytest tests/ --cov=esapp --cov-report=html
    # Open htmlcov/index.html

**Using VS Code**

1. Install Python extension
2. Open Testing view (beaker icon in left sidebar)
3. Tests appear automatically
4. Click to run individual tests or entire files
5. Green checkmarks = passing, red X's = failing

**Fixtures and Configuration**

Common fixtures are defined in ``conftest.py``:

- ``saw_instance``: Live PowerWorld connection for integration tests
- ``temp_file``: Temporary file for I/O testing
- ``test_gobject_class``: Sample component class for unit tests

Test Coverage Goals
~~~~~~~~~~~~~~~~~~~

- **Core API**: 95%+ coverage of GridWorkBench and Indexable classes
- **Components**: All auto-generated component classes validated
- **SAW Mixins**: Coverage of major functionality in each mixin module
- **Error Handling**: All custom exceptions tested
- **Integration**: Real PowerWorld scenarios validated

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

The test suite is designed for automated CI/CD pipelines:

- Unit tests run on every commit
- Integration tests run on-demand with PowerWorld
- Coverage reports tracked over time
- Test failures block pull requests until resolved

Troubleshooting
~~~~~~~~~~~~~~~

**PowerWorld not found during tests:**
  Ensure ``tests/config_test.py`` exists with valid case file path

**Online tests skipped in VS Code:**
  Make sure ``tests/config_test.py`` is in the tests directory

**Mock errors during unit tests:**
  Check that mock objects in ``conftest.py`` are properly configured

**Slow test execution:**
  Skip integration tests with ``pytest -m "not online"``

**Import errors:**
  Install in development mode: ``pip install -e .``