# ESA++ Test Suite

Test coverage for the ESA++ library. Tests verify that the library correctly interacts with PowerWorld Simulator and handles data properly.

## Test Files Overview

| Test File | What It Tests | Needs PowerWorld? |
|-----------|---------------|-------------------|
| **test_grid_components.py** | Component definitions (Bus, Gen, Load, etc.) | No |
| **test_exceptions.py** | Error handling and messages | No |
| **test_indexable_data_access.py** | Reading/writing data | No |
| **test_saw_core_methods.py** | Core library methods | No |
| **test_integration_saw_powerworld.py** | Power flow, contingency analysis with real cases | **Yes** |
| **test_integration_workbench.py** | Component access with real case data | **Yes** |

## Setup for PowerWorld Tests

To run tests that connect to PowerWorld:

1. Copy `config_test.example.py` to `config_test.py` in the tests folder
2. Edit `config_test.py` and set the path to your PowerWorld case:
   ```python
   SAW_TEST_CASE = r"C:\Path\To\Your\Case.pwb"
   ```
3. Tests will automatically use this case file when running in your IDE

## Test Files Overview

### Unit Tests (No PowerWorld Required)
These tests use mocked dependencies and can run without PowerWorld Simulator:

- **test_grid_components.py** - Tests GObject metaclass, FieldPriority flags, and component class generation
- **test_exceptions.py** - Tests custom exception hierarchy and error handling patterns
- **test_indexable_data_access.py** - Tests Indexable class `__getitem__` and `__setitem__` methods for data I/O
- **test_saw_core_methods.py** - Tests SAW class methods with mocked COM interface

### Integration Tests (Requires PowerWorld)
These tests connect to a live PowerWorld Simulator instance:

- **test_integration_saw_powerworld.py** - Validates SAW methods against a real PowerWorld case (power flow, contingency analysis, file operations, etc.)
- **test_integration_workbench.py** - Tests GridWorkBench component access and manipulation with live PowerWorld data

### Support Files
- **conftest.py** - Shared pytest fixtures and configuration
- **config_test.py** - Integration test configuration (case file path)
- **run_all_tests.py** - Script to run all tests with various options

## What Each Test File Tests

### test_grid_components.py
Tests the component system that represents PowerWorld objects (buses, generators, loads, branches, etc.). Validates that all component types are properly defined with correct field names and data types.

### test_exceptions.py
Tests error handling when things go wrong - verifies that meaningful error messages are shown when PowerWorld encounters problems or when invalid data is provided.

### test_indexable_data_access.py
Tests reading and writing data to/from PowerWorld components. For example, reading bus voltages or updating generator MW output.

### test_saw_core_methods.py
Tests core library functions like opening cases, running power flow solutions, and executing PowerWorld commands - but without actually running PowerWorld (uses simulated responses).

### test_integration_saw_powerworld.py *(Requires PowerWorld)*
Runs actual PowerWorld analyses with your case file:
- Power flow solutions
- Contingency analysis
- Data export to various formats
- Real data retrieval from your case

### test_integration_workbench.py *(Requires PowerWorld)*
Tests accessing and reading component data from a real PowerWorld case, validating that you can retrieve bus, generator, load, and branch information correctly.

## Running Tests in VS Code

If you're using VS Code (recommended):

1. Install the Python extension
2. Open the Testing view (beaker icon in left sidebar)
3. Tests will appear automatically - click to run individual tests or entire files
4. Green checkmarks = passing, red X's = failing

You can run tests without PowerWorld first to verify the library basics, then configure your case file to run the full integration tests.

## Support Files

- **conftest.py** - Shared test setup and fixtures (handles PowerWorld connections automatically)
- **config_test.py** - Your PowerWorld case path configuration
- **run_all_tests.py** - Optional script to run all tests at once

```bash
pytest --cov=esapp --cov-report=html
# Open htmlcov/index.html
```

## Troubleshooting

- **PowerWorld not found**: Configure case path in `tests/config_test.py`
- **Online tests skipped in VS Code**: Make sure `tests/config_test.py` exists with valid path
- **Mock errors**: Check conftest.py for proper mock configuration
- **Slow tests**: Skip with `-m "not slow"`
- **Import errors**: Install with `pip install -e .`
