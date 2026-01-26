# ESA++ Test Suite

**Coverage: 85.74%** (unit tests only, excluding integration)

## Quick Start

```bash
pytest                         # Run all tests with coverage
pytest --no-cov                # Skip coverage reporting
pytest -k "not integration"    # Unit tests only (no PowerWorld)
pytest -m "not slow"           # Skip slow tests
```

**PowerWorld Setup**: Copy `config_test.example.py` → `config_test.py`, set `SAW_TEST_CASE` path.

## Test Organization

| Category | Files | Purpose |
|----------|-------|----------|
| **Unit Tests** | `test_exceptions.py`<br>`test_saw_core_methods.py`<br>`test_workbench.py` | Mock-based tests, no PowerWorld required |
| **Integration** | `test_integration_*.py` | Real PowerWorld case testing |
| **Component** | `test_grid_components.py`<br>`test_indexable_data_access.py` | Data access & grid definitions |
| **Apps** | `test_apps_network_gic.py` | High-level application testing |

> **Note**: `test_grid_components.py` generates ~3,800 parametrized tests validating 958 auto-generated component classes.

## Recent Changes (2026-01-25)

### Test Consolidation
- **Merged** `test_workbench.py` + `test_workbench_unit.py` → `test_workbench.py`
  - Reduced from 791 lines to 529 lines
  - Eliminated duplicate tests
  - Better organized with clear test class structure
  - 51+ comprehensive tests covering all workbench functionality

### Coverage Improvements
- **Overall coverage**: 85.32% → 85.74% (+0.42%)
- **Workbench coverage**: 59.35% → 62.58% (+3.23%)
- **Test count**: 3,574 tests passing (2 minor failures to fix)

## Coverage by Module

| Module | Coverage | Status |
|--------|----------|--------|
| `case_actions.py` | 100.00% | ✅ Fully tested |
| `modify.py` | 100.00% | ✅ Fully tested |
| `opf.py` | 100.00% | ✅ Fully tested |
| `pv.py` | 100.00% | ✅ Fully tested |
| `regions.py` | 100.00% | ✅ Fully tested |
| `saw.py` | 100.00% | ✅ Fully tested |
| `scheduled.py` | 100.00% | ✅ Fully tested |
| `sensitivity.py` | 100.00% | ✅ Fully tested |
| `weather.py` | 100.00% | ✅ Fully tested |
| `gobject.py` | 97.80% | ✅ Well tested |
| `contingency.py` | 97.30% | ✅ Well tested |
| `gic.py` | 97.67% | ✅ Well tested |
| `general.py` | 98.46% | ✅ Well tested |
| `timestep.py` | 97.14% | ✅ Well tested |
| `oneline.py` | 94.44% | ✅ Well tested |
| `qv.py` | 94.44% | ✅ Well tested |
| `powerflow.py` | 93.48% | ✅ Well tested |
| `atc.py` | 91.89% | ✅ Well tested |
| `topology.py` | 90.32% | ✅ Well tested |
| `transient.py` | 89.01% | ⚠️ CCT, results extraction |
| `matrices.py` | 84.62% | ⚠️ Matrix decomposition paths |
| `fault.py` | 81.25% | ⚠️ Fault calculation edge cases |
| `indexable.py` | 78.01% | ⚠️ Edge cases, complex filters |
| `base.py` | 69.11% | 🔴 Error handling paths |
| `workbench.py` | 62.58% | 🔴 Property accessors, advanced methods |

**Intentionally Excluded**:
- `grid.py` — Auto-generated (175k+ lines)
- `apps/static.py`, `apps/dynamics.py` — Research code
- `utils/*` — Specialized data processing tools

## Priority Coverage Gaps

### High Priority (Core Functionality)
1. **workbench.py** (62.58%) - Missing:
   - Property accessors (voltages_kv, generations, loads, shunts, lines, transformers, areas, zones)
   - Advanced topology methods (state chain, dispatch management)
   - Diff flow operations

2. **base.py** (69.11%) - Missing:
   - Error handling and recovery paths
   - Complex parameter validation
   - Edge cases in data transformation

### Medium Priority
3. **indexable.py** (78.01%) - Missing:
   - Complex field selection edge cases
   - Error conditions in __getitem__ and __setitem__

4. **transient.py** (89.01%) - Missing:
   - Critical clearing time (CCT) calculations
   - Results extraction methods

## Test File Structure

```
tests/
├── conftest.py                        # Shared fixtures and utilities
├── config_test.py                     # User configuration (not in git)
├── config_test.example.py             # Configuration template
│
├── test_exceptions.py                 # Exception hierarchy tests (376 lines)
├── test_workbench.py                  # Workbench comprehensive tests (529 lines)
├── test_saw_core_methods.py           # SAW mixin tests (2978 lines) ⚠️ Large
├── test_grid_components.py            # Grid component tests (253 lines)
├── test_indexable_data_access.py      # Indexable tests (578 lines)
├── test_apps_network_gic.py           # Network/GIC app tests (247 lines)
│
├── test_integration_powerflow.py      # Power flow integration (299 lines)
├── test_integration_contingency.py    # Contingency integration (323 lines)
├── test_integration_analysis.py       # Analysis integration (290 lines)
├── test_integration_saw_powerworld.py # SAW/PW integration (331 lines)
└── test_integration_workbench.py      # Workbench integration (336 lines)
```

> ⚠️ **Note**: `test_saw_core_methods.py` is very large (2978 lines, 37 test classes). Consider splitting into:
> - `test_saw_base.py` - Base SAW functionality
> - `test_saw_powerflow.py` - Power flow mixin tests
> - `test_saw_contingency.py` - Contingency mixin tests
> - `test_saw_analysis.py` - Analysis/sensitivity mixin tests
> - `test_saw_helpers.py` - Helper functions

## Running Tests

### By Category
```bash
# Unit tests only (fast, no PowerWorld)
pytest -m unit

# Integration tests only (requires PowerWorld)
pytest -m integration

# Specific module
pytest tests/test_workbench.py -v

# Specific test class
pytest tests/test_workbench.py::TestPowerFlowOperations -v

# Specific test
pytest tests/test_workbench.py::TestPowerFlowOperations::test_pflow_calls_solve -v
```

### With Coverage
```bash
# Full coverage report
pytest --cov=esapp --cov-report=html

# Specific module coverage
pytest --cov=esapp.workbench --cov-report=term-missing

# Show only uncovered lines
pytest --cov=esapp --cov-report=term-missing:skip-covered
```

### Performance
```bash
# Run tests in parallel (faster)
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# Run only fast unit tests
pytest -k "not integration" -m "not slow"
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| PowerWorld not found | Set path in `config_test.py` or environment variable `SAW_TEST_CASE` |
| Import errors | Run `pip install -e .` from repository root |
| Slow integration tests | Use `pytest -k "not integration"` or `pytest -m "not slow"` |
| Coverage report not found | Run `pytest` first, then open `htmlcov/index.html` |
| Tests fail with numpy warning | Normal, tests still pass - numpy version compatibility warning |
| COM errors in tests | Ensure mocks are properly configured in conftest.py |

## Writing New Tests

### Test Organization Guidelines

1. **File naming**:
   - Unit tests: `test_<module>.py`
   - Integration tests: `test_integration_<feature>.py`

2. **Test class naming**:
   - Use descriptive names: `TestGridWorkBenchInitialization`
   - Group related tests in classes

3. **Test method naming**:
   - Use descriptive names: `test_pflow_returns_voltages_by_default`
   - Start with `test_`
   - Describe what is being tested and expected behavior

4. **Use fixtures**:
   - Leverage `conftest.py` fixtures for common setup
   - Create local fixtures for test-specific setup

5. **Mock appropriately**:
   - Unit tests: Mock external dependencies (SAW, file I/O)
   - Integration tests: Use real PowerWorld connections

### Example Test Structure

```python
class TestMyFeature:
    """Tests for MyFeature functionality."""

    def test_basic_operation(self, fixture):
        """Test that basic operation works correctly."""
        # Arrange
        expected = "result"

        # Act
        result = fixture.my_method()

        # Assert
        assert result == expected

    def test_edge_case(self, fixture):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            fixture.my_method(invalid_input)
```

## Contributing

When adding tests:
1. Maintain or improve coverage
2. Follow existing test patterns
3. Add docstrings to test classes and methods
4. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
5. Update this README if adding new test categories
