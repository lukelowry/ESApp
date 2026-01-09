# ESA++ Test Suite

**Coverage: 92%**

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
| **Unit Tests** | `test_exceptions.py`<br>`test_saw_core_methods.py`<br>`test_workbench_unit.py` | Mock-based tests, no PowerWorld required |
| **Integration** | `test_integration_*.py` | Real PowerWorld case testing |
| **Component** | `test_grid_components.py`<br>`test_indexable_data_access.py` | Data access & grid definitions |
| **Apps** | `test_apps_network_gic.py` | High-level application testing |

> **Note**: `test_grid_components.py` generates ~3,800 parametrized tests validating 958 auto-generated component classes.

## Coverage by Module

| Module | Coverage | Priority Gaps |
|--------|----------|--------------|
| `powerflow.py` | 91.59% | ✅ Well tested |
| `transient.py` | 89.56% | ⚠️ CCT, results extraction |
| `base.py` | 81.96% | ⚠️ Error handling paths |
| `indexable.py` | 76.67% | ⚠️ Edge cases, complex filters |
| `workbench.py` | 60.90% | 🔴 High-level convenience methods |
| `contingency.py` | ✅ | Fully tested |
| `fault.py` | ✅ | Fully tested |

**Intentionally Excluded**:
- `grid.py` — Auto-generated (175k+ lines)
- `apps/static.py`, `apps/dynamics.py` — Research code
- `utils/*` — Specialized data processing

## Troubleshooting

| Problem | Solution |
|---------|----------|
| PowerWorld not found | Set path in `config_test.py` |
| Import errors | Run `pip install -e .` from root |
| Slow integration tests | Use `pytest -m "not slow"` |
| Coverage report | Open `htmlcov/index.html` after `pytest` |
