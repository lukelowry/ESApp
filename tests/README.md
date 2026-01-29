# ESA++ Test Suite

## Quick Start

```bash
pytest                         # Run all tests
pytest -k "not integration"    # Unit tests only (no PowerWorld)
pytest -m integration          # Integration tests only
```

**PowerWorld Setup**: Copy `config_test.example.py` to `config_test.py` and set `SAW_TEST_CASE` path.

## Test Categories

| Category | Description |
|----------|-------------|
| Unit | Mock-based tests, no PowerWorld required |
| Integration | Requires live PowerWorld connection |
| Component | Grid component and data access validation |

## Running with Coverage

```bash
pytest --cov=esapp --cov-report=html
```

## Configuration

Create `config_test.py` from the example template:

```python
SAW_TEST_CASE = r"C:\path\to\test_case.pwb"
```
