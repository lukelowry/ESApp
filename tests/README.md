# ESA++ Test Suite

## Quick Start

```bash
pytest                         # Run all tests
pytest -m "not integration"    # Unit tests only (no PowerWorld)
pytest -m integration          # Integration tests only
```

**PowerWorld Setup**: Set the `SAW_TEST_CASE` environment variable to a case path
(and optionally `SAW_GIC_TEST_CASES`, a `;`-separated list for the parametrized
GIC tests). Alternatively, copy `config_test.example.py` to `config_test.py`.

Live tests fail on missing configuration, missing case files, or COM setup
errors. Configured GIC paths are never silently dropped. Run these tests
serially, without pytest-xdist.

## Live Test Isolation

- `saw_session` owns the COM connection and opens a temporary copy of the main case.
- New behavioral tests use `live_case`, which loads a fresh copy for each test
  and restores the session's on-disk case and COM properties afterward.
  Do not use it to preserve in-memory changes made by another test.
- `radial_case` builds a two-bus, one-line network for deterministic topology tests.
- GIC tests also use disposable copies. Source-file hashes guard against changes
  or deletion; cleanup errors are test errors, not warnings.
- Use `tmp_path` for outputs. Assert setup and observable results, not just a
  successful call or the existence of a pre-created file.
- Skip only a verified unavailable optional capability. Unexpected Simulator
  errors, including access violations, must fail. The two known unreadable
  SimAuto component types assert their exact rejection on the verified build.

The modify tests no longer depend on numbered ordering or `SaveState` rollback.
Both merge operations were rechecked in separate Python/Simulator processes on
Simulator 24 (December 11, 2025) with prepared topology and succeeded. Their
blanket crash skips have been removed. Other legacy modules still contain
ordered smoke tests and should be migrated incrementally, not by simply
removing their order markers.

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

Preferred: environment variables (keep machine-specific paths out of the repo):

```powershell
setx SAW_TEST_CASE "C:\path\to\test_case.pwb"
setx SAW_GIC_TEST_CASES "C:\path\case1.pwb;C:\path\case2.pwb"
```

Alternative: create `config_test.py` from the example template:

```python
SAW_TEST_CASE = r"C:\path\to\test_case.pwb"
```
