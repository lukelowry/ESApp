"""
Global fixtures for the ESA++ test suite.

Provides reusable test fixtures for both offline (mocked) and online
(integration) testing of the ESA++ library.
"""
import pytest
import os
import hashlib
import shutil
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from esapp.saw import SAW


def _get_test_case_path():
    """
    Get the test case path from configuration.

    Priority order:
    1. Environment variable SAW_TEST_CASE
    2. config_test.py file
    3. None (live fixture reports missing configuration)
    """
    env_path = os.environ.get("SAW_TEST_CASE")
    if env_path:
        return env_path

    try:
        import config_test
        if hasattr(config_test, 'SAW_TEST_CASE'):
            return config_test.SAW_TEST_CASE
    except ModuleNotFoundError as error:
        if error.name != "config_test":
            raise

    return None


def _get_gic_test_cases():
    """
    Get additional GIC test case paths from configuration.

    Priority order:
    1. Environment variable SAW_GIC_TEST_CASES (os.pathsep-separated paths)
    2. config_test.py file
    3. Empty list (fall back to the main case)

    Returns a list of (path, label) tuples for parametrization.
    Missing files are reported by the live fixture, not silently omitted.
    """
    def _cases(paths):
        return [(str(path), Path(path).stem) for path in paths]

    env_paths = os.environ.get("SAW_GIC_TEST_CASES")
    if env_paths:
        return _cases(p.strip() for p in env_paths.split(os.pathsep) if p.strip())

    try:
        import config_test
        if hasattr(config_test, 'GIC_TEST_CASES'):
            return _cases(config_test.GIC_TEST_CASES)
    except ModuleNotFoundError as error:
        if error.name != "config_test":
            raise
    return []


# -------------------------------------------------------------------------
# Integration fixture (live PowerWorld)
# -------------------------------------------------------------------------

@pytest.fixture(scope="session")
def case_path():
    """Validate live configuration and guard the original file against writes."""
    case_path = _get_test_case_path()
    if not case_path:
        pytest.fail("Live tests require SAW_TEST_CASE or tests/config_test.py; use -m 'not integration' for offline tests.", pytrace=False)
    with _unchanged_case(case_path) as source:
        yield source


@pytest.fixture(scope="session")
def saw_session(case_path, tmp_path_factory):
    """One COM connection; even legacy tests open a disposable case copy."""
    working_case = shutil.copy2(case_path, tmp_path_factory.mktemp("powerworld") / case_path.name)
    saw = SAW(str(working_case), CreateIfNotFound=True, early_bind=True)
    try:
        yield saw
    finally:
        saw.exit()


# -------------------------------------------------------------------------
# Case file integrity check
# -------------------------------------------------------------------------

def _file_hash(path):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def _unchanged_case(path):
    source = Path(path).resolve()
    if not source.is_file():
        pytest.fail(f"Configured PowerWorld case does not exist: {source}", pytrace=False)
    original_hash = _file_hash(source)
    try:
        yield source
    finally:
        assert source.is_file(), f"Original case deleted: {source}"
        assert _file_hash(source) == original_hash, f"Original case modified: {source}"


@contextmanager
def _fresh_case(saw, source, directory):
    """Reload topology and options, not just the quantities in SaveState."""
    previous_case = saw.pwb_file_path
    properties = {name: getattr(saw, name) for name in saw.SIMAUTO_PROPERTIES}
    working_case = shutil.copy2(source, directory / Path(source).name)
    saw.CloseCase()
    try:
        saw.set_simauto_property("CreateIfNotFound", True)
        saw.set_simauto_property("UIVisible", False)
        saw.set_simauto_property("CurrentDir", str(directory))
        saw.OpenCase(str(working_case))
        saw.RunScriptCommand("EnterMode(EDIT);")
        yield saw
    finally:
        saw.CloseCase()
        saw.OpenCase(previous_case)
        for name, value in properties.items():
            saw.set_simauto_property(name, value)


@pytest.fixture
def live_case(saw_session, case_path, tmp_path):
    """Independent case for behavioral tests; no state is shared between tests."""
    with _fresh_case(saw_session, case_path, tmp_path) as saw:
        yield saw


@pytest.fixture
def radial_case(live_case):
    """Two buses joined by one line, with fully specified creation fields."""
    live_case.NewCase()
    for number in (1, 2):
        live_case.CreateData(
            "Bus", ["BusNum", "BusName", "BusNomVolt", "AreaNum", "ZoneNum"],
            [number, f"TestBus{number}", 115.0, 1, 1],
        )
    live_case.CreateData(
        "Branch",
        ["BusNum", "BusNum:1", "LineCircuit", "LineR", "LineX",
         "LineAMVA", "LineAMVA:1", "LineAMVA:2", "LineStatus"],
        [1, 2, "1", 0.02, 0.2, 100.0, 100.0, 100.0, "Closed"],
    )
    buses = live_case.GetParametersMultipleElement("Bus", ["BusNum"])
    assert buses is not None and set(buses["BusNum"].astype(int)) == {1, 2}
    branches = live_case.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1"])
    assert branches is not None and len(branches) == 1
    assert tuple(branches.iloc[0].astype(int)) == (1, 2)
    return live_case


# -------------------------------------------------------------------------
# GIC multi-case fixture
# -------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    """Parametrize tests that request the gic_saw fixture."""
    if "gic_saw" in metafunc.fixturenames:
        cases = _get_gic_test_cases()
        if not cases:
            main = _get_test_case_path()
            cases = [(main, Path(main).stem if main else "unconfigured")]
        metafunc.parametrize(
            "gic_saw", [path for path, _ in cases],
            ids=[label for _, label in cases], indirect=True,
        )


@pytest.fixture
def gic_saw(request, saw_session, tmp_path):
    """Exercise each configured GIC case on a fresh copy."""
    with _unchanged_case(request.param) as source:
        with _fresh_case(saw_session, source, tmp_path) as saw:
            yield saw


# -------------------------------------------------------------------------
# Unit test fixture (mocked COM)
# -------------------------------------------------------------------------

@pytest.fixture(scope="function")
def saw_obj():
    """
    Function-scoped mocked SAW object for offline unit tests.

    Patches COM dispatch calls to prevent actual PowerWorld connection.
    """
    with patch("win32com.client.dynamic.Dispatch") as mock_dispatch, \
         patch("win32com.client.gencache.EnsureDispatch", create=True) as mock_ensure_dispatch, \
         patch("tempfile.NamedTemporaryFile") as mock_tempfile, \
         patch("os.unlink"):

        mock_pwcom = MagicMock()
        mock_dispatch.return_value = mock_pwcom
        mock_ensure_dispatch.return_value = mock_pwcom

        mock_ntf = Mock()
        mock_ntf.name = "dummy_temp.axd"
        mock_tempfile.return_value = mock_ntf

        mock_pwcom.RunScriptCommand.return_value = ("",)
        mock_pwcom.ChangeParametersSingleElement.return_value = ("",)
        mock_pwcom.ProcessAuxFile.return_value = ("",)
        mock_pwcom.SaveCase.return_value = ("",)
        mock_pwcom.CloseCase.return_value = ("",)
        mock_pwcom.GetCaseHeader.return_value = ("",)
        mock_pwcom.ChangeParametersMultipleElementRect.return_value = ("",)
        mock_pwcom.GetParametersMultipleElement.return_value = ("", [[1, 2], ["Bus1", "Bus2"]])
        mock_pwcom.OpenCase.return_value = ("",)
        mock_pwcom.GetParametersSingleElement.return_value = ("", ("23", "Jan 01 2023"))
        field_list_data = [
            ["*1*", "BusNum", "Integer", "Bus Number", "Bus Number"],
            ["*2*", "BusName", "String", "Bus Name", "Bus Name"],
        ]
        mock_pwcom.GetFieldList.return_value = ("", field_list_data)

        saw_instance = SAW(FileName="dummy.pwb")
        saw_instance._pwcom = mock_pwcom

        yield saw_instance


# -------------------------------------------------------------------------
# Utility fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test file operations."""
    return tmp_path


@pytest.fixture
def temp_file(tmp_path):
    """Legacy filename factory backed by pytest's per-test directory."""
    from itertools import count
    sequence = count()

    def _create(suffix):
        path = tmp_path / f"output_{next(sequence)}{suffix}"
        path.touch()
        return str(path)

    return _create


# -------------------------------------------------------------------------
# Test configuration
# -------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on file naming."""
    for item in items:
        if "test_integration_" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.requires_case)
        elif "test_" in item.nodeid and "test_integration_" not in item.nodeid:
            item.add_marker(pytest.mark.unit)


# -------------------------------------------------------------------------
# Shared test utilities
# -------------------------------------------------------------------------

def get_all_gobject_subclasses():
    """Recursively find all GObject subclasses with a _TYPE attribute."""
    try:
        from esapp import components as grid
    except ImportError:
        return []

    all_subclasses = []
    q = list(grid.GObject.__subclasses__())
    visited = set(q)
    while q:
        cls = q.pop(0)
        if hasattr(cls, '_TYPE'):
            all_subclasses.append(cls)
        for subclass in cls.__subclasses__():
            if subclass not in visited:
                visited.add(subclass)
                q.append(subclass)
    return all_subclasses


def get_sample_gobject_subclasses(require_keys=False, require_multiple_editable=False, require_editable_non_key=False):
    """Return a representative sample of GObject subclasses for faster parametrized tests.

    Parameters
    ----------
    require_keys : bool
        If True, only return classes with at least one key field.
    require_multiple_editable : bool
        If True, only return classes with at least 2 editable non-key fields.
    require_editable_non_key : bool
        If True, only return classes with at least 1 editable non-key field.
    """
    try:
        from esapp import components as grid
        all_classes = get_all_gobject_subclasses()

        if not all_classes:
            import warnings
            warnings.warn("No GObject subclasses found.")
            return []

        # Apply filters if requested
        if require_keys:
            all_classes = [c for c in all_classes if hasattr(c, 'keys') and c.keys()]

        if require_editable_non_key:
            def has_editable_non_key(cls):
                if not hasattr(cls, 'editable') or not hasattr(cls, 'keys'):
                    return False
                editable_non_key = [f for f in cls.editable() if f not in cls.keys()]
                return len(editable_non_key) >= 1
            all_classes = [c for c in all_classes if has_editable_non_key(c)]

        if require_multiple_editable:
            def has_multiple_editable(cls):
                if not hasattr(cls, 'editable') or not hasattr(cls, 'keys'):
                    return False
                editable_non_key = [f for f in cls.editable() if f not in cls.keys()]
                return len(editable_non_key) >= 2
            all_classes = [c for c in all_classes if has_multiple_editable(c)]

        priority_types = ['Bus', 'Gen', 'Load', 'Branch', 'Shunt', 'Area', 'Zone',
                         'Contingency', 'Interface', 'InjectionGroup']

        sample = []
        for type_name in priority_types:
            for cls in all_classes:
                if hasattr(cls, 'TYPE') and cls.TYPE() == type_name:
                    sample.append(cls)
                    break

        import random
        random.seed(42)
        remaining = [c for c in all_classes if c not in sample]
        if remaining and len(sample) < 15:
            sample.extend(random.sample(remaining, min(5, len(remaining))))

        return sample
    except (ImportError, Exception) as e:
        import warnings
        warnings.warn(f"Error getting GObject subclasses: {e}")
        return []


def assert_dataframe_valid(df, expected_columns=None, min_rows=1, name="DataFrame"):
    """Assert a DataFrame is valid and has expected structure."""
    import pandas as pd
    assert df is not None, f"{name} is None"
    assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"
    assert len(df) >= min_rows, f"{name} has {len(df)} rows, expected at least {min_rows}"
    if expected_columns:
        for col in expected_columns:
            assert col in df.columns, f"{name} missing column: {col}"


def ensure_areas(saw, min_count=2):
    """Ensure at least *min_count* areas exist with buses, creating if needed.

    If the case has fewer areas than *min_count*, new areas are created
    and buses are reassigned from the largest area so each area has
    network elements (required for ATC, directions, etc.).

    Returns the area DataFrame (guaranteed to have >= min_count rows).
    """
    areas = saw.GetParametersMultipleElement("Area", ["AreaNum"])
    if areas is not None and len(areas) >= min_count:
        return areas
    existing = set(int(a) for a in areas["AreaNum"]) if areas is not None and not areas.empty else set()
    next_num = max(existing, default=0) + 1
    buses = saw.GetParametersMultipleElement("Bus", ["BusNum", "AreaNum"])
    assert buses is not None and not buses.empty, "Test case must contain buses"
    buses["BusNum"] = buses["BusNum"].astype(str)
    buses["AreaNum"] = buses["AreaNum"].astype(str)
    while len(existing) < min_count:
        saw.CreateData("Area", ["AreaNum", "AreaName"], [next_num, f"TestArea{next_num}"])
        area_counts = buses["AreaNum"].value_counts()
        largest_area = area_counts.index[0]
        donor_buses = buses[buses["AreaNum"] == largest_area]
        if len(donor_buses) > 1:
            bus_to_move = str(donor_buses.iloc[-1]["BusNum"])
            saw.ChangeParametersSingleElement(
                "Bus", ["BusNum", "AreaNum"], [bus_to_move, next_num]
            )
            buses.loc[buses["BusNum"] == bus_to_move, "AreaNum"] = str(next_num)
        existing.add(next_num)
        next_num += 1
    return saw.GetParametersMultipleElement("Area", ["AreaNum"])


# -------------------------------------------------------------------------
# PW Log Capture — on test failure the PowerWorld message log is
# retrieved and printed so you can see exactly what PW did.
# -------------------------------------------------------------------------

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Record failures for the log fixture."""
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        item._pw_test_failed = True


@pytest.fixture(autouse=True)
def _capture_pw_log(request):
    """Clear the PW log before each test; on failure, dump it to stdout."""
    fixture_name = next(
        (name for name in ("live_case", "gic_saw", "saw_session") if name in request.fixturenames),
        None,
    )
    if fixture_name is None:
        yield
        return

    saw_session = request.getfixturevalue(fixture_name)

    try:
        saw_session.LogClear()
    except Exception as error:
        request.node.add_report_section("setup", "PowerWorld log", f"Could not clear log: {error}")

    yield

    # Only retrieve the log when the test failed
    if not getattr(request.node, "_pw_test_failed", False):
        return

    try:
        log_path = request.getfixturevalue("tmp_path") / "powerworld.log"
        saw_session.LogSave(str(log_path))
        request.node.add_report_section("teardown", "PowerWorld log", log_path.read_text(errors="replace"))
    except Exception as error:
        request.node.add_report_section("teardown", "PowerWorld log", f"Could not retrieve log: {error}")
