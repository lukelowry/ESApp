"""Offline regression tests for live-test isolation and configuration."""

from pathlib import Path
from unittest.mock import Mock, call

import pytest

import conftest


def test_live_case_requires_configuration(monkeypatch):
    monkeypatch.setattr(conftest, "_get_test_case_path", lambda: None)
    with pytest.raises(pytest.fail.Exception, match="Live tests require SAW_TEST_CASE"):
        next(conftest.case_path.__wrapped__())


def test_gic_configuration_keeps_missing_paths(monkeypatch, tmp_path):
    missing = tmp_path / "missing.pwb"
    monkeypatch.setenv("SAW_GIC_TEST_CASES", str(missing))
    assert conftest._get_gic_test_cases() == [(str(missing), "missing")]


def test_missing_case_fails(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="does not exist"):
        with conftest._unchanged_case(tmp_path / "missing.pwb"):
            pytest.fail("Missing case was accepted")


@pytest.mark.parametrize("delete", [False, True], ids=["modified", "deleted"])
def test_case_integrity_detects_damage(tmp_path, delete):
    source = tmp_path / "source.pwb"
    source.write_bytes(b"original")
    with pytest.raises(AssertionError, match="Original case"):
        with conftest._unchanged_case(source):
            if delete:
                source.unlink()
            else:
                source.write_bytes(b"changed")


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_fresh_case_restores_connection(tmp_path, fail):
    source = tmp_path / "source.pwb"
    source.write_bytes(b"original")
    working = tmp_path / "working"
    working.mkdir()
    saw = Mock(
        pwb_file_path="session.pwb",
        SIMAUTO_PROPERTIES=("CreateIfNotFound", "UIVisible", "CurrentDir"),
        CreateIfNotFound=False, UIVisible=False, CurrentDir=str(tmp_path),
    )

    def exercise():
        with conftest._fresh_case(saw, source, working) as active:
            assert active is saw
            copied = Path(saw.OpenCase.call_args.args[0])
            assert copied.parent == working
            assert copied.read_bytes() == b"original"
            copied.write_bytes(b"modified working case")
            if fail:
                raise ValueError("test failed")

    if fail:
        with pytest.raises(ValueError, match="test failed"):
            exercise()
    else:
        exercise()

    assert source.read_bytes() == b"original"
    assert saw.CloseCase.call_count == 2
    saw.OpenCase.assert_called_with("session.pwb")
    assert saw.set_simauto_property.call_args_list[-3:] == [
        call("CreateIfNotFound", False),
        call("UIVisible", False),
        call("CurrentDir", str(tmp_path)),
    ]


def test_session_cleanup_error_is_not_suppressed(monkeypatch, tmp_path):
    source = tmp_path / "case.pwb"
    source.write_bytes(b"original")
    working = tmp_path / "session"
    working.mkdir()
    factory = Mock()
    factory.mktemp.return_value = working
    saw = Mock()
    saw.exit.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(conftest, "SAW", Mock(return_value=saw))
    session = conftest.saw_session.__wrapped__(source, factory)
    assert next(session) is saw
    with pytest.raises(RuntimeError, match="cleanup failed"):
        next(session)


def test_log_capture_does_not_open_a_connection_for_integration_marker():
    request = Mock(fixturenames=[])
    capture = conftest._capture_pw_log.__wrapped__(request)
    next(capture)
    with pytest.raises(StopIteration):
        next(capture)
    request.getfixturevalue.assert_not_called()
