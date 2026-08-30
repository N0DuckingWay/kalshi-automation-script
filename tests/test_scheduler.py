"""Tests for scheduler.py — the weekly daemon's subprocess-spawning job and
its logging setup. All Kalshi API interaction is mocked per project policy
(tests must run offline); run_job() itself never touches the network, only
subprocess.run, which is monkeypatched here."""
import logging
import subprocess
import sys
from types import SimpleNamespace

from kalshi_betting import scheduler
from kalshi_betting.config import PROJECT_ROOT, SCHEDULER_JOB_TIMEOUT_SECONDS


class TestRunJob:
    def test_run_job_uses_project_root_cwd_and_timeout(self, monkeypatch):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
        scheduler.run_job()

        assert captured["argv"] == [
            sys.executable, "-m", "kalshi_betting.main", "--mode", "prod",
        ]
        assert captured["kwargs"]["cwd"] == str(PROJECT_ROOT)
        assert captured["kwargs"]["timeout"] == SCHEDULER_JOB_TIMEOUT_SECONDS

    def test_run_job_timeout_logged_and_swallowed(self, monkeypatch, caplog):
        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
        with caplog.at_level(logging.ERROR):
            scheduler.run_job()  # must not raise — a hung run is logged, not fatal

        assert any("timeout" in r.getMessage().lower() for r in caplog.records)

    def test_run_job_nonzero_exit_logs_error(self, monkeypatch, caplog):
        def fake_run(argv, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")

        monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
        with caplog.at_level(logging.ERROR):
            scheduler.run_job()

        assert any(
            "failed" in r.getMessage().lower() and r.levelno == logging.ERROR
            for r in caplog.records
        )
