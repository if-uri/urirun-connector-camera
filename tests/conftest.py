"""Isolate the shared transaction ledger during tests so they never append to the user's real
~/.urirun/ledger.jsonl. Each test writes to a throwaway file (tests that need a specific path
override URIRUN_LEDGER themselves)."""
import pytest


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("URIRUN_LEDGER", str(tmp_path_factory.mktemp("led") / "ledger.jsonl"))
