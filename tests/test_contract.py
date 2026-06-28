# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.
"""Kontrakt connectora konformuje i pokrywa KAŻDĄ trasę handlera (anty-dryf)."""
from __future__ import annotations

import json
import os

import pytest

_uc = pytest.importorskip("urirun_contract")
_scaffold = pytest.importorskip("urirun_contract.contract_scaffold")
conform, Contract = _uc.conform, _uc.Contract

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "urirun_connector_camera")
CONTRACTS = os.path.join(PKG, "contracts.json")


def _load() -> dict:
    doc = json.load(open(CONTRACTS))
    return {r: Contract(version=c["version"], effect=c["effect"], reversible=c["reversible"],
                        inverse_route=c.get("inverseRoute", ""),
                        inp=c["inp"], out=c["out"], errors=tuple(c["errors"]),
                        examples=tuple(c["examples"]))
            for r, c in doc["contracts"].items()}


def test_contract_conforms():
    conform(_load())


def test_every_handler_route_has_a_contract():
    core = open(os.path.join(PKG, "core.py"), encoding="utf-8").read()
    declared = set(json.load(open(CONTRACTS))["contracts"])
    for route in _scaffold.discover_routes(core):
        assert _scaffold.route_key(route) in declared, f"trasa {route!r} z core.py nie ma kontraktu"
