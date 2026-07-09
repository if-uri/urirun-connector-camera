"""Compatibility helpers for urirun authoring APIs used by this connector."""
from __future__ import annotations

import importlib
from typing import Any

import urirun


def connector(connector_id: str, **kwargs: Any):
    """Create a urirun connector using the current or legacy export location."""
    make_connector = getattr(urirun, "connector", None)
    if make_connector is None:
        make_connector = importlib.import_module("urirun._connector").connector
    return make_connector(connector_id, **kwargs)


def load_manifest(package: str, name: str = "connector.manifest.json") -> dict:
    """Load a connector manifest using the current or legacy export location."""
    load = getattr(urirun, "load_manifest", None)
    if load is None:
        load = importlib.import_module("urirun._connector").load_manifest
    return load(package, name)
