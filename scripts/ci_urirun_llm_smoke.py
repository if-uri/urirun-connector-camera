#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

from urirun_llm_runtime import Executor


SMOKE_URI = "camera://host/doctor/query/report"
CONNECTOR_ID = "camera"
FORBIDDEN_PREFIXES = ('kvm://', 'fs://', 'pdf://', 'browser://', 'http-check://', 'httpcheck://', 'router://', 'mqtt://', 'github://', 'ocr://', 'usb://', 'adb://', 'node://', 'email://', 'calendar://', 'doc://', 'docid://', 'document://', 'flow://', 'invoice://', 'koru://', 'ksef://', 'linkedin://', 'llm://', 'dns://', 'netscan://', 'task://', 'sheet://', 'smartcrop://', 'data://', 'artifact://', 'check://', 'log://', 'time://', 'twin://', 'urifix://', 'urivision://', 'runtime://', 'vdisplay://', 'view://', 'vql://', 'webnode://', 'youtube://', 'agent://', 'connector://', 'adopt://', 'monitor://', 'device://', 'webcam://')


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _route_value(response: dict) -> dict:
    result = response.get("result")
    value = result.get("value", result) if isinstance(result, dict) else result
    if not isinstance(value, dict):
        fail(f"Response does not contain a dict result value: {response!r}")
    return value


def main() -> None:
    node_url = os.environ.get("URIRUN_NODE_URL", "http://127.0.0.1:18765")
    executor = Executor(node_url)

    health = executor.health()
    if not isinstance(health, dict):
        fail(f"/health returned non-dict response: {health!r}")

    routes = executor.routes()
    if SMOKE_URI not in routes:
        fail(f"{SMOKE_URI} is missing from /routes. Routes: {routes!r}")

    unexpected = [route for route in routes if route.startswith(FORBIDDEN_PREFIXES)]
    if unexpected:
        fail(f"Unexpected non-current connector routes found: {unexpected!r}")

    response = executor.execute(SMOKE_URI, {})
    if not isinstance(response, dict):
        fail(f"Executor returned non-dict response: {response!r}")
    if response.get("ok") is not True:
        fail(f"Executor returned failed response: {response!r}")

    value = _route_value(response)
    if value.get("ok") is not True:
        fail(f"Doctor route returned failed response: {response!r}")
    if value.get("connector") != CONNECTOR_ID:
        fail(f"Doctor route returned wrong connector: {response!r}")
    if value.get("status") != "ready":
        fail(f"Doctor route returned non-ready status: {response!r}")

    print("OK: urirun-llm-runtime -> urirun node -> connector smoke test passed")


if __name__ == "__main__":
    main()
