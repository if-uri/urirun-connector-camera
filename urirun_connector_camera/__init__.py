# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.

"""camera:// connector — capture a webcam frame, crop to the main object, and OCR it."""

from .core import (
    CAMERA,
    analyze,
    capture,
    compare,
    connector_manifest,
    describe_photo,
    ingest,
    inspect_photo,
    list_cameras,
    main,
    photo_ocr,
    read_barcodes,
    receipt_parse,
    urirun_bindings,
)

__all__ = [
    "CAMERA",
    "analyze",
    "capture",
    "compare",
    "connector_manifest",
    "describe_photo",
    "ingest",
    "inspect_photo",
    "list_cameras",
    "main",
    "photo_ocr",
    "read_barcodes",
    "receipt_parse",
    "urirun_bindings",
]
