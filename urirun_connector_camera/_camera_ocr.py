"""Barcode decode and OCR helpers extracted from core.py."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any


def _decode_barcodes(path: str) -> dict[str, Any]:
    """Decode barcodes / QR codes in an image. Uses pyzbar directly when available (it
    also returns positions), falling back to img2nl's analyze_barcodes. Returns a list of
    {type, data, rect} plus a backend tag; backend='none' when no decoder is installed."""
    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "backend": "none", "error": f"Pillow unavailable: {exc}", "codes": []}

    try:
        from pyzbar.pyzbar import decode as _zbar_decode  # type: ignore
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw)
            decoded = _zbar_decode(img)
        codes = [{
            "type": d.type,
            "data": d.data.decode("utf-8", "replace"),
            "rect": [d.rect.left, d.rect.top, d.rect.width, d.rect.height],
            "quality": getattr(d, "quality", None),
        } for d in decoded]
        return {"ok": True, "backend": "pyzbar", "count": len(codes), "codes": codes}
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "backend": "pyzbar", "error": str(exc), "codes": []}

    # fall back to the img2nl engine (which itself wraps pyzbar) reachable via the OCR connector
    try:
        import urirun_connector_ocr.core as _ocr  # type: ignore
        _ocr._extend_source_paths()
        from img2nl.features.barcodes import analyze_barcodes  # type: ignore
        from PIL import Image  # type: ignore
        data = analyze_barcodes(Image.open(path))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "backend": "none",
                "error": f"no barcode decoder (pip install pyzbar): {exc}", "codes": []}
    if not data.get("available"):
        return {"ok": False, "backend": "img2nl", "codes": [],
                "error": str(data.get("reason") or "barcode backend unavailable")}
    return {"ok": True, "backend": "img2nl", "count": data.get("count", 0),
            "codes": data.get("codes", [])}


def _ocr_via_connector(path: str, lang: str, max_chars: int) -> dict[str, Any] | None:
    """Use urirun-connector-ocr's richer image backends when that connector is installed."""
    try:
        from urirun_connector_ocr.core import image_text  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    res = image_text(image=path, backend="auto", lang=lang, max_chars=max_chars)
    value = res.get("result", res) if isinstance(res, dict) else {}
    # image_text already returns a urirun.ok/fail envelope.
    if not isinstance(res, dict):
        return None
    if res.get("ok"):
        return {"ok": True, "backend": f"ocr-connector:{res.get('backend', 'auto')}",
                "text": res.get("text", ""), "chars": res.get("chars", 0)}
    return {"ok": False, "backend": "ocr-connector", "error": res.get("error", "ocr failed")}


def _ocr_tesseract(path: str, lang: str, max_chars: int, psm: int, timeout: int) -> dict[str, Any]:
    if not shutil.which("tesseract"):
        return {"ok": False, "backend": "tesseract", "error": "tesseract is not installed"}
    argv = ["tesseract", path, "stdout"]
    if lang:
        argv += ["-l", lang]
    if psm:
        argv += ["--psm", str(psm)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "backend": "tesseract", "error": f"tesseract timed out after {timeout}s"}
    if proc.returncode != 0 and "+" in lang:
        proc = subprocess.run(["tesseract", path, "stdout", "-l", lang.split("+", 1)[0]],
                              capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        return {"ok": False, "backend": "tesseract",
                "error": (proc.stderr or f"tesseract exited {proc.returncode}").strip()}
    text = proc.stdout
    truncated = len(text) > max_chars > 0
    return {"ok": True, "backend": "tesseract", "text": text[:max_chars] if truncated else text,
            "chars": min(len(text), max_chars) if max_chars else len(text), "truncated": truncated}


def _ocr(path: str, lang: str, max_chars: int, psm: int, timeout: int, prefer_connector: bool) -> dict[str, Any]:
    if prefer_connector:
        via = _ocr_via_connector(path, lang, max_chars)
        if via and via.get("ok") and str(via.get("text", "")).strip():
            return via
    return _ocr_tesseract(path, lang, max_chars, psm, timeout)
