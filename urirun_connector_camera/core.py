# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.
#
# camera:// connector — point a USB/built-in webcam at the world and turn a frame into
# data. It (1) discovers cameras (reusing the usb connector when present, else /dev/video*),
# (2) captures a still frame with ffmpeg (or OpenCV when available), (3) finds the main
# object/region in the frame by edge density, crops to it ("dociąć do obiektu"), and
# (4) runs OCR on the crop to read whatever text is on it. The whole pipeline lives behind
# camera://host/photo/query/analyze. Capture uses ffmpeg + Pillow (no OpenCV required);
# OCR uses tesseract, with the richer urirun-connector-ocr backends used automatically
# when that connector is installed.

from __future__ import annotations

import base64
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from typing import Any

import urirun

CONNECTOR_ID = "camera"
CAMERA = urirun.connector(CONNECTOR_ID, scheme="camera", target="host", meta={"label": "Camera capture + OCR"})


def _tag(result: dict[str, Any], kind: str, *, live: bool = False) -> dict[str, Any]:
    """Stamp a result with the static-vs-live contract: `kind` (photo/scan/text/stream/…) and
    `live` (true = self-updating widget / live view; false = a frozen, immutable artifact). A
    UI renders by `live`, not by media type — a captured frame and a recorded clip are both
    artifacts; only an open stream is a widget."""
    if isinstance(result, dict):
        result["kind"] = kind
        result["live"] = live
    return result


def _ledger(event: str, **fields: Any) -> None:
    """Best-effort append of one transaction line to the shared ledger so every run leaves a
    trace. Path: env URIRUN_LEDGER (default ~/.urirun/ledger.jsonl); set to 0/off to disable.
    Never raises and never logs secrets or full OCR text."""
    path = os.getenv("URIRUN_LEDGER", os.path.expanduser("~/.urirun/ledger.jsonl"))
    if path.lower() in ("0", "off", "none", ""):
        return
    try:
        rec = {"ts": time.time(), "connector": CONNECTOR_ID, "event": event,
               "live": False, **fields}  # ledger only holds frozen artifacts, never widgets
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - telemetry must never break a route
        pass


# --------------------------------------------------------------------------- helpers

def _device_path(device: str) -> str:
    """Accept '/dev/video0', 'video0', or a bare index '0' → '/dev/video0'."""
    device = (device or "").strip()
    if not device:
        return ""
    if device.startswith("/dev/"):
        return device
    if device.isdigit():
        return f"/dev/video{device}"
    return f"/dev/{device}"


def _list_video_nodes() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def _default_device() -> str:
    nodes = _list_video_nodes()
    return nodes[0] if nodes else ""


def _b64_file(path: str, max_bytes: int) -> tuple[str, int]:
    size = os.path.getsize(path)
    if size > max_bytes:
        return "", size
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii"), size


def _audio_beep(
    enabled: bool,
    *,
    frequency: int = 1200,
    duration_ms: int = 180,
    count: int = 1,
    interval_ms: int = 80,
) -> dict[str, Any]:
    """Audible pre-scan cue. Best effort: try a real audio device, then terminal BEL."""
    if not enabled:
        return {"ok": True, "enabled": False}
    freq = max(80, min(int(frequency or 1200), 8000))
    duration = max(20, min(int(duration_ms or 180), 3000))
    repeat = max(1, min(int(count or 1), 8))
    interval = max(0, min(int(interval_ms or 80), 2000))

    if shutil.which("beep"):
        argv = ["beep", "-f", str(freq), "-l", str(duration), "-r", str(repeat), "-d", str(interval)]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
        if proc.returncode == 0:
            return {"ok": True, "enabled": True, "backend": "beep", "frequency": freq, "durationMs": duration, "count": repeat}

    if shutil.which("play"):
        argv = ["play", "-q", "-n", "synth", str(duration / 1000.0), "sine", str(freq)]
        ok = True
        last_err = ""
        for _ in range(repeat):
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
            if proc.returncode != 0:
                ok = False
                last_err = proc.stderr.strip()
                break
            if interval:
                time.sleep(interval / 1000.0)
        if ok:
            return {"ok": True, "enabled": True, "backend": "play", "frequency": freq, "durationMs": duration, "count": repeat}
        if last_err:
            last_error = last_err
        else:
            last_error = "play failed"
    else:
        last_error = "no beep/play command"

    # Generate a short WAV with stdlib and play it through common desktop audio tools.
    with tempfile.TemporaryDirectory(prefix="urirun-camera-beep-") as tmp:
        wav_path = os.path.join(tmp, "beep.wav")
        rate = 44100
        frames = int(rate * duration / 1000.0)
        with wave.open(wav_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            samples = bytearray()
            for i in range(frames):
                val = int(12000 * math.sin(2 * math.pi * freq * (i / rate)))
                samples.extend(int(val).to_bytes(2, "little", signed=True))
            wav.writeframes(bytes(samples))
        for player in ("paplay", "aplay", "ffplay"):
            if not shutil.which(player):
                continue
            argv = [player, wav_path] if player != "ffplay" else [player, "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path]
            ok = True
            last_err = ""
            for _ in range(repeat):
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
                if proc.returncode != 0:
                    ok = False
                    last_err = proc.stderr.strip()
                    break
                if interval:
                    time.sleep(interval / 1000.0)
            if ok:
                return {"ok": True, "enabled": True, "backend": player, "frequency": freq, "durationMs": duration, "count": repeat}
            last_error = last_err or f"{player} failed"

    try:
        print("\a", end="", file=sys.stderr, flush=True)
        return {"ok": True, "enabled": True, "backend": "terminal-bell", "warning": last_error}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "enabled": True, "error": f"{last_error}; terminal bell failed: {exc}"}


# --------------------------------------------------------------------------- capture

def _capture_ffmpeg(device: str, out_path: str, *, warmup: int, width: int,
                    height: int, timeout: int) -> dict[str, Any]:
    """Grab a still frame with ffmpeg's v4l2 input. We capture `warmup` frames into a
    temp pattern and keep the last one, so auto-exposure has time to settle."""
    if not shutil.which("ffmpeg"):
        return {"ok": False, "error": "ffmpeg is not installed"}
    if not os.path.exists(device):
        return {"ok": False, "error": f"camera device not found: {device}"}

    frames = max(1, int(warmup))
    with tempfile.TemporaryDirectory(prefix="urirun-camera-") as tmp:
        pattern = os.path.join(tmp, "frame_%04d.jpg")
        argv = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "v4l2"]
        if width and height:
            argv += ["-video_size", f"{int(width)}x{int(height)}"]
        argv += ["-i", device, "-frames:v", str(frames), pattern]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"ffmpeg capture timed out after {timeout}s", "backend": "ffmpeg"}
        captured = sorted(glob.glob(os.path.join(tmp, "frame_*.jpg")))
        if proc.returncode != 0 or not captured:
            return {"ok": False, "backend": "ffmpeg",
                    "error": (proc.stderr or f"ffmpeg exited {proc.returncode}").strip()}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        shutil.copyfile(captured[-1], out_path)
    return {"ok": True, "backend": "ffmpeg", "path": out_path}


def _capture_cv2(device: str, out_path: str, *, warmup: int, width: int,
                 height: int) -> dict[str, Any]:
    """Optional OpenCV capture backend (used only if cv2 is importable)."""
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "backend": "cv2", "error": f"opencv unavailable: {exc}"}
    index = int(device.replace("/dev/video", "")) if device.replace("/dev/video", "").isdigit() else 0
    cap = cv2.VideoCapture(index)
    try:
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if not cap.isOpened():
            return {"ok": False, "backend": "cv2", "error": f"cannot open camera index {index}"}
        frame = None
        for _ in range(max(1, int(warmup))):
            ok, frame = cap.read()
        if frame is None:
            return {"ok": False, "backend": "cv2", "error": "no frame read from camera"}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        cv2.imwrite(out_path, frame)
    finally:
        cap.release()
    return {"ok": True, "backend": "cv2", "path": out_path}


def _capture(device: str, out_path: str, *, backend: str, warmup: int, width: int,
             height: int, timeout: int) -> dict[str, Any]:
    selected = (backend or "auto").strip().lower()
    if selected in ("cv2", "opencv"):
        return _capture_cv2(device, out_path, warmup=warmup, width=width, height=height)
    if selected == "ffmpeg":
        result = _capture_ffmpeg(device, out_path, warmup=warmup, width=width, height=height, timeout=timeout)
        return result
    if selected == "auto":
        attempts = []
        ffmpeg_result = _capture_ffmpeg(device, out_path, warmup=warmup, width=width, height=height, timeout=timeout)
        attempts.append(ffmpeg_result)
        if ffmpeg_result.get("ok"):
            return ffmpeg_result

        cv2_result = _capture_cv2(device, out_path, warmup=warmup, width=width, height=height)
        attempts.append(cv2_result)
        if cv2_result.get("ok"):
            return cv2_result

        errors = [str(r.get("error") or r.get("message") or r.get("backend") or "failed") for r in attempts]
        return {"ok": False, "backend": "auto", "error": "; ".join(errors), "attempts": attempts}
    return {"ok": False, "error": f"unknown capture backend: {backend}"}


# --------------------------------------------------------------- object detection / crop

def _image_size(path: str) -> tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:  # noqa: BLE001
        return 0, 0


def _main_bbox(path: str, *, edge_threshold: int, pad: float, min_fraction: float) -> dict[str, Any]:
    """Find the dominant object/region by edge density (Pillow only — no OpenCV).
    Returns the padded bounding box of the high-detail area, or the full frame when the
    detail spans most of the image (e.g. a document filling the view)."""
    try:
        from PIL import Image, ImageFilter, ImageOps  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Pillow unavailable: {exc}"}
    try:
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("L")
            w, h = img.width, img.height
            edges = img.filter(ImageFilter.FIND_EDGES)
            mask = edges.point(lambda p: 255 if p >= edge_threshold else 0)
            # FIND_EDGES lights up the outermost pixel ring; ignore that border so the
            # bbox reflects real content, not the frame artifact.
            border = max(2, min(w, h) // 200)
            inner = mask.crop((border, border, w - border, h - border))
            box = inner.getbbox()
            if box:
                box = (box[0] + border, box[1] + border, box[2] + border, box[3] + border)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    if not box:
        return {"ok": True, "found": False, "bbox": [0, 0, w, h], "width": w, "height": h}

    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    # Pad the box a little so we don't clip the object's border.
    px, py = int(bw * pad), int(bh * pad)
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(w, x1 + px); y1 = min(h, y1 + py)
    fraction = ((x1 - x0) * (y1 - y0)) / float(w * h or 1)
    found = fraction <= (1.0 - min_fraction) or (bw < w or bh < h)
    return {"ok": True, "found": bool(found), "bbox": [x0, y0, x1 - x0, y1 - y0],
            "coverage": round(fraction, 4), "width": w, "height": h}


def _crop(path: str, bbox: list[int], out_path: str) -> dict[str, Any]:
    try:
        from PIL import Image, ImageOps  # type: ignore
        x, y, bw, bh = bbox
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw)
            crop = img.crop((x, y, x + bw, y + bh))
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            crop.save(out_path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "path": out_path, "bbox": bbox}


# ----------------------------------------------- scene understanding (img2nl, optional)

def _load_img2nl():
    """Return img2nl.api.analyze_image if the engine is reachable, else None. We reuse
    the OCR connector's source-path discovery so img2nl/imgl checkouts get onto sys.path."""
    try:
        import urirun_connector_ocr.core as _ocr  # type: ignore
        _ocr._extend_source_paths()
    except Exception:  # noqa: BLE001
        pass
    try:
        from img2nl.api import analyze_image  # type: ignore
        return analyze_image
    except Exception:  # noqa: BLE001
        return None


def _denorm_bbox(bbox_norm: list[float] | None, w: int, h: int) -> list[int]:
    if not bbox_norm or len(bbox_norm) != 4 or w <= 0 or h <= 0:
        return []
    x0, y0, x1, y1 = bbox_norm
    return [int(round(x0 * w)), int(round(y0 * h)),
            int(round((x1 - x0) * w)), int(round((y1 - y0) * h))]


def _img2nl_regions(path: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Run img2nl on an image → (natural-language description, object regions, extras).
    Returns ('', [], {}) when img2nl is unavailable so callers can fall back."""
    analyze_image = _load_img2nl()
    if not analyze_image:
        return "", [], {}
    try:
        data = analyze_image(path, skip_thumbnail=True, source_type="photo",
                             goal="describe", enable_ui_detect=True).to_dict()
    except Exception:  # noqa: BLE001
        return "", [], {}
    feats = data.get("features") or {}
    w, h = _image_size(path)
    regions = []
    for region in (feats.get("objects") or {}).get("large_regions", []):
        bbox = _denorm_bbox(region.get("bbox_norm"), w, h)
        if bbox:
            regions.append({"bbox": bbox, "areaRatio": region.get("area_ratio")})
    special = feats.get("special_hits") or {}
    extras = {
        "colors": (feats.get("colors") or {}).get("dominant_colors"),
        "scene": feats.get("scene"),
        "barcodes": special.get("barcodes"),
        "hasQr": special.get("has_qr"),
        "hasText": special.get("has_text"),
        "objectCount": (feats.get("objects") or {}).get("large_region_count"),
    }
    return str(data.get("text") or ""), regions, extras


def _describe_basic(path: str) -> dict[str, Any]:
    """Dependency-light fallback description (size, brightness, dominant colors) via Pillow."""
    try:
        from PIL import Image, ImageStat  # type: ignore
        with Image.open(path) as raw:
            img = raw.convert("RGB")
            stat = ImageStat.Stat(img)
            brightness = sum(stat.mean) / 3.0
            pal = img.quantize(colors=3).getpalette()[:9]
            dominant = ["#%02x%02x%02x" % (pal[i], pal[i + 1], pal[i + 2]) for i in range(0, 9, 3)]
            text = (f"Image {img.width}x{img.height}px, mean brightness {brightness:.0f}/255, "
                    f"dominant colors {', '.join(dominant)}.")
    except Exception as exc:  # noqa: BLE001
        return {"backend": "none", "text": "", "error": str(exc)}
    return {"backend": "pillow", "text": text, "dominantColors": dominant,
            "brightness": round(brightness, 1)}


def _describe(path: str) -> dict[str, Any]:
    """Natural-language 'what is in the photo'. Prefers img2nl; falls back to Pillow stats."""
    text, regions, extras = _img2nl_regions(path)
    if text or regions:
        return {"backend": "img2nl", "text": text, "objects": regions,
                "objectCount": extras.get("objectCount") or len(regions),
                "colors": extras.get("colors"), "scene": extras.get("scene"),
                "barcodes": extras.get("barcodes"), "hasQr": extras.get("hasQr"),
                "hasText": extras.get("hasText")}
    return _describe_basic(path)


def _object_bbox(path: str, *, edge_threshold: int, pad: float, min_fraction: float) -> dict[str, Any]:
    """Find the dominant object to crop to. Uses img2nl's region detector when available
    (returns the largest region), otherwise the Pillow edge-density bounding box."""
    _text, regions, _extras = _img2nl_regions(path)
    if regions:
        best = max(regions, key=lambda r: r.get("areaRatio") or 0)
        return {"ok": True, "found": True, "bbox": best["bbox"],
                "coverage": best.get("areaRatio"), "detector": "img2nl"}
    det = _main_bbox(path, edge_threshold=edge_threshold, pad=pad, min_fraction=min_fraction)
    det["detector"] = "edges"
    return det


def _document_bbox(path: str, *, pad: float = 0.02, content_ratio: float = 0.12) -> dict[str, Any]:
    """Detect a sheet/receipt ('paragon') and return a tight crop box around it.

    A receipt is a bright, text-dense rectangle on a darker, plainer background. We combine
    two cues that survive a phone snapshot: brightness (the paper) and edge density (the
    print), then keep the rows/columns whose combined content rises above `content_ratio` of
    the peak. Projection trimming ignores stray specks, so the box hugs the document instead
    of the whole frame. numpy-based; falls back to the edge bbox when numpy is unavailable."""
    try:
        import numpy as np  # type: ignore
        from PIL import Image, ImageOps  # type: ignore
    except Exception:  # noqa: BLE001
        det = _main_bbox(path, edge_threshold=40, pad=pad, min_fraction=0.02)
        det["detector"] = "edges"
        return det
    try:
        with Image.open(path) as raw:
            full = ImageOps.exif_transpose(raw).convert("L")
        full_w, full_h = full.width, full.height
        scale = max(1.0, full_w / 800.0)
        small = full.resize((max(1, int(full_w / scale)), max(1, int(full_h / scale))))
        arr = np.asarray(small, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "detector": "document"}

    h, w = arr.shape
    if h < 8 or w < 8:
        return {"ok": True, "found": False, "bbox": [0, 0, full_w, full_h], "detector": "document"}

    # edge magnitude (gradient) and a bright-paper mask
    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    edges = gx + gy
    bright = arr > (0.6 * float(arr.max()) + 0.4 * float(arr.mean()))
    content = (edges > 28).astype(np.float32) + bright.astype(np.float32) * 0.5

    rows = content.sum(axis=1)
    cols = content.sum(axis=0)

    def _span(proj):
        peak = float(proj.max())
        if peak <= 0:
            return 0, len(proj) - 1
        keep = np.where(proj >= content_ratio * peak)[0]
        return (int(keep[0]), int(keep[-1])) if keep.size else (0, len(proj) - 1)

    r0, r1 = _span(rows)
    c0, c1 = _span(cols)
    # back to full resolution with a little padding
    x0, y0, x1, y1 = c0 * scale, r0 * scale, (c1 + 1) * scale, (r1 + 1) * scale
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * pad, bh * pad
    x0 = max(0, int(x0 - px)); y0 = max(0, int(y0 - py))
    x1 = min(full_w, int(x1 + px)); y1 = min(full_h, int(y1 + py))
    coverage = ((x1 - x0) * (y1 - y0)) / float(full_w * full_h or 1)
    # "found" only if we actually trimmed something meaningful off the frame
    found = coverage < 0.985 and (x1 - x0) > 8 and (y1 - y0) > 8
    return {"ok": True, "found": bool(found), "bbox": [x0, y0, x1 - x0, y1 - y0],
            "coverage": round(coverage, 4), "detector": "document"}


def _target_bbox(path: str, target: str, *, edge_threshold: int, pad: float,
                 min_fraction: float) -> dict[str, Any]:
    """Pick a crop box according to `target`:
      * 'document' / 'receipt' / 'paragon' -> sheet detector (tight crop to the paper);
      * 'object'                            -> dominant object (img2nl / edges);
      * 'auto' (default)                    -> document crop when it trims the frame, else object.
    """
    want = (target or "auto").strip().lower()
    if want == "none":
        return {"ok": True, "found": False, "detector": "none"}
    if want in ("document", "receipt", "paragon", "doc", "page"):
        det = _document_bbox(path, pad=pad)
        if det.get("ok"):
            return det
        return _object_bbox(path, edge_threshold=edge_threshold, pad=pad, min_fraction=min_fraction)
    if want == "object":
        return _object_bbox(path, edge_threshold=edge_threshold, pad=pad, min_fraction=min_fraction)
    # auto: prefer a real document crop, fall back to the object detector
    doc = _document_bbox(path, pad=pad)
    if doc.get("ok") and doc.get("found"):
        return doc
    return _object_bbox(path, edge_threshold=edge_threshold, pad=pad, min_fraction=min_fraction)


# ------------------------------------ perspective correction / deskew (numpy, cv2 optional)

def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _order_pts(pts):
    """Order 4 (x,y) points as top-left, top-right, bottom-right, bottom-left."""
    pts = [(float(x), float(y)) for x, y in pts]
    s = [x + y for x, y in pts]
    d = [x - y for x, y in pts]
    return [pts[s.index(min(s))], pts[d.index(max(d))],
            pts[s.index(max(s))], pts[d.index(min(d))]]


def _quad_cv2(gray):
    """Robust document quad via OpenCV (Canny + contour approx) when cv2 is installed."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        g = gray.astype("uint8")
        g = cv2.GaussianBlur(g, (5, 5), 0)
        edged = cv2.dilate(cv2.Canny(g, 50, 150), np.ones((5, 5), "uint8"))
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.1 * gray.size:
                return _order_pts(approx.reshape(4, 2))
    except Exception:  # noqa: BLE001
        return None
    return None


def _block_reduce(mask, factor: int):
    """Average-pool a boolean mask by `factor` and re-threshold. Compact specks (a stray
    reflection a few pixels wide) average below 0.5 and vanish; the large sheet survives —
    a dependency-free stand-in for morphological opening / connected-component cleanup."""
    import numpy as np  # type: ignore
    if factor <= 1:
        return mask, 1
    h, w = mask.shape
    big_h, big_w = (h // factor) * factor, (w // factor) * factor
    if big_h < factor or big_w < factor:
        return mask, 1
    reduced = mask[:big_h, :big_w].reshape(big_h // factor, factor,
                                           big_w // factor, factor).mean(axis=(1, 3))
    return reduced > 0.5, factor


def _quad_numpy(gray, min_area_ratio: float):
    """Document quad from the bright-sheet mask via the sum/diff corner-extremes trick.
    The mask is speckle-cleaned by block-reduction first, so isolated bright spots in the
    background don't steal a corner."""
    import numpy as np  # type: ignore
    thr = 0.6 * float(gray.max()) + 0.4 * float(gray.mean())
    mask = gray > thr
    total = gray.size
    if mask.sum() < 0.03 * total:
        return None

    h, w = gray.shape
    factor = max(1, min(h, w) // 120)
    clean, factor = _block_reduce(mask, factor)
    if clean.sum() < 0.03 * clean.size:        # cleanup removed everything → fall back raw
        clean, factor = mask, 1

    coords = np.argwhere(clean)                 # rows of (y, x) in reduced space
    ys = coords[:, 0].astype(np.float64) * factor
    xs = coords[:, 1].astype(np.float64) * factor
    s = xs + ys
    d = xs - ys
    quad = [(xs[s.argmin()], ys[s.argmin()]), (xs[d.argmax()], ys[d.argmax()]),
            (xs[s.argmax()], ys[s.argmax()]), (xs[d.argmin()], ys[d.argmin()])]
    # shoelace area must cover a meaningful part of the frame
    area = 0.0
    for i in range(4):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    if abs(area) / 2.0 < min_area_ratio * total:
        return None
    return [(float(x), float(y)) for x, y in quad]


def _perspective_coeffs(dest, src):
    """8 PIL PERSPECTIVE coefficients mapping output (dest) pixels back to source (src)."""
    import numpy as np  # type: ignore
    matrix = []
    rhs = []
    for (x, y), (X, Y) in zip(dest, src):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); rhs.append(X)
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y]); rhs.append(Y)
    return tuple(np.linalg.solve(np.array(matrix, dtype=np.float64),
                                 np.array(rhs, dtype=np.float64)))


def _deskew_document(path: str, out_path: str, *, min_area_ratio: float = 0.05) -> dict[str, Any]:
    """Find the document's four corners and warp it flat (perspective correction), so a
    receipt photographed at an angle becomes a straight rectangle before OCR. Uses OpenCV
    for corner detection when available, else a numpy bright-sheet detector; the warp itself
    is done with Pillow (no OpenCV needed)."""
    try:
        import numpy as np  # type: ignore
        from PIL import Image, ImageOps  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"numpy/Pillow unavailable: {exc}"}
    try:
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
        full_w, full_h = img.size
        scale = max(1.0, full_w / 800.0)
        small = img.convert("L").resize((max(1, int(full_w / scale)), max(1, int(full_h / scale))))
        gray = np.asarray(small, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    detector = "cv2"
    quad = _quad_cv2(gray)
    if quad is None:
        detector = "numpy"
        quad = _quad_numpy(gray, min_area_ratio)
    if quad is None:
        return {"ok": True, "found": False}

    # scale corners back to full resolution
    tl, tr, br, bl = [(x * scale, y * scale) for x, y in quad]
    width = int(round(max(_dist(br, bl), _dist(tr, tl))))
    height = int(round(max(_dist(tr, br), _dist(tl, bl))))
    if width < 16 or height < 16:
        return {"ok": True, "found": False}
    dest = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    try:
        coeffs = _perspective_coeffs(dest, [tl, tr, br, bl])
        warped = img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        warped.save(out_path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"warp failed: {exc}", "detector": detector}
    return {"ok": True, "found": True, "path": out_path, "detector": f"{detector}-deskew",
            "corners": [[int(round(x)), int(round(y))] for x, y in (tl, tr, br, bl)],
            "size": [width, height]}


# ------------------------------------------------- change / motion detection (Pillow)

def _frame_diff(path_a: str, path_b: str, *, pixel_threshold: int, downscale: int) -> dict[str, Any]:
    """Compare two frames and report how much changed. Grayscale absolute difference,
    thresholded per pixel; returns the changed fraction and the bounding box of the change
    — the basis for motion / 'something appeared' detection. Pillow only, no OpenCV."""
    try:
        from PIL import Image, ImageChops, ImageOps  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Pillow unavailable: {exc}"}
    try:
        a = ImageOps.exif_transpose(Image.open(path_a)).convert("L")
        b = ImageOps.exif_transpose(Image.open(path_b)).convert("L")
        # normalise to a common, optionally downscaled size so the diff is cheap and aligned
        w = min(a.width, b.width)
        h = min(a.height, b.height)
        if downscale and w > downscale:
            h = max(1, int(h * downscale / w))
            w = downscale
        a = a.resize((w, h))
        b = b.resize((w, h))
        diff = ImageChops.difference(a, b)
        mask = diff.point(lambda p: 255 if p >= pixel_threshold else 0)
        changed_px = mask.histogram()[-1]
        total = w * h or 1
        bbox = mask.getbbox()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    ratio = changed_px / total
    region = None
    if bbox:
        region = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    return {"ok": True, "changeRatio": round(ratio, 4), "pixelsChanged": changed_px,
            "comparedSize": [w, h], "changedRegion": region}


def _append_jsonl(path: str, record: dict[str, Any]) -> str:
    """Append a single JSON record (one line) to an audit log, creating parents as needed."""
    out = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def _write_sidecar(out_dir: str, record: dict[str, Any]) -> str:
    """Write the full structured result next to the photo as inspection.json."""
    out = os.path.join(os.path.expanduser(out_dir), "inspection.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    return out


# ----------------------------------------------------- barcodes / QR codes (optional)

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


# --------------------------------------------------------------------------- OCR

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


# --------------------------------------------------------------------------- routes

@CAMERA.handler("devices/query/list", isolated=True,
                meta={"label": "List available cameras", "cliAlias": "devices"})
def list_cameras() -> dict[str, Any]:
    """List cameras available for capture. Prefers the usb connector's camera discovery
    (so you get product names + the matching /dev/video* nodes); falls back to /dev/video*."""
    nodes = _list_video_nodes()
    cameras: list[dict[str, Any]] = []
    try:
        from urirun_connector_usb.core import cameras as usb_cameras  # type: ignore
        res = usb_cameras()
        if res.get("ok"):
            for cam in res.get("cameras", []):
                cameras.append({"name": cam.get("name"), "id": cam.get("id"),
                                "videoNodes": cam.get("videoNodes", [])})
    except Exception:  # noqa: BLE001
        pass
    return urirun.ok(connector=CONNECTOR_ID, count=len(nodes), videoNodes=nodes,
                     cameras=cameras, default=_default_device(),
                     capture={"ffmpeg": bool(shutil.which("ffmpeg")), "tesseract": bool(shutil.which("tesseract"))})


@CAMERA.handler("photo/command/capture", isolated=True,
                meta={"label": "Capture a still photo from the camera", "cliAlias": "capture"})
def capture(device: str = "", output: str = "", backend: str = "auto", warmup: int = 4,
            width: int = 0, height: int = 0, return_base64: bool = False,
            max_base64_bytes: int = 8 * 1024 * 1024, timeout: int = 30,
            beep: bool = False, beep_required: bool = False,
            beep_frequency: int = 1200, beep_duration_ms: int = 180,
            beep_count: int = 1) -> dict[str, Any]:
    """Capture one still frame from a camera and save it to `output` (default: a temp jpg).
    `device` accepts '/dev/video0', 'video0' or '0' (defaults to the first camera).
    Set return_base64=True to also get the image bytes inline."""
    dev = _device_path(device) or _default_device()
    if not dev:
        return urirun.fail("no camera device found (no /dev/video*)", connector=CONNECTOR_ID)
    out = os.path.expanduser(output) if output else os.path.join(
        tempfile.gettempdir(), f"urirun-camera-{os.path.basename(dev)}.jpg")
    beep_result = _audio_beep(beep, frequency=beep_frequency, duration_ms=beep_duration_ms, count=beep_count)
    if beep_required and not beep_result.get("ok"):
        return urirun.fail(str(beep_result.get("error", "pre-scan beep failed")),
                           connector=CONNECTOR_ID, device=dev, beep=beep_result)
    result = _capture(dev, out, backend=backend, warmup=warmup, width=width, height=height, timeout=timeout)
    if not result.get("ok"):
        return urirun.fail(str(result.get("error", "capture failed")),
                           connector=CONNECTOR_ID, device=dev, backend=result.get("backend"),
                           attempts=result.get("attempts", []))
    w, h = _image_size(out)
    payload: dict[str, Any] = {"device": dev, "backend": result.get("backend"), "path": out,
                               "width": w, "height": h, "bytes": os.path.getsize(out),
                               "beep": beep_result}
    if return_base64:
        b64, size = _b64_file(out, max_base64_bytes)
        payload["bytes_b64"] = b64
        if not b64:
            payload["base64_skipped"] = f"image is {size} bytes > max_base64_bytes"
    return _tag(urirun.ok(connector=CONNECTOR_ID, **payload), "photo")


@CAMERA.handler("photo/query/analyze", isolated=True,
                meta={"label": "Capture, crop to the main object, and OCR it", "cliAlias": "analyze"})
def analyze(device: str = "", image: str = "", output_dir: str = "", backend: str = "auto",
            warmup: int = 4, width: int = 0, height: int = 0, crop: bool = True,
            target: str = "auto", deskew: bool = False, ocr: bool = True, describe: bool = True,
            lang: str = "eng+pol",
            max_chars: int = 12000, psm: int = 3, edge_threshold: int = 40, pad: float = 0.04,
            min_fraction: float = 0.02, timeout: int = 30,
            beep: bool = False, beep_required: bool = False,
            beep_frequency: int = 1200, beep_duration_ms: int = 180,
            beep_count: int = 1) -> dict[str, Any]:
    """End-to-end: take a photo (or use an existing `image`), describe what is in it
    (img2nl scene/objects when available), crop to the target, and OCR the crop to read its
    text. `target` chooses the crop: 'document'/'receipt'/'paragon' tightly crops the sheet,
    'object' crops the dominant object, 'auto' (default) crops the document when one is found
    else the object. deskew=True first attempts a 4-point perspective correction (flatten a
    receipt/page shot at an angle) and OCRs the warped sheet. Returns the photo path, a
    natural-language description, the detected bbox/corners + crop path, and the OCR text.
    crop=False OCRs the whole frame, ocr=False skips reading, describe=False skips scene."""
    out_dir = os.path.expanduser(output_dir) if output_dir else tempfile.mkdtemp(prefix="urirun-camera-")
    os.makedirs(out_dir, exist_ok=True)

    # 1. obtain a frame (capture or reuse a provided image)
    if image:
        photo = os.path.expanduser(image)
        if not os.path.isfile(photo):
            return urirun.fail(f"image not found: {photo}", connector=CONNECTOR_ID)
        device_used = ""
        beep_result = {"ok": True, "enabled": False, "reason": "existing image supplied"}
    else:
        device_used = _device_path(device) or _default_device()
        if not device_used:
            return urirun.fail("no camera device found (no /dev/video*)", connector=CONNECTOR_ID)
        beep_result = _audio_beep(beep, frequency=beep_frequency, duration_ms=beep_duration_ms, count=beep_count)
        if beep_required and not beep_result.get("ok"):
            return urirun.fail(str(beep_result.get("error", "pre-scan beep failed")),
                               connector=CONNECTOR_ID, device=device_used, beep=beep_result)
        photo = os.path.join(out_dir, "photo.jpg")
        cap = _capture(device_used, photo, backend=backend, warmup=warmup, width=width,
                       height=height, timeout=timeout)
        if not cap.get("ok"):
            return urirun.fail(str(cap.get("error", "capture failed")),
                               connector=CONNECTOR_ID, device=device_used, backend=cap.get("backend"),
                               attempts=cap.get("attempts", []))

    w, h = _image_size(photo)
    report: dict[str, Any] = {
        "device": device_used,
        "photo": {"path": photo, "width": w, "height": h, "bytes": os.path.getsize(photo)},
        "beep": beep_result,
    }

    # 2. describe what is in the photo (img2nl scene/objects, else Pillow stats)
    if describe:
        report["description"] = _describe(photo)

    # 3. crop to the target. When deskew=True we first try a 4-point perspective correction
    #    (flatten a receipt/document shot at an angle); otherwise an axis-aligned crop.
    ocr_target = photo
    if crop:
        deskewed = False
        if deskew:
            dsk = _deskew_document(photo, os.path.join(out_dir, "document.jpg"))
            if dsk.get("ok") and dsk.get("found"):
                report["object"] = {"found": True, "detector": dsk["detector"], "target": target,
                                    "corners": dsk["corners"], "size": dsk["size"],
                                    "cropPath": dsk["path"], "deskewed": True}
                ocr_target = dsk["path"]
                deskewed = True
            elif not dsk.get("ok"):
                report["deskewError"] = dsk.get("error")
        if not deskewed:
            det = _target_bbox(photo, target, edge_threshold=edge_threshold, pad=pad, min_fraction=min_fraction)
            if det.get("ok"):
                obj: dict[str, Any] = {"found": det.get("found"), "bbox": det.get("bbox"),
                                       "coverage": det.get("coverage"), "detector": det.get("detector"),
                                       "target": target}
                if det.get("found") and det.get("bbox"):
                    crop_path = os.path.join(out_dir, "object.jpg")
                    cres = _crop(photo, det["bbox"], crop_path)
                    if cres.get("ok"):
                        obj["cropPath"] = crop_path
                        ocr_target = crop_path
                    else:
                        obj["cropError"] = cres.get("error")
                report["object"] = obj
            else:
                report["object"] = {"found": False, "error": det.get("error")}

    # 4. OCR the crop (or the whole frame)
    if ocr:
        ores = _ocr(ocr_target, lang, max_chars, psm, timeout, prefer_connector=True)
        report["ocr"] = {"target": ocr_target, **{k: v for k, v in ores.items() if k != "connector"}}
        text = str(ores.get("text", "")).strip()
        report["contents"] = {
            "hasText": bool(text),
            "textPreview": text[:280],
            "objectFound": bool(report.get("object", {}).get("found")),
            "summary": str(report.get("description", {}).get("text", "")),
        }

    return _tag(urirun.ok(connector=CONNECTOR_ID, outputDir=out_dir, **report), "scan")


@CAMERA.handler("photo/query/describe", isolated=True,
                meta={"label": "Capture a photo and describe what is in it", "cliAlias": "describe"})
def describe_photo(device: str = "", image: str = "", backend: str = "auto", warmup: int = 4,
                   width: int = 0, height: int = 0, output_dir: str = "",
                   timeout: int = 30, beep: bool = False, beep_required: bool = False,
                   beep_frequency: int = 1200, beep_duration_ms: int = 180,
                   beep_count: int = 1) -> dict[str, Any]:
    """Capture a frame (or use `image`) and return a natural-language description of what
    is in it — scene, dominant colours, detected objects/regions and any barcodes — using
    the img2nl engine when available, otherwise a basic Pillow summary."""
    out_dir = os.path.expanduser(output_dir) if output_dir else tempfile.mkdtemp(prefix="urirun-camera-")
    os.makedirs(out_dir, exist_ok=True)
    if image:
        photo = os.path.expanduser(image)
        if not os.path.isfile(photo):
            return urirun.fail(f"image not found: {photo}", connector=CONNECTOR_ID)
        device_used = ""
        beep_result = {"ok": True, "enabled": False, "reason": "existing image supplied"}
    else:
        device_used = _device_path(device) or _default_device()
        if not device_used:
            return urirun.fail("no camera device found (no /dev/video*)", connector=CONNECTOR_ID)
        beep_result = _audio_beep(beep, frequency=beep_frequency, duration_ms=beep_duration_ms, count=beep_count)
        if beep_required and not beep_result.get("ok"):
            return urirun.fail(str(beep_result.get("error", "pre-scan beep failed")),
                               connector=CONNECTOR_ID, device=device_used, beep=beep_result)
        photo = os.path.join(out_dir, "photo.jpg")
        cap = _capture(device_used, photo, backend=backend, warmup=warmup, width=width,
                       height=height, timeout=timeout)
        if not cap.get("ok"):
            return urirun.fail(str(cap.get("error", "capture failed")),
                               connector=CONNECTOR_ID, device=device_used, backend=cap.get("backend"),
                               attempts=cap.get("attempts", []))
    w, h = _image_size(photo)
    return _tag(urirun.ok(connector=CONNECTOR_ID, device=device_used,
                     photo={"path": photo, "width": w, "height": h, "bytes": os.path.getsize(photo)},
                     beep=beep_result,
                     description=_describe(photo)), "description")


@CAMERA.handler("photo/query/ocr", isolated=True,
                meta={"label": "Capture a photo and OCR it", "cliAlias": "ocr"})
def photo_ocr(device: str = "", image: str = "", backend: str = "auto", warmup: int = 4,
              crop: bool = False, target: str = "auto", deskew: bool = False, lang: str = "eng+pol",
              max_chars: int = 12000, psm: int = 3, timeout: int = 30, beep: bool = False,
              beep_required: bool = False, beep_frequency: int = 1200,
              beep_duration_ms: int = 180, beep_count: int = 1) -> dict[str, Any]:
    """Convenience route: capture a frame (or use `image`) and return only the OCR text.
    crop=True first crops to the `target` (document/receipt, object, or auto) before reading;
    deskew=True perspective-corrects a document/receipt shot at an angle first."""
    res = analyze(device=device, image=image, backend=backend, warmup=warmup, crop=crop,
                  target=target, deskew=deskew, ocr=True, lang=lang, max_chars=max_chars, psm=psm,
                  timeout=timeout, beep=beep, beep_required=beep_required, beep_frequency=beep_frequency,
                  beep_duration_ms=beep_duration_ms, beep_count=beep_count)
    if not res.get("ok"):
        return res
    ocr = res.get("ocr", {})
    return urirun.ok(connector=CONNECTOR_ID, device=res.get("device", ""),
                     photo=res.get("photo", {}).get("path", ""), target=ocr.get("target", ""),
                     beep=res.get("beep", {}),
                     backend=ocr.get("backend", ""), text=ocr.get("text", ""), chars=ocr.get("chars", 0)), "text") if False else _tag(urirun.ok(connector=CONNECTOR_ID, device=res.get("device", ""), photo=res.get("photo", {}).get("path", ""), target=ocr.get("target", ""), beep=res.get("beep", {}), backend=ocr.get("backend", ""), text=ocr.get("text", ""), chars=ocr.get("chars", 0)), "text")


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower() if needle else True


@CAMERA.handler("photo/query/inspect", isolated=True,
                meta={"label": "Capture and inspect a photo with optional alert", "cliAlias": "inspect"})
def inspect_photo(
    device: str = "",
    image: str = "",
    output_dir: str = "",
    backend: str = "auto",
    warmup: int = 4,
    crop: bool = True,
    target: str = "auto",
    deskew: bool = False,
    lang: str = "eng+pol",
    required_text: str = "",
    forbidden_text: str = "",
    min_chars: int = 1,
    require_object: bool = False,
    brightness_min: float = -1,
    brightness_max: float = -1,
    fail_on_alert: bool = False,
    beep: bool = True,
    beep_on_alert: bool = False,
    beep_frequency: int = 1200,
    alert_beep_frequency: int = 440,
    audit_log: str = "",
    timeout: int = 30,
) -> dict[str, Any]:
    """Capture/analyze/OCR and evaluate simple inspection rules.

    Alerts are returned as structured data. Set fail_on_alert=True when a flow should stop
    on a failed inspection; otherwise the route stays ok=True and reports passed=False.
    Set audit_log to a path to append a one-line JSON verdict per scan (alert history),
    and a full inspection.json sidecar is written next to the photo. `target` controls the
    crop ('document'/'receipt' for paragons, 'object', or 'auto').
    """
    res = analyze(
        device=device,
        image=image,
        output_dir=output_dir,
        backend=backend,
        warmup=warmup,
        crop=crop,
        target=target,
        deskew=deskew,
        ocr=True,
        describe=True,
        lang=lang,
        timeout=timeout,
        beep=beep,
        beep_frequency=beep_frequency,
    )
    if not res.get("ok"):
        return res

    alerts: list[dict[str, Any]] = []
    text = str((res.get("ocr") or {}).get("text") or "")
    if required_text and not _contains(text, required_text):
        alerts.append({"code": "TEXT_MISSING", "message": f"required text not found: {required_text}"})
    if forbidden_text and _contains(text, forbidden_text):
        alerts.append({"code": "FORBIDDEN_TEXT", "message": f"forbidden text found: {forbidden_text}"})
    if int(min_chars) > 0 and len(text.strip()) < int(min_chars):
        alerts.append({"code": "LOW_TEXT", "message": f"OCR text shorter than {min_chars} chars", "chars": len(text.strip())})
    if require_object and not bool((res.get("object") or {}).get("found")):
        alerts.append({"code": "OBJECT_MISSING", "message": "dominant object was not detected"})

    basic = _describe_basic((res.get("photo") or {}).get("path", ""))
    brightness = basic.get("brightness")
    if isinstance(brightness, (int, float)):
        if brightness_min >= 0 and brightness < brightness_min:
            alerts.append({"code": "TOO_DARK", "message": f"brightness {brightness} < {brightness_min}", "brightness": brightness})
        if brightness_max >= 0 and brightness > brightness_max:
            alerts.append({"code": "TOO_BRIGHT", "message": f"brightness {brightness} > {brightness_max}", "brightness": brightness})

    alert_beep = _audio_beep(bool(beep_on_alert and alerts), frequency=alert_beep_frequency,
                             duration_ms=220, count=2)
    inspection = {
        "passed": not alerts,
        "alerts": alerts,
        "requiredText": required_text,
        "forbiddenText": forbidden_text,
        "textChars": len(text.strip()),
        "brightness": brightness,
        "alertBeep": alert_beep,
        "timestamp": time.time(),
    }
    payload = {k: v for k, v in res.items() if k not in ("ok", "connector")}

    # Persist the verdict so alerting/inspection has a durable trail without needing a
    # separate log connector: a JSON sidecar next to the photo and an optional JSONL log.
    out_dir = res.get("outputDir", "")
    if out_dir:
        try:
            inspection["sidecar"] = _write_sidecar(out_dir, {"inspection": inspection, **payload})
        except OSError as exc:
            inspection["sidecarError"] = str(exc)
    if audit_log:
        record = {"timestamp": inspection["timestamp"], "passed": inspection["passed"],
                  "alerts": [a["code"] for a in alerts], "textChars": inspection["textChars"],
                  "brightness": brightness, "device": res.get("device", ""),
                  "photo": (res.get("photo") or {}).get("path", "")}
        try:
            inspection["auditLog"] = _append_jsonl(audit_log, record)
        except OSError as exc:
            inspection["auditLogError"] = str(exc)

    _ledger("inspect", passed=inspection["passed"], alerts=[a["code"] for a in alerts],
            textChars=inspection["textChars"], device=res.get("device", ""),
            photo=(res.get("photo") or {}).get("path", ""))
    if alerts and fail_on_alert:
        return urirun.fail("inspection failed", connector=CONNECTOR_ID, inspection=inspection, **payload)
    return _tag(urirun.ok(connector=CONNECTOR_ID, inspection=inspection, **payload), "inspection")


@CAMERA.handler("photo/query/compare", isolated=True,
                meta={"label": "Detect change/motion between two frames", "cliAlias": "compare"})
def compare(device: str = "", reference: str = "", image: str = "", output_dir: str = "",
            backend: str = "auto", warmup: int = 4, interval_ms: int = 600,
            change_threshold: float = 0.02, pixel_threshold: int = 25, downscale: int = 320,
            beep: bool = False, beep_on_change: bool = False, beep_frequency: int = 1200,
            change_beep_frequency: int = 660, fail_on_change: bool = False,
            timeout: int = 30) -> dict[str, Any]:
    """Change / motion detection. Three modes:
      * reference + image  -> compare those two files;
      * reference only     -> capture one frame and compare it to the reference;
      * neither            -> capture two frames `interval_ms` apart and compare (motion).
    Returns changeRatio, a `changed` boolean (ratio >= change_threshold) and the changed
    region — the trigger for 'scan/alert only when something appears or moves'. Set
    beep_on_change / fail_on_change to alert or stop a flow on change."""
    out_dir = os.path.expanduser(output_dir) if output_dir else tempfile.mkdtemp(prefix="urirun-camera-")
    os.makedirs(out_dir, exist_ok=True)
    device_used = ""
    captured: list[str] = []

    def _grab(name: str) -> dict[str, Any]:
        nonlocal device_used
        device_used = _device_path(device) or _default_device()
        if not device_used:
            return {"ok": False, "error": "no camera device found (no /dev/video*)"}
        dst = os.path.join(out_dir, name)
        cap = _capture(device_used, dst, backend=backend, warmup=warmup, width=0, height=0, timeout=timeout)
        if cap.get("ok"):
            captured.append(dst)
        return cap

    if reference and image:                       # two supplied files
        path_a, path_b = os.path.expanduser(reference), os.path.expanduser(image)
        for p in (path_a, path_b):
            if not os.path.isfile(p):
                return urirun.fail(f"image not found: {p}", connector=CONNECTOR_ID)
    elif reference:                               # reference vs a fresh frame
        path_a = os.path.expanduser(reference)
        if not os.path.isfile(path_a):
            return urirun.fail(f"reference not found: {path_a}", connector=CONNECTOR_ID)
        beep_result = _audio_beep(beep, frequency=beep_frequency)
        cap = _grab("current.jpg")
        if not cap.get("ok"):
            return urirun.fail(str(cap.get("error", "capture failed")), connector=CONNECTOR_ID,
                               beep=beep_result, backend=cap.get("backend"), attempts=cap.get("attempts", []))
        path_b = cap["path"]
    else:                                         # two-shot live motion detection
        beep_result = _audio_beep(beep, frequency=beep_frequency)
        cap_a = _grab("frame_a.jpg")
        if not cap_a.get("ok"):
            return urirun.fail(str(cap_a.get("error", "capture failed")), connector=CONNECTOR_ID,
                               backend=cap_a.get("backend"), attempts=cap_a.get("attempts", []))
        if interval_ms > 0:
            time.sleep(min(interval_ms, 10000) / 1000.0)
        cap_b = _grab("frame_b.jpg")
        if not cap_b.get("ok"):
            return urirun.fail(str(cap_b.get("error", "capture failed")), connector=CONNECTOR_ID,
                               backend=cap_b.get("backend"), attempts=cap_b.get("attempts", []))
        path_a, path_b = cap_a["path"], cap_b["path"]

    diff = _frame_diff(path_a, path_b, pixel_threshold=pixel_threshold, downscale=downscale)
    if not diff.get("ok"):
        return urirun.fail(str(diff.get("error", "diff failed")), connector=CONNECTOR_ID)

    changed = diff["changeRatio"] >= change_threshold
    change_beep = _audio_beep(bool(beep_on_change and changed), frequency=change_beep_frequency,
                              duration_ms=200, count=2)
    payload = {"device": device_used, "frames": {"a": path_a, "b": path_b},
               "changed": changed, "changeThreshold": change_threshold,
               "changeBeep": change_beep, **{k: v for k, v in diff.items() if k != "ok"}}
    if changed and fail_on_change:
        return urirun.fail("change detected", connector=CONNECTOR_ID, **payload)
    return _tag(urirun.ok(connector=CONNECTOR_ID, **payload), "comparison")


@CAMERA.handler("photo/query/barcodes", isolated=True,
                meta={"label": "Capture and decode barcodes / QR codes", "cliAlias": "barcodes"})
def read_barcodes(device: str = "", image: str = "", output_dir: str = "", backend: str = "auto",
                  warmup: int = 4, required: str = "", beep: bool = False, beep_required: bool = False,
                  beep_on_read: bool = False, beep_frequency: int = 1200,
                  read_beep_frequency: int = 880, fail_if_missing: bool = False,
                  timeout: int = 30) -> dict[str, Any]:
    """Capture a frame (or use `image`) and decode any barcodes / QR codes in it (pyzbar,
    img2nl fallback). Returns every code with its type, data and bounding rect. `required`
    filters/asserts a substring is present in some code; beep_on_read beeps when at least
    one code is found, and fail_if_missing stops a flow when none (or the required one) is
    seen — the trigger for 'scan the label, alert if the expected code is absent'."""
    out_dir = os.path.expanduser(output_dir) if output_dir else tempfile.mkdtemp(prefix="urirun-camera-")
    os.makedirs(out_dir, exist_ok=True)
    device_used = ""
    if image:
        photo = os.path.expanduser(image)
        if not os.path.isfile(photo):
            return urirun.fail(f"image not found: {photo}", connector=CONNECTOR_ID)
        beep_result = {"ok": True, "enabled": False, "reason": "existing image supplied"}
    else:
        device_used = _device_path(device) or _default_device()
        if not device_used:
            return urirun.fail("no camera device found (no /dev/video*)", connector=CONNECTOR_ID)
        beep_result = _audio_beep(beep, frequency=beep_frequency)
        if beep_required and not beep_result.get("ok"):
            return urirun.fail(str(beep_result.get("error", "pre-scan beep failed")),
                               connector=CONNECTOR_ID, device=device_used, beep=beep_result)
        photo = os.path.join(out_dir, "photo.jpg")
        cap = _capture(device_used, photo, backend=backend, warmup=warmup, width=0, height=0, timeout=timeout)
        if not cap.get("ok"):
            return urirun.fail(str(cap.get("error", "capture failed")), connector=CONNECTOR_ID,
                               device=device_used, backend=cap.get("backend"), attempts=cap.get("attempts", []))

    result = _decode_barcodes(photo)
    codes = result.get("codes", [])
    matched = [c for c in codes if required.lower() in str(c.get("data", "")).lower()] if required else codes
    found = bool(matched)
    read_beep = _audio_beep(bool(beep_on_read and found), frequency=read_beep_frequency,
                            duration_ms=160, count=2)
    payload = {"device": device_used, "photo": photo, "barcodeBackend": result.get("backend"),
               "count": len(codes), "codes": codes, "required": required,
               "matched": matched if required else None, "found": found,
               "beep": beep_result, "readBeep": read_beep}
    if not result.get("ok") and result.get("error"):
        payload["decodeError"] = result["error"]
    if fail_if_missing and not found:
        reason = f"required barcode not found: {required}" if required else "no barcode detected"
        return urirun.fail(reason, connector=CONNECTOR_ID, **payload)
    return _tag(urirun.ok(connector=CONNECTOR_ID, **payload), "barcodes")


def _decode_b64_to_file(bytes_b64: str, out_dir: str, filename: str,
                        max_input_bytes: int) -> tuple[str, int]:
    """Decode a base64 (optionally data-URL) image into a file, returning (path, bytes).
    Raises ValueError on bad input or oversize payloads."""
    import binascii
    payload = bytes_b64.strip()
    if payload.startswith("data:"):                     # strip a data: URL prefix
        payload = payload.split(",", 1)[-1]
    try:
        raw = base64.b64decode(payload.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError) as exc:
        raise ValueError(f"invalid bytes_b64: {exc}") from exc
    if not raw:
        raise ValueError("empty image payload")
    if len(raw) > max_input_bytes:
        raise ValueError(f"image exceeds max_input_bytes ({len(raw)} > {max_input_bytes})")
    suffix = os.path.splitext(filename or "photo.jpg")[1].lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        suffix = ".jpg"
    os.makedirs(os.path.expanduser(out_dir), exist_ok=True)
    path = os.path.join(os.path.expanduser(out_dir), f"photo{suffix}")
    with open(path, "wb") as fh:
        fh.write(raw)
    return path, len(raw)


@CAMERA.handler("upload/command/ingest", isolated=True,
                meta={"label": "Process a browser/mobile-uploaded frame", "cliAlias": "ingest"})
def ingest(bytes_b64: str = "", filename: str = "photo.jpg", action: str = "analyze",
           output_dir: str = "", lang: str = "eng+pol", crop: bool = True, target: str = "auto",
           deskew: bool = False, required_text: str = "", forbidden_text: str = "", min_chars: int = 1,
           require_object: bool = False, required: str = "", fail_if_missing: bool = False,
           max_chars: int = 12000, audit_log: str = "",
           max_input_bytes: int = 20 * 1024 * 1024, timeout: int = 30) -> dict[str, Any]:
    """Run the camera pipeline on a frame uploaded as base64 — the entry point for a phone
    or tablet capturing through the browser (getUserMedia) instead of a local /dev/video*.
    `action` selects what to do: analyze | inspect | barcodes | describe | ocr. `target`
    chooses the crop (document/receipt, object, or auto). No capture or beep happens here;
    the bytes are decoded to a file and passed to the matching route."""
    if not bytes_b64:
        return urirun.fail("bytes_b64 is required", connector=CONNECTOR_ID)
    out_dir = os.path.expanduser(output_dir) if output_dir else tempfile.mkdtemp(prefix="urirun-camera-upload-")
    try:
        photo, size = _decode_b64_to_file(bytes_b64, out_dir, filename, max_input_bytes)
    except ValueError as exc:
        return urirun.fail(str(exc), connector=CONNECTOR_ID)

    act = (action or "analyze").strip().lower()
    if act == "analyze":
        res = analyze(image=photo, output_dir=out_dir, crop=crop, target=target, deskew=deskew,
                      ocr=True, describe=True, lang=lang, max_chars=max_chars, timeout=timeout)
    elif act == "inspect":
        res = inspect_photo(image=photo, output_dir=out_dir, crop=crop, target=target, deskew=deskew,
                            lang=lang, required_text=required_text, forbidden_text=forbidden_text,
                            min_chars=min_chars, require_object=require_object,
                            beep=False, audit_log=audit_log, timeout=timeout)
    elif act == "barcodes":
        res = read_barcodes(image=photo, output_dir=out_dir, required=required,
                            fail_if_missing=fail_if_missing)
    elif act == "describe":
        res = describe_photo(image=photo, output_dir=out_dir)
    elif act == "ocr":
        res = photo_ocr(image=photo, crop=crop, target=target, deskew=deskew, lang=lang,
                        max_chars=max_chars, timeout=timeout)
    elif act in ("receipt", "parse"):
        res = receipt_parse(image=photo, output_dir=out_dir, target=target, deskew=deskew,
                            lang=lang, max_chars=max_chars, timeout=timeout)
    else:
        return urirun.fail(f"unknown action: {action} (use analyze|inspect|barcodes|describe|ocr|receipt)",
                           connector=CONNECTOR_ID)

    value = res.get("result", res) if isinstance(res, dict) else {}
    if isinstance(res, dict):
        res.setdefault("source", "browser-upload")
        res["action"] = act
        res["uploadBytes"] = size
        res["photo"] = res.get("photo") or photo
    # inspect/receipt already logged their own line; record the rest of the mobile uploads here
    if act in ("analyze", "barcodes", "ocr", "describe"):
        _ledger("ingest", action=act, source="browser-upload", uploadBytes=size,
                ok=bool(isinstance(res, dict) and res.get("ok", True)))
    return res


# ------------------------------------------------------- receipt parsing (paragon → JSON)

_PRICE_RE = re.compile(r"(\d{1,3}(?:[  ]\d{3})*|\d+)[.,](\d{2})(?!\d)")
_TOTAL_KEYS = ("suma", "razem", "do zaplaty", "do zaplaty", "total", "lacznie", "summa", "naleznosc")
_DATE_RE = re.compile(r"(\d{4}[-./]\d{2}[-./]\d{2}|\d{2}[-./]\d{2}[-./]\d{4})")
_NIP_RE = re.compile(r"NIP[:\s]*([0-9][0-9\- \t]{8,14})", re.IGNORECASE)


def _fold(text: str) -> str:
    """Lowercase and strip Polish diacritics so keyword matching is robust to OCR/locale."""
    table = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    return text.translate(table).lower()


def _to_amount(whole: str, cents: str) -> float:
    return round(int(whole.replace(" ", "").replace(" ", "")) + int(cents) / 100.0, 2)


def _parse_receipt(text: str) -> dict[str, Any]:
    """Turn raw receipt OCR text into structured data: line items (name + price), the total,
    currency, date and NIP. Heuristic and locale-tolerant (PL/EN) — every field is best
    effort and may be null when the print/OCR is too noisy."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    items: list[dict[str, Any]] = []
    total: float | None = None
    for line in lines:
        prices = _PRICE_RE.findall(line)
        if not prices:
            continue
        amount = _to_amount(*prices[-1])           # the rightmost money token on the line
        folded = _fold(line)
        if any(key in folded for key in _TOTAL_KEYS):
            total = amount                          # a total/sum line, not a product
            continue
        name = _PRICE_RE.sub("", line).strip(" .:-xX*\t")
        name = re.sub(r"\s{2,}", " ", name)
        if len(name) >= 2:
            items.append({"name": name, "price": amount})

    currency = None
    for cur, pat in (("PLN", r"\bpln\b|z[lł]\b"), ("EUR", r"\beur\b|€"), ("USD", r"\busd\b|\$")):
        if re.search(pat, _fold(text)):
            currency = cur
            break
    date_m = _DATE_RE.search(text or "")
    nip_m = _NIP_RE.search(text or "")
    items_sum = round(sum(i["price"] for i in items), 2)
    total_source = "total-line" if total is not None else None
    if total is None and items:                     # fall back to the largest amount seen
        total = max(i["price"] for i in items)
        total_source = "max-item"
    return {
        "items": items,
        "itemCount": len(items),
        "total": total,
        "totalSource": total_source,
        "itemsSum": items_sum,
        "currency": currency,
        "date": date_m.group(1) if date_m else None,
        "nip": re.sub(r"\D", "", nip_m.group(1))[:10] if nip_m else None,
        "lines": len(lines),
    }


@CAMERA.handler("receipt/query/parse", isolated=True,
                meta={"label": "Scan a receipt and parse items/total to JSON", "cliAlias": "receipt"})
def receipt_parse(device: str = "", image: str = "", bytes_b64: str = "", text: str = "",
                  output_dir: str = "", lang: str = "pol+eng", target: str = "receipt",
                  deskew: bool = True, max_chars: int = 12000,
                  max_input_bytes: int = 20 * 1024 * 1024, timeout: int = 30) -> dict[str, Any]:
    """Read a receipt ('paragon') and return structured data — line items (name + price),
    total, currency, date, NIP. Give `text` to parse an existing OCR string, or a frame
    source (`image`, `bytes_b64`, or capture from `device`) to scan it first: the frame is
    cropped to the sheet (`target=receipt`) and deskewed before OCR, then parsed."""
    if text.strip():
        parsed = _parse_receipt(text)
        _ledger("receipt", source="text", total=parsed.get("total"), currency=parsed.get("currency"),
                itemCount=parsed.get("itemCount"), nip=parsed.get("nip"))
        return _tag(urirun.ok(connector=CONNECTOR_ID, source="text", **parsed), "receipt")

    out_dir = os.path.expanduser(output_dir) if output_dir else tempfile.mkdtemp(prefix="urirun-receipt-")
    os.makedirs(out_dir, exist_ok=True)
    if bytes_b64:
        try:
            photo, _size = _decode_b64_to_file(bytes_b64, out_dir, "receipt.jpg", max_input_bytes)
        except ValueError as exc:
            return urirun.fail(str(exc), connector=CONNECTOR_ID)
        res = analyze(image=photo, output_dir=out_dir, target=target, deskew=deskew, ocr=True,
                      describe=False, lang=lang, max_chars=max_chars, timeout=timeout)
    elif image or device:
        res = analyze(device=device, image=image, output_dir=out_dir, target=target, deskew=deskew,
                      ocr=True, describe=False, lang=lang, max_chars=max_chars, timeout=timeout)
    else:
        return urirun.fail("provide text, image, bytes_b64 or device", connector=CONNECTOR_ID)

    if not res.get("ok"):
        return res
    ocr_text = str((res.get("ocr") or {}).get("text") or "")
    parsed = _parse_receipt(ocr_text)
    _ledger("receipt", source="ocr", total=parsed.get("total"), currency=parsed.get("currency"),
            itemCount=parsed.get("itemCount"), nip=parsed.get("nip"),
            photo=(res.get("photo") or {}).get("path", ""))
    return urirun.ok(connector=CONNECTOR_ID, source="ocr",
                     photo=(res.get("photo") or {}).get("path", ""),
                     ocrBackend=(res.get("ocr") or {}).get("backend", ""),
                     object=res.get("object"), text=ocr_text[:max_chars], **parsed), "receipt")


def urirun_bindings() -> dict[str, Any]:
    """Serializable v2 bindings for this connector."""
    return CAMERA.bindings()


def connector_manifest() -> dict[str, Any]:
    """Full manifest: prose plus derived routes."""
    return CAMERA.manifest(urirun.load_manifest(__package__))


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point."""
    return CAMERA.cli(argv, manifest_prose=urirun.load_manifest(__package__))


if __name__ == "__main__":
    raise SystemExit(main())
