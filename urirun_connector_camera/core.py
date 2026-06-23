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
import os
import shutil
import subprocess
import tempfile
from typing import Any

import urirun

CONNECTOR_ID = "camera"
CAMERA = urirun.connector(CONNECTOR_ID, scheme="camera", target="host", meta={"label": "Camera capture + OCR"})


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
    if selected in ("ffmpeg", "auto"):
        result = _capture_ffmpeg(device, out_path, warmup=warmup, width=width, height=height, timeout=timeout)
        if result.get("ok") or selected == "ffmpeg":
            return result
        return _capture_cv2(device, out_path, warmup=warmup, width=width, height=height)
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
            max_base64_bytes: int = 8 * 1024 * 1024, timeout: int = 30) -> dict[str, Any]:
    """Capture one still frame from a camera and save it to `output` (default: a temp jpg).
    `device` accepts '/dev/video0', 'video0' or '0' (defaults to the first camera).
    Set return_base64=True to also get the image bytes inline."""
    dev = _device_path(device) or _default_device()
    if not dev:
        return urirun.fail("no camera device found (no /dev/video*)", connector=CONNECTOR_ID)
    out = os.path.expanduser(output) if output else os.path.join(
        tempfile.gettempdir(), f"urirun-camera-{os.path.basename(dev)}.jpg")
    result = _capture(dev, out, backend=backend, warmup=warmup, width=width, height=height, timeout=timeout)
    if not result.get("ok"):
        return urirun.fail(str(result.get("error", "capture failed")),
                           connector=CONNECTOR_ID, device=dev, backend=result.get("backend"))
    w, h = _image_size(out)
    payload: dict[str, Any] = {"device": dev, "backend": result.get("backend"), "path": out,
                               "width": w, "height": h, "bytes": os.path.getsize(out)}
    if return_base64:
        b64, size = _b64_file(out, max_base64_bytes)
        payload["bytes_b64"] = b64
        if not b64:
            payload["base64_skipped"] = f"image is {size} bytes > max_base64_bytes"
    return urirun.ok(connector=CONNECTOR_ID, **payload)


@CAMERA.handler("photo/query/analyze", isolated=True,
                meta={"label": "Capture, crop to the main object, and OCR it", "cliAlias": "analyze"})
def analyze(device: str = "", image: str = "", output_dir: str = "", backend: str = "auto",
            warmup: int = 4, width: int = 0, height: int = 0, crop: bool = True,
            ocr: bool = True, describe: bool = True, lang: str = "eng+pol",
            max_chars: int = 12000, psm: int = 3, edge_threshold: int = 40, pad: float = 0.04,
            min_fraction: float = 0.02, timeout: int = 30) -> dict[str, Any]:
    """End-to-end: take a photo (or use an existing `image`), describe what is in it
    (img2nl scene/objects when available), find the dominant object, crop to it, and OCR
    the crop to read its text. Returns the photo path, a natural-language description, the
    detected object bbox + crop path, and the OCR text. crop=False OCRs the whole frame,
    ocr=False skips reading, describe=False skips scene understanding."""
    out_dir = os.path.expanduser(output_dir) if output_dir else tempfile.mkdtemp(prefix="urirun-camera-")
    os.makedirs(out_dir, exist_ok=True)

    # 1. obtain a frame (capture or reuse a provided image)
    if image:
        photo = os.path.expanduser(image)
        if not os.path.isfile(photo):
            return urirun.fail(f"image not found: {photo}", connector=CONNECTOR_ID)
        device_used = ""
    else:
        device_used = _device_path(device) or _default_device()
        if not device_used:
            return urirun.fail("no camera device found (no /dev/video*)", connector=CONNECTOR_ID)
        photo = os.path.join(out_dir, "photo.jpg")
        cap = _capture(device_used, photo, backend=backend, warmup=warmup, width=width,
                       height=height, timeout=timeout)
        if not cap.get("ok"):
            return urirun.fail(str(cap.get("error", "capture failed")),
                               connector=CONNECTOR_ID, device=device_used, backend=cap.get("backend"))

    w, h = _image_size(photo)
    report: dict[str, Any] = {
        "device": device_used,
        "photo": {"path": photo, "width": w, "height": h, "bytes": os.path.getsize(photo)},
    }

    # 2. describe what is in the photo (img2nl scene/objects, else Pillow stats)
    if describe:
        report["description"] = _describe(photo)

    # 3. find the dominant object and crop to it
    ocr_target = photo
    if crop:
        det = _object_bbox(photo, edge_threshold=edge_threshold, pad=pad, min_fraction=min_fraction)
        if det.get("ok"):
            obj: dict[str, Any] = {"found": det.get("found"), "bbox": det.get("bbox"),
                                   "coverage": det.get("coverage"), "detector": det.get("detector")}
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

    return urirun.ok(connector=CONNECTOR_ID, outputDir=out_dir, **report)


@CAMERA.handler("photo/query/describe", isolated=True,
                meta={"label": "Capture a photo and describe what is in it", "cliAlias": "describe"})
def describe_photo(device: str = "", image: str = "", backend: str = "auto", warmup: int = 4,
                   width: int = 0, height: int = 0, output_dir: str = "",
                   timeout: int = 30) -> dict[str, Any]:
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
    else:
        device_used = _device_path(device) or _default_device()
        if not device_used:
            return urirun.fail("no camera device found (no /dev/video*)", connector=CONNECTOR_ID)
        photo = os.path.join(out_dir, "photo.jpg")
        cap = _capture(device_used, photo, backend=backend, warmup=warmup, width=width,
                       height=height, timeout=timeout)
        if not cap.get("ok"):
            return urirun.fail(str(cap.get("error", "capture failed")),
                               connector=CONNECTOR_ID, device=device_used, backend=cap.get("backend"))
    w, h = _image_size(photo)
    return urirun.ok(connector=CONNECTOR_ID, device=device_used,
                     photo={"path": photo, "width": w, "height": h, "bytes": os.path.getsize(photo)},
                     description=_describe(photo))


@CAMERA.handler("photo/query/ocr", isolated=True,
                meta={"label": "Capture a photo and OCR it", "cliAlias": "ocr"})
def photo_ocr(device: str = "", image: str = "", backend: str = "auto", warmup: int = 4,
              crop: bool = False, lang: str = "eng+pol", max_chars: int = 12000,
              psm: int = 3, timeout: int = 30) -> dict[str, Any]:
    """Convenience route: capture a frame (or use `image`) and return only the OCR text.
    crop=True first crops to the dominant object before reading it."""
    res = analyze(device=device, image=image, backend=backend, warmup=warmup, crop=crop,
                  ocr=True, lang=lang, max_chars=max_chars, psm=psm, timeout=timeout)
    if not res.get("ok"):
        return res
    ocr = res.get("ocr", {})
    return urirun.ok(connector=CONNECTOR_ID, device=res.get("device", ""),
                     photo=res.get("photo", {}).get("path", ""), target=ocr.get("target", ""),
                     backend=ocr.get("backend", ""), text=ocr.get("text", ""), chars=ocr.get("chars", 0))


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
