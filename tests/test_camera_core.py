"""Tests for the camera connector. The capture path needs real hardware, so it is gated
behind URIRUN_CAMERA_LIVE=1; the detect/crop/OCR pipeline is tested offline by feeding a
synthetic image through analyze(image=...)."""
import os
import shutil
import pytest

import urirun_connector_camera.core as c

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

HAS_TESS = bool(shutil.which("tesseract"))
LIVE = os.environ.get("URIRUN_CAMERA_LIVE") == "1"


def test_bindings_valid():
    b = c.urirun_bindings()
    assert set(b["bindings"]) == {
        "camera://host/devices/query/list",
        "camera://host/photo/command/capture",
        "camera://host/photo/query/analyze",
        "camera://host/photo/query/describe",
        "camera://host/photo/query/ocr",
    }


@pytest.mark.parametrize("raw,expected", [
    ("0", "/dev/video0"),
    ("video1", "/dev/video1"),
    ("/dev/video2", "/dev/video2"),
    ("", ""),
])
def test_device_path(raw, expected):
    assert c._device_path(raw) == expected


def test_list_cameras_runs():
    r = c.list_cameras()
    assert r["ok"] and "videoNodes" in r and "capture" in r


def _make_image(path, *, text="INVOICE 12345", bg="white"):
    img = Image.new("RGB", (640, 480), bg)
    d = ImageDraw.Draw(img)
    # a framed dark rectangle holding text → a clear dominant object in the frame
    d.rectangle([180, 170, 470, 320], outline="black", width=4)
    d.text((205, 230), text, fill="black")
    img.save(path)


def test_main_bbox_finds_object(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p)
    det = c._main_bbox(p, edge_threshold=40, pad=0.04, min_fraction=0.02)
    assert det["ok"] and det["found"]
    x, y, w, h = det["bbox"]
    # the detected region should be inside the frame and well smaller than the full image
    assert w < 640 and h < 480 and w > 0 and h > 0


def test_analyze_image_crops(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p)
    out = str(tmp_path / "out")
    r = c.analyze(image=p, output_dir=out, ocr=False)
    assert r["ok"] and r["object"]["found"]
    assert os.path.isfile(r["object"]["cropPath"])


@pytest.mark.skipif(not HAS_TESS, reason="tesseract not installed")
def test_analyze_image_ocr_reads_text(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p, text="HELLO 2026")
    out = str(tmp_path / "out")
    r = c.analyze(image=p, output_dir=out, crop=True, ocr=True, lang="eng")
    assert r["ok"]
    text = (r.get("ocr", {}).get("text") or "").upper()
    assert "HELLO" in text or "2026" in text


def test_analyze_missing_image_fails():
    r = c.analyze(image="/no/such/file.png")
    assert not r["ok"]


def test_describe_basic_fallback_always_works(tmp_path):
    # Without img2nl the description must still come back via the Pillow fallback.
    p = str(tmp_path / "scene.png")
    _make_image(p)
    d = c._describe_basic(p)
    assert d["text"] and d["dominantColors"] and "brightness" in d


def test_describe_photo_on_image(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p)
    r = c.describe_photo(image=p, output_dir=str(tmp_path / "out"))
    assert r["ok"] and r["description"]["text"]
    assert r["description"]["backend"] in ("img2nl", "pillow")


@pytest.mark.skipif(not (LIVE and c._default_device()), reason="set URIRUN_CAMERA_LIVE=1 with a real camera")
def test_capture_live(tmp_path):
    out = str(tmp_path / "live.jpg")
    r = c.capture(output=out)
    assert r["ok"] and os.path.isfile(out) and r["width"] > 0
