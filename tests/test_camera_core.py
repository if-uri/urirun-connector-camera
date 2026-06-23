"""Tests for the camera connector. The capture path needs real hardware, so it is gated
behind URIRUN_CAMERA_LIVE=1; the detect/crop/OCR pipeline is tested offline by feeding a
synthetic image through analyze(image=...)."""
import json
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
        "camera://host/photo/query/inspect",
        "camera://host/photo/query/compare",
        "camera://host/photo/query/barcodes",
        "camera://host/photo/query/ocr",
        "camera://host/receipt/query/parse",
        "camera://host/upload/command/ingest",
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


def test_capture_beep_failure_can_block(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "_default_device", lambda: "/dev/video0")
    monkeypatch.setattr(c, "_audio_beep", lambda *a, **k: {"ok": False, "enabled": True, "error": "no audio"})

    r = c.capture(output=str(tmp_path / "shot.jpg"), beep=True, beep_required=True)

    assert not r["ok"]
    assert r["error"] == "no audio"
    assert r["beep"]["enabled"] is True


def test_capture_auto_preserves_failed_backend_attempts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        c,
        "_capture_ffmpeg",
        lambda *a, **k: {"ok": False, "backend": "ffmpeg", "error": "v4l2 device busy"},
    )
    monkeypatch.setattr(
        c,
        "_capture_cv2",
        lambda *a, **k: {"ok": False, "backend": "cv2", "error": "opencv unavailable"},
    )

    r = c._capture("/dev/video0", str(tmp_path / "shot.jpg"), backend="auto", warmup=1, width=0, height=0, timeout=3)

    assert not r["ok"]
    assert r["backend"] == "auto"
    assert "v4l2 device busy" in r["error"]
    assert "opencv unavailable" in r["error"]
    assert [a["backend"] for a in r["attempts"]] == ["ffmpeg", "cv2"]


def test_capture_route_preserves_attempts_on_failure(monkeypatch, tmp_path):
    attempts = [
        {"ok": False, "backend": "ffmpeg", "error": "device busy"},
        {"ok": False, "backend": "cv2", "error": "opencv unavailable"},
    ]
    monkeypatch.setattr(c, "_default_device", lambda: "/dev/video0")
    monkeypatch.setattr(c, "_capture", lambda *a, **k: {
        "ok": False,
        "backend": "auto",
        "error": "device busy; opencv unavailable",
        "attempts": attempts,
    })

    r = c.capture(output=str(tmp_path / "shot.jpg"), backend="auto")

    assert not r["ok"]
    assert r["backend"] == "auto"
    assert r["attempts"] == attempts


def test_inspect_photo_reports_alerts_without_failing(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p, text="HELLO 2026")

    r = c.inspect_photo(image=p, output_dir=str(tmp_path / "out"), required_text="MISSING", min_chars=1)

    assert r["ok"]
    assert r["inspection"]["passed"] is False
    assert any(a["code"] == "TEXT_MISSING" for a in r["inspection"]["alerts"])


def test_inspect_photo_can_fail_on_alert(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p, text="HELLO 2026")

    r = c.inspect_photo(image=p, output_dir=str(tmp_path / "out"), required_text="MISSING", fail_on_alert=True)

    assert not r["ok"]
    assert r["inspection"]["passed"] is False


def test_inspect_writes_audit_log_and_sidecar(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p, text="HELLO 2026")
    out = str(tmp_path / "out")
    audit = str(tmp_path / "audit.jsonl")

    r = c.inspect_photo(image=p, output_dir=out, required_text="MISSING",
                        min_chars=1, audit_log=audit)

    assert r["ok"] and r["inspection"]["passed"] is False
    # one JSONL verdict line was appended
    lines = [l for l in open(audit, encoding="utf-8").read().splitlines() if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["passed"] is False and "TEXT_MISSING" in rec["alerts"]
    # a full sidecar exists next to the photo
    assert os.path.isfile(os.path.join(out, "inspection.json"))


def _make_receipt(path, *, fg=(250, 250, 250), bg=(70, 80, 75)):
    """A 'paragon': a tall bright sheet centred on a darker desk, with print lines."""
    img = Image.new("RGB", (640, 480), bg)
    d = ImageDraw.Draw(img)
    # receipt occupies roughly the central column, with margins on all sides
    d.rectangle([250, 60, 390, 430], fill=fg)
    for i, y in enumerate(range(90, 410, 26)):
        w = 110 if i % 3 else 70
        d.text((262, y), "ITEM " + str(i) + "  " + str(i * 7) + ",99 PLN", fill=(15, 15, 15))
    img.save(path)


def test_document_bbox_crops_to_receipt(tmp_path):
    pytest.importorskip("numpy")
    p = str(tmp_path / "receipt.png")
    _make_receipt(p)
    det = c._document_bbox(p, pad=0.02)
    assert det["ok"] and det["found"]
    x, y, w, h = det["bbox"]
    # the crop should hug the central sheet: well inside the 640px width, taller than wide
    assert x >= 180 and (x + w) <= 460
    assert h > w                      # receipts are portrait
    assert det["coverage"] < 0.6      # trimmed most of the desk away


def test_target_document_used_by_analyze(tmp_path):
    pytest.importorskip("numpy")
    p = str(tmp_path / "receipt.png")
    _make_receipt(p)
    r = c.analyze(image=p, output_dir=str(tmp_path / "out"), target="document", ocr=False, describe=False)
    assert r["ok"] and r["object"]["found"]
    assert r["object"]["detector"] == "document" and r["object"]["target"] == "document"
    assert os.path.isfile(r["object"]["cropPath"])


def _make_skewed_receipt(path, corners=((230, 70), (430, 110), (400, 420), (210, 380))):
    """A bright sheet drawn as a tilted quadrilateral on a dark desk (angled photo)."""
    img = Image.new("RGB", (640, 480), (60, 70, 65))
    d = ImageDraw.Draw(img)
    d.polygon(list(corners), fill=(248, 248, 248))
    for y in range(140, 360, 30):
        d.text((255, y), "LINE " + str(y), fill=(20, 20, 20))
    img.save(path)
    return corners


def test_perspective_coeffs_roundtrip():
    np = pytest.importorskip("numpy")
    dest = [(0, 0), (100, 0), (100, 50), (0, 50)]
    src = [(10, 5), (110, 12), (105, 60), (8, 58)]
    coeffs = c._perspective_coeffs(dest, src)
    a, b, cc, dd, e, f, g, h = coeffs
    for (x, y), (X, Y) in zip(dest, src):
        denom = g * x + h * y + 1
        assert abs((a * x + b * y + cc) / denom - X) < 1e-6
        assert abs((dd * x + e * y + f) / denom - Y) < 1e-6


def test_deskew_recovers_flat_sheet(tmp_path):
    np = pytest.importorskip("numpy")
    p = str(tmp_path / "skewed.png")
    _make_skewed_receipt(p)
    out = str(tmp_path / "document.jpg")
    r = c._deskew_document(p, out, min_area_ratio=0.03)
    assert r["ok"] and r["found"] and os.path.isfile(r["path"])
    assert r["detector"].endswith("-deskew") and len(r["corners"]) == 4
    # the warped output should be dominated by the bright flattened sheet
    warped = np.asarray(Image.open(r["path"]).convert("L"))
    assert warped.mean() > 150


def test_deskew_ignores_background_speck(tmp_path):
    np = pytest.importorskip("numpy")
    p = str(tmp_path / "skewed.png")
    corners = _make_skewed_receipt(p)
    # add a tiny bright reflection far in the top-right corner — must NOT become a corner
    img = Image.open(p)
    ImageDraw.Draw(img).ellipse([600, 8, 612, 20], fill=(255, 255, 255))
    img.save(p)
    r = c._deskew_document(p, str(tmp_path / "doc.jpg"), min_area_ratio=0.03)
    assert r["ok"] and r["found"]
    # every detected corner stays near the real sheet, not the speck at (~606, ~14)
    for cx, cy in r["corners"]:
        assert not (cx > 520 and cy < 60), f"corner {cx},{cy} latched onto the speck"


def test_analyze_deskew_sets_detector(tmp_path):
    pytest.importorskip("numpy")
    p = str(tmp_path / "skewed.png")
    _make_skewed_receipt(p)
    r = c.analyze(image=p, output_dir=str(tmp_path / "out"), target="receipt",
                  deskew=True, ocr=False, describe=False)
    assert r["ok"] and r["object"]["found"] and r["object"].get("deskewed") is True
    assert r["object"]["detector"].endswith("-deskew")
    assert os.path.isfile(r["object"]["cropPath"]) and len(r["object"]["corners"]) == 4


def test_frame_diff_detects_change(tmp_path):
    a = str(tmp_path / "a.png")
    b = str(tmp_path / "b.png")
    _make_image(a, text="AAAA", bg="white")
    # b is mostly black → large change vs the white frame
    Image.new("RGB", (640, 480), "black").save(b)
    d = c._frame_diff(a, b, pixel_threshold=25, downscale=320)
    assert d["ok"] and d["changeRatio"] > 0.5 and d["changedRegion"]


def test_frame_diff_same_image_is_stable(tmp_path):
    a = str(tmp_path / "a.png")
    _make_image(a, text="AAAA")
    d = c._frame_diff(a, a, pixel_threshold=25, downscale=320)
    assert d["ok"] and d["changeRatio"] == 0.0


def test_compare_two_files_flags_changed(tmp_path):
    a = str(tmp_path / "a.png")
    b = str(tmp_path / "b.png")
    _make_image(a, text="AAAA", bg="white")
    Image.new("RGB", (640, 480), "black").save(b)
    r = c.compare(reference=a, image=b, output_dir=str(tmp_path / "out"), change_threshold=0.02)
    assert r["ok"] and r["changed"] is True

    r2 = c.compare(reference=a, image=a, output_dir=str(tmp_path / "out2"), change_threshold=0.02)
    assert r2["ok"] and r2["changed"] is False


def _make_qr(path, data="https://ifuri.com/INV-2026-001"):
    qrcode = pytest.importorskip("qrcode")
    pytest.importorskip("pyzbar")
    qrcode.make(data).convert("RGB").save(path)


def test_decode_barcodes_reads_qr(tmp_path):
    p = str(tmp_path / "qr.png")
    _make_qr(p, "HELLO-QR-2026")
    r = c._decode_barcodes(p)
    assert r["ok"] and r["count"] >= 1
    assert any(code["data"] == "HELLO-QR-2026" and code["type"] == "QRCODE" for code in r["codes"])
    assert r["codes"][0]["rect"] and len(r["codes"][0]["rect"]) == 4


def test_read_barcodes_route_and_required_match(tmp_path):
    p = str(tmp_path / "qr.png")
    _make_qr(p, "https://ifuri.com/INV-2026-001")
    r = c.read_barcodes(image=p, output_dir=str(tmp_path / "out"), required="INV-2026-001")
    assert r["ok"] and r["found"] is True and r["count"] >= 1
    assert r["matched"] and r["matched"][0]["data"].endswith("INV-2026-001")


def test_read_barcodes_fail_if_missing(tmp_path):
    p = str(tmp_path / "qr.png")
    _make_qr(p, "SOME-OTHER-CODE")
    r = c.read_barcodes(image=p, output_dir=str(tmp_path / "out"),
                        required="EXPECTED-123", fail_if_missing=True)
    assert not r["ok"] and r["found"] is False


def test_read_barcodes_no_code_in_plain_image(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p, text="NO CODE HERE")
    r = c.read_barcodes(image=p, output_dir=str(tmp_path / "out"))
    # a plain photo has no barcode → ok=True, found=False (not an error)
    assert r["ok"] and r["found"] is False and r["count"] == 0


RECEIPT_TEXT = """SKLEP IFURI Sp. z o.o.
ul. Testowa 1, Warszawa
NIP 123-456-32-18
2026-06-23 14:05
Chleb razowy        4,99
Mleko 2%       2 x  3,50
Kawa ziarnista     29,90 A
SUMA PLN           38,39
"""


def test_parse_receipt_extracts_items_total_meta():
    p = c._parse_receipt(RECEIPT_TEXT)
    assert p["total"] == 38.39 and p["totalSource"] == "total-line"
    assert p["currency"] == "PLN"
    assert p["date"] == "2026-06-23"
    assert p["nip"] == "1234563218"
    names = [i["name"] for i in p["items"]]
    assert any("Chleb" in n for n in names) and any("Kawa" in n for n in names)
    # the SUMA line is the total, never a line item
    assert all("SUMA" not in n.upper() for n in names)
    assert p["items"][0]["price"] == 4.99


def test_parse_receipt_total_falls_back_to_max_item():
    p = c._parse_receipt("Bread 4,99\nMilk 3,50\nCoffee 29,90")
    assert p["total"] == 29.90 and p["totalSource"] == "max-item" and p["currency"] is None


def test_receipt_parse_route_from_text():
    r = c.receipt_parse(text=RECEIPT_TEXT)
    assert r["ok"] and r["source"] == "text" and r["total"] == 38.39
    assert r["itemCount"] >= 3


def test_receipt_parse_route_from_image(tmp_path):
    pytest.importorskip("numpy")
    p = str(tmp_path / "receipt.png")
    _make_receipt(p)
    r = c.receipt_parse(image=p, output_dir=str(tmp_path / "out"), lang="eng")
    assert r["ok"] and r["source"] == "ocr" and "items" in r


def _b64_of_image(path):
    import base64
    return base64.b64encode(open(path, "rb").read()).decode("ascii")


def test_ingest_analyze_from_base64(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p, text="UPLOAD 99")
    b64 = _b64_of_image(p)
    r = c.ingest(bytes_b64=b64, filename="frame.png", action="analyze",
                 output_dir=str(tmp_path / "out"), lang="eng")
    assert r["ok"] and r["source"] == "browser-upload" and r["action"] == "analyze"
    assert r["uploadBytes"] > 0 and r.get("object")


def test_ingest_strips_data_url_prefix(tmp_path):
    p = str(tmp_path / "scene.png")
    _make_image(p)
    data_url = "data:image/png;base64," + _b64_of_image(p)
    r = c.ingest(bytes_b64=data_url, action="describe", output_dir=str(tmp_path / "out"))
    assert r["ok"] and r["description"]["text"]


def test_ingest_barcodes_from_base64(tmp_path):
    p = str(tmp_path / "qr.png")
    _make_qr(p, "https://ifuri.com/INV-2026-777")
    r = c.ingest(bytes_b64=_b64_of_image(p), action="barcodes",
                 required="INV-2026-777", output_dir=str(tmp_path / "out"))
    assert r["ok"] and r["found"] is True


def test_ingest_rejects_empty_and_bad_payload(tmp_path):
    assert not c.ingest(bytes_b64="").get("ok")
    assert not c.ingest(bytes_b64="!!!not-base64!!!", output_dir=str(tmp_path / "o")).get("ok")


@pytest.mark.skipif(not (LIVE and c._default_device()), reason="set URIRUN_CAMERA_LIVE=1 with a real camera")
def test_capture_live(tmp_path):
    out = str(tmp_path / "live.jpg")
    r = c.capture(output=out, beep=False)
    assert r["ok"] and os.path.isfile(out) and r["width"] > 0


def test_ledger_appends_on_receipt_and_respects_off(tmp_path, monkeypatch):
    ledger = str(tmp_path / "ledger.jsonl")
    monkeypatch.setenv("URIRUN_LEDGER", ledger)
    c.receipt_parse(text="SKLEP\nNIP 778-14-22-455\nKawa 9,90\nSUMA PLN 9,90")
    lines = [l for l in open(ledger, encoding="utf-8") if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["connector"] == "camera" and rec["event"] == "receipt" and rec["total"] == 9.90
    # disabling writes nothing more
    monkeypatch.setenv("URIRUN_LEDGER", "off")
    c.receipt_parse(text="X 1,00\nSUMA PLN 1,00")
    assert len([l for l in open(ledger) if l.strip()]) == 1


def test_static_vs_live_contract_on_outputs(tmp_path):
    p = str(tmp_path / "r.png"); _make_image(p, text="X 9,99")
    a = c.analyze(image=p, output_dir=str(tmp_path / "o"), ocr=False)
    assert a["kind"] == "scan" and a["live"] is False        # captured/processed = artifact
    r = c.receipt_parse(text="SUMA PLN 9,99")
    assert r["kind"] == "receipt" and r["live"] is False


def test_ledger_records_are_marked_static(tmp_path, monkeypatch):
    led = str(tmp_path / "l.jsonl"); monkeypatch.setenv("URIRUN_LEDGER", led)
    c.receipt_parse(text="SUMA PLN 1,00")
    rec = json.loads([l for l in open(led) if l.strip()][0])
    assert rec["live"] is False
