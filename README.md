# urirun-connector-camera

**Camera capture + OCR** — connector ekosystemu [ifURI / urirun](https://github.com/if-uri/urirun).
Schemat URI: `camera://`

Point a USB/built-in webcam at the world and turn a frame into data over `camera://` URIs:
**discover cameras → beep before scan → capture a still photo → find & crop to the
dominant object → OCR/inspect it.**

## Routes (URI)

| URI | What it does |
| --- | --- |
| `camera://host/devices/query/list` | List cameras (names + `/dev/video*` via the usb connector) |
| `camera://host/photo/command/capture` | Capture one still frame to a file (optional base64) |
| `camera://host/photo/query/describe` | Capture → natural-language description of what's in the photo |
| `camera://host/photo/query/analyze` | Capture → describe → detect object → crop → OCR (full pipeline) |
| `camera://host/photo/query/inspect` | Capture/analyze/OCR and evaluate rules (`required_text`, brightness, object found), returning structured alerts + persisting a verdict (sidecar + `audit_log`) |
| `camera://host/photo/query/compare` | Change/motion detection — two files, reference vs live frame, or two frames `interval_ms` apart → `changed` + `changeRatio` + changed region |
| `camera://host/photo/query/barcodes` | Capture → decode barcodes / QR codes (pyzbar) → type, data, rect; `required` + `fail_if_missing` to assert an expected code |
| `camera://host/photo/query/ocr` | Capture (optionally crop) → return just the text |
| `camera://host/receipt/query/parse` | Scan a receipt (paragon) → structured JSON: line items (name+price), total, currency, date, NIP. Give `text` to parse existing OCR |
| `camera://host/upload/command/ingest` | Process a browser/mobile base64 frame → analyze\|inspect\|barcodes\|ocr\|describe\|receipt |

## Pipeline (`analyze`)

1. **Capture** a frame with `ffmpeg` (v4l2). OpenCV is an optional alternate backend.
2. **Describe** what's in the photo — scene, dominant colours, detected objects/regions,
   barcodes/QR — using the [img2nl](https://github.com/wronai/img2nl) engine when available
   (reached through [urirun-connector-ocr](../urirun-connector-ocr)); Pillow stats fallback.
3. **Detect & crop** to the `target` ("dociąć do obiektu / paragonu"):
   - `target="document"` / `"receipt"` / `"paragon"` — a numpy projection detector that hugs
     a bright text-dense sheet on a darker background (tight crop to the receipt/page);
   - `target="object"` — the dominant object (img2nl region detector, else Pillow edge bbox);
   - `target="auto"` (default) — document crop when a sheet is found, else the object;
   - `target="none"` — OCR the whole frame.
   - `deskew=true` — first try a **4‑point perspective correction**: find the document's
     corners and warp it flat, so a receipt shot at an angle becomes an upright rectangle
     before OCR. Corner detection uses OpenCV when present, else a numpy bright‑sheet
     detector; the warp itself is Pillow‑only (no OpenCV required).
4. **OCR** the crop with `tesseract` — automatically upgraded to the richer OCR connector
   backends (imgl / img2nl) when installed.

Returns the photo path + size, the natural-language description, the detected bbox + crop
path (with the `detector` used), and the recognised text. `target` is available on
`analyze`, `inspect`, `ocr` and `upload/command/ingest` — e.g. scan a receipt with
`target=receipt` so OCR runs on the trimmed sheet, not the whole desk.

Set `beep=true` on `capture`, `analyze`, `describe`, `ocr` or `inspect` to emit an
audible pre-scan cue. The connector tries `beep`, `play`, generated WAV through
`paplay`/`aplay`/`ffplay`, then terminal BEL. Set `beep_required=true` when a flow
must not capture unless the sound cue succeeded.

`inspect` wraps `analyze` and adds rule checks:

- `required_text` / `forbidden_text`
- `min_chars`
- `require_object`
- `brightness_min` / `brightness_max`
- `beep_on_alert`
- `fail_on_alert` to stop a URI flow when the inspection fails
- `audit_log` to append a one-line JSON verdict per scan; an `inspection.json` sidecar is
  also written next to the photo, so alerting keeps a durable trail with no `log://` connector

`compare` is change/motion detection (Pillow-only). With no images it captures two frames
`interval_ms` apart (motion); with `reference` it compares a fresh frame to a baseline; with
`reference` + `image` it compares two files. `beep_on_change` alerts on change and
`fail_on_change` stops a flow — e.g. "scan only when something appears in view".

`barcodes` decodes barcodes / QR codes (needs `pyzbar` + system `libzbar0`; falls back to
img2nl). Each code returns `{type, data, rect}`. Pass `required="..."` to match an expected
substring, `beep_on_read` to beep when a code is found, and `fail_if_missing=true` to stop a
flow when the expected code is absent — e.g. "read the shipping label's QR, alert if missing".
Install with `pip install -e '.[barcode]'` (plus `apt install libzbar0`).

## Wymagania

- **system:** `ffmpeg` (capture), `tesseract` (OCR), Linux `/dev/video*`
- **python:** `urirun`, `pillow`
- **optional:** `urirun-connector-ocr` (img2nl scene description + richer OCR), `opencv-python` (alt capture), `urirun-connector-usb` (named camera discovery)

Install the scene/OCR upgrade with `pip install -e '.[scene]'` (pulls in the OCR connector,
which loads the local `img2nl`/`imgl` checkouts).

## Instalacja (dev)

```bash
pip install -e .
pytest -q
```

## Szybki start

```bash
# which cameras are connected?
urirun-camera devices

# take a photo
urirun-camera capture --output /tmp/shot.jpg --beep true

# describe what the camera is looking at
urirun-camera describe

# capture, describe, crop to the main object, OCR it
urirun-camera analyze --output_dir /tmp/cam --beep true

# just read the text in front of the camera
urirun-camera ocr --crop true --lang eng+pol

# inspect a label/document and emit alert data if expected text is absent
urirun-camera inspect --required_text "FAKTURA" --beep true --beep_on_alert true
```

Discovery is shared with [urirun-connector-usb](../urirun-connector-usb): `usb://host/cameras/query/list`
tells the camera connector which `/dev/video*` belongs to which physical webcam.

## Powiązane

- Rdzeń: [if-uri/urirun](https://github.com/if-uri/urirun)
- USB: [urirun-connector-usb](../urirun-connector-usb) — device discovery
- OCR: [urirun-connector-ocr](../urirun-connector-ocr) — richer text extraction
- Hub connectorów: [connect.ifuri.com](https://connect.ifuri.com)

---
Kategoria: Hardware · Słowa kluczowe: camera, webcam, capture, photo, ocr, tesseract, ffmpeg, crop, object-detection, beep, inspection, alert · Wydawca: if-uri
