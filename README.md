# urirun-connector-camera

**Camera capture + OCR** — connector ekosystemu [ifURI / urirun](https://github.com/if-uri/urirun).
Schemat URI: `camera://`

Point a USB/built-in webcam at the world and turn a frame into data over `camera://` URIs:
**discover cameras → capture a still photo → find & crop to the dominant object → OCR it.**

## Routes (URI)

| URI | What it does |
| --- | --- |
| `camera://host/devices/query/list` | List cameras (names + `/dev/video*` via the usb connector) |
| `camera://host/photo/command/capture` | Capture one still frame to a file (optional base64) |
| `camera://host/photo/query/describe` | Capture → natural-language description of what's in the photo |
| `camera://host/photo/query/analyze` | Capture → describe → detect object → crop → OCR (full pipeline) |
| `camera://host/photo/query/ocr` | Capture (optionally crop) → return just the text |

## Pipeline (`analyze`)

1. **Capture** a frame with `ffmpeg` (v4l2). OpenCV is an optional alternate backend.
2. **Describe** what's in the photo — scene, dominant colours, detected objects/regions,
   barcodes/QR — using the [img2nl](https://github.com/wronai/img2nl) engine when available
   (reached through [urirun-connector-ocr](../urirun-connector-ocr)); Pillow stats fallback.
3. **Detect** the dominant object — img2nl's region detector (largest region), falling back
   to a Pillow edge-density bounding box when img2nl isn't installed.
4. **Crop** to that object ("dociąć do obiektu") and save the crop.
5. **OCR** the crop with `tesseract` — automatically upgraded to the richer OCR connector
   backends (imgl / img2nl) when installed.

Returns the photo path + size, the natural-language description, the detected object bbox +
crop path, and the recognised text.

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
urirun-camera capture --output /tmp/shot.jpg

# describe what the camera is looking at
urirun-camera describe

# capture, describe, crop to the main object, OCR it
urirun-camera analyze --output_dir /tmp/cam

# just read the text in front of the camera
urirun-camera ocr --crop true --lang eng+pol
```

Discovery is shared with [urirun-connector-usb](../urirun-connector-usb): `usb://host/cameras/query/list`
tells the camera connector which `/dev/video*` belongs to which physical webcam.

## Powiązane

- Rdzeń: [if-uri/urirun](https://github.com/if-uri/urirun)
- USB: [urirun-connector-usb](../urirun-connector-usb) — device discovery
- OCR: [urirun-connector-ocr](../urirun-connector-ocr) — richer text extraction
- Hub connectorów: [connect.ifuri.com](https://connect.ifuri.com)

---
Kategoria: Hardware · Słowa kluczowe: camera, webcam, capture, photo, ocr, tesseract, ffmpeg, crop, object-detection · Wydawca: if-uri
