# Examples — urirun-connector-camera

```bash
# Discover cameras (reuses usb:// camera discovery for product names):
urirun-camera devices

# Capture a single still frame:
urirun-camera capture --output /tmp/shot.jpg --warmup 5

# Full pipeline — capture, crop to the dominant object, OCR the crop:
urirun-camera analyze --output_dir /tmp/cam --lang eng+pol

# Run the pipeline on an existing image instead of capturing:
urirun-camera analyze --image /tmp/shot.jpg --output_dir /tmp/cam

# Just read the text the camera is pointing at:
urirun-camera ocr --crop true
```

## As URIs over a urirun node

```
camera://host/devices/query/list
camera://host/photo/command/capture     payload: {"device":"/dev/video0","output":"/tmp/shot.jpg"}
camera://host/photo/query/analyze       payload: {"crop":true,"lang":"eng+pol"}
camera://host/photo/query/ocr           payload: {"crop":true}
```

Chain with the office flow: `usb://host/cameras/query/list` → pick a `/dev/video*` →
`camera://host/photo/query/analyze` to photograph a document/label and read it.
