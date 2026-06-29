"""Perspective-correction / deskew helpers extracted from core.py."""
from __future__ import annotations

import os
from typing import Any


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


def _largest_mask_component(mask):
    """Keep the largest 4-connected component in a boolean mask.

    Deskew corner extraction uses sum/diff extremes. Without this cleanup a tiny bright
    reflection far from the page can steal a corner even after block reduction.
    """
    import numpy as np  # type: ignore

    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    starts = np.argwhere(mask)
    for sy, sx in starts:
        y = int(sy)
        x = int(sx)
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        component: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) > len(best):
            best = component

    if not best:
        return mask
    cleaned = np.zeros(mask.shape, dtype=bool)
    ys, xs = zip(*best)
    cleaned[list(ys), list(xs)] = True
    return cleaned


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
    clean_component = _largest_mask_component(clean)
    if clean_component.sum() >= 0.03 * clean.size:
        clean = clean_component

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
