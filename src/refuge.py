"""Readers for REFUGE-MultiRater: per-expert masks, the official merge, and the ellipse sheets.

Every encoding here was verified against the released files before being written down, because
each one has already cost a bug:

* `{id}_cup.bmp` and `{id}_disc.bmp` are **8-bit palette** BMPs, not greyscale. The cup is
  palette index **0**, the disc is index **128**, background is **255**. Reading them as grey
  levels and thresholding at 128 produces an empty disc mask and a vCDR of `inf`.
* `{id}_seg_cup_{k}.png` / `{id}_seg_disc_{k}.png` hold expert k's own outline, k = 1..7.
  These are the **training targets**. The .xlsx ellipses are for analysis only.
* In the ellipse sheets the `angle` field is measured from the tool's upright default, so the
  rotation is `angle - 90` degrees ("convention T"). Verified against the rasterised masks to
  three decimals.

Pillow is used when present (Colab has it) and a stdlib zlib/struct decoder is used when it is
not (this project's Windows laptop has no Pillow).
"""

from __future__ import annotations

import json
import os
import re
import struct
import zipfile

import numpy as np

RATERS = [f"user{k}" for k in range(1, 8)]
SPLITS = ("Training-400", "Validation-400", "Test-400")
CAMERA = {"Training-400": "Zeiss", "Validation-400": "Canon", "Test-400": "Canon"}

CUP_INDEX, DISC_INDEX, BACKGROUND_INDEX = 0, 128, 255

try:  # pragma: no cover - environment dependent
    from PIL import Image

    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    _HAVE_PIL = False


# --------------------------------------------------------------------------- the official merge
def read_bmp8(path: str) -> np.ndarray:
    """Read an 8-bit BMP as raw palette indices. Returns rows top-to-bottom."""
    with open(path, "rb") as fh:
        b = fh.read()
    offset = struct.unpack("<I", b[10:14])[0]
    width, height = struct.unpack("<ii", b[18:26])
    bits = struct.unpack("<H", b[28:30])[0]
    if bits != 8:
        raise ValueError(f"{path}: expected 8-bit BMP, got {bits}-bit")
    stride = (width + 3) // 4 * 4
    rows = np.frombuffer(b, dtype=np.uint8, count=stride * abs(height), offset=offset)
    rows = rows.reshape(abs(height), stride)[:, :width]
    return rows[::-1] if height > 0 else rows  # positive height = bottom-up


def official_masks(image_dir: str, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    """The 8th senior specialist's merged cup and disc, as boolean masks."""
    cup = read_bmp8(os.path.join(image_dir, f"{image_id}_cup.bmp")) == CUP_INDEX
    disc = read_bmp8(os.path.join(image_dir, f"{image_id}_disc.bmp")) == DISC_INDEX
    return cup, disc


# --------------------------------------------------------------------------- per-expert masks
def _read_png_grey_stdlib(path: str) -> np.ndarray:
    """Minimal PNG decoder for 8-bit non-interlaced images, using only zlib and struct."""
    import zlib

    with open(path, "rb") as fh:
        b = fh.read()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat = 8, b""
    width = height = channels = 0
    while pos < len(b):
        length = struct.unpack(">I", b[pos : pos + 4])[0]
        kind = b[pos + 4 : pos + 8]
        body = b[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            if depth != 8:
                raise ValueError(f"{path}: {depth}-bit PNG not supported")
            channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour]
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * channels
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    p = 0
    for y in range(height):
        filt = raw[p]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=p + 1).astype(np.int32)
        p += 1 + stride
        if filt == 1:  # Sub
            for x in range(channels, stride):
                line[x] = (line[x] + line[x - channels]) & 0xFF
        elif filt == 2:  # Up
            line = (line + prev) & 0xFF
        elif filt == 3:  # Average
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + int(prev[x])) >> 1)) & 0xFF
        elif filt == 4:  # Paeth
            for x in range(stride):
                a = line[x - channels] if x >= channels else 0
                c = int(prev[x - channels]) if x >= channels else 0
                bb = int(prev[x])
                pp, pa, pb, pc = a + bb - c, 0, 0, 0
                pa, pb, pc = abs(pp - a), abs(pp - bb), abs(pp - c)
                pred = a if (pa <= pb and pa <= pc) else (bb if pb <= pc else c)
                line[x] = (line[x] + pred) & 0xFF
        prev = line.astype(np.uint8)
        out[y] = prev
    return out.reshape(height, width, channels)[:, :, 0] if channels > 1 else out


def read_grey(path: str) -> np.ndarray:
    """An 8-bit single-channel image, via Pillow when available."""
    if _HAVE_PIL:
        return np.array(Image.open(path).convert("L"))
    return _read_png_grey_stdlib(path)


def expert_masks(image_dir: str, image_id: str, rater: int) -> tuple[np.ndarray, np.ndarray]:
    """Expert `rater` (1..7) own cup and disc outlines — the training targets."""
    cup = read_grey(os.path.join(image_dir, f"{image_id}_seg_cup_{rater}.png"))
    disc = read_grey(os.path.join(image_dir, f"{image_id}_seg_disc_{rater}.png"))
    return cup > 127, disc > 127


# --------------------------------------------------------------------------- ellipse sheets
def read_ellipses(xlsx_path: str) -> list[tuple[dict, dict]]:
    """(disc, cup) ellipse parameters for the seven raters, in rater order.

    Parses the sheet with regex rather than openpyxl so it runs in a bare environment.
    """
    with zipfile.ZipFile(xlsx_path) as z:
        shared = re.findall(r"<t[^>]*>([^<]*)</t>", z.read("xl/sharedStrings.xml").decode("utf-8"))
        sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    cells: dict[str, str] = {}
    pattern = r'<c r="([A-Z]+[0-9]+)"([^>]*)>(?:<f>[^<]*</f>)?<v>([^<]*)</v>'
    for ref, attrs, value in re.findall(pattern, sheet):
        cells[ref] = shared[int(value)] if 't="s"' in attrs else value
    return [
        (json.loads(cells[f"C{r}"]), json.loads(cells[f"D{r}"]))
        for r in range(3, 10)
        if cells.get(f"A{r}") in RATERS
    ]


def ellipse_extents(e: dict) -> tuple[float, float]:
    """Full vertical and horizontal extent of a rotated ellipse (convention T: angle - 90)."""
    a, b = e["width"] / 2.0, e["height"] / 2.0
    t = np.deg2rad(e["angle"] - 90.0)
    vertical = 2.0 * np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)
    horizontal = 2.0 * np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
    return float(vertical), float(horizontal)


def ellipse_inside(e: dict, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Boolean mask: which (X, Y) points fall inside the ellipse."""
    t = np.deg2rad(e["angle"] - 90.0)
    dx, dy = X - e["centerX"], Y - e["centerY"]
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    return (u / (e["width"] / 2.0)) ** 2 + (v / (e["height"] / 2.0)) ** 2 <= 1.0


# --------------------------------------------------------------------------- walking the release
def iter_images(root: str, splits=SPLITS):
    """Yield (split, image_id, image_dir) for every annotated image in the release.

    Skips `.DS_Store` and any other non-numeric entry — that one has bitten this project before.
    """
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            path = os.path.join(split_dir, name)
            if name.isdigit() and os.path.isdir(path):
                yield split, name, path
