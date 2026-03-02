#!/usr/bin/env python3
"""
Arabic PDF Accessibility Processor (v3 – Word-Accurate)
=========================================================
Makes Arabic PDFs accessible for screen readers (TalkBack / VoiceOver) by:
  1. Detecting whether each PDF already has a searchable text layer.
  2. Running Arabic OCR (Tesseract) on scanned/image-only PDFs and embedding
     a precise, word-level invisible text layer using a proper Arabic font.

Key improvements in v3
-----------------------
  • Word-level bounding boxes (image_to_data) replace line-level rendering:
    – Each word is placed at its exact pixel position from Tesseract.
    – Multi-column layouts work correctly with zero word overlap.
    – Mobile screen readers get accurate per-word tap-target regions.
  • beginText() text objects replace drawRightString():
    – Independent state per word; no shared canvas state bleed.
    – Font size scales proportionally to each word's bounding box height.
  • Confidence filtering (MIN_CONF=40) discards OCR noise automatically.
  • Amiri Arabic font (auto-downloaded) ensures correct Unicode glyphs.
  • Parallel page processing via ProcessPoolExecutor (all CPU cores).
  • Document /Lang metadata set to "ar" for Arabic TTS engine selection.
  • Progress bar via tqdm (falls back gracefully if not installed).

Requirements
------------
  sudo apt-get install tesseract-ocr tesseract-ocr-ara poppler-utils
  pip install pdfplumber pytesseract pdf2image pypdf reportlab \
              arabic-reshaper python-bidi requests tqdm

Usage
-----
  # Process all PDFs in the current directory (in-place):
  python3 make_accessible.py

  # Dry-run: only report which files need OCR, make no changes:
  python3 make_accessible.py --dry-run

  # Write processed files to a separate output directory:
  python3 make_accessible.py --output-dir ./accessible_pdfs

  # Control parallelism (default: number of CPU cores):
  python3 make_accessible.py --workers 4
"""

import argparse
import os
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _require(module, package=None):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        pkg = package or module
        sys.exit(f"Missing dependency: '{pkg}'.  Install with:  pip install {pkg}")


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):          # noqa: F811  – silent fallback
        return iterable

# ---------------------------------------------------------------------------
# Arabic font – download Amiri if not already cached
# ---------------------------------------------------------------------------

FONT_NAME = "Amiri"
FONT_PATH = Path(__file__).parent / "Amiri-Regular.ttf"

# Multiple fallback URLs tried in order
FONT_URLS = [
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf",
    "https://github.com/harfbuzz/harfbuzz-monster-fonts/raw/main/Amiri-Regular.ttf",
]


def ensure_arabic_font() -> Path:
    """Download Amiri-Regular.ttf next to this script if missing."""
    if FONT_PATH.exists():
        return FONT_PATH

    for url in FONT_URLS:
        print(f"Downloading Arabic font (Amiri) from {url} …", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, FONT_PATH)
            print("done")
            return FONT_PATH
        except Exception as e:
            print(f"failed ({e})")

    # Last resort: pip install amiri
    print("Trying: pip install amiri …", end=" ", flush=True)
    try:
        import subprocess, shutil
        subprocess.check_call([sys.executable, "-m", "pip", "install", "amiri", "-q"])
        import amiri as _amiri_pkg
        src = Path(_amiri_pkg.__file__).parent / "Amiri-Regular.ttf"
        if src.exists():
            shutil.copy(src, FONT_PATH)
            print("done")
            return FONT_PATH
    except Exception as e:
        print(f"failed ({e})")

    sys.exit(
        f"\nCould not download Amiri font automatically.\n"
        f"Please manually download Amiri-Regular.ttf and place it at:\n"
        f"  {FONT_PATH}\n"
        f"Download from: https://fonts.google.com/specimen/Amiri"
    )


def _register_font(font_path: Path) -> None:
    """Register Amiri with ReportLab (idempotent)."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    try:
        pdfmetrics.getFont(FONT_NAME)
    except KeyError:
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))


# ---------------------------------------------------------------------------
# Text detection
# ---------------------------------------------------------------------------

def has_extractable_text(pdf_path: Path, min_chars: int = 20) -> bool:
    """Return True if the PDF already contains a selectable text layer."""
    pdfplumber = _require("pdfplumber")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text.strip()) >= min_chars:
                    return True
    except Exception as e:
        print(f"  [WARN] Could not read {pdf_path.name}: {e}")
    return False


# ---------------------------------------------------------------------------
# Per-page OCR worker (runs in a subprocess)
# ---------------------------------------------------------------------------

# Minimum Tesseract confidence (0-100) to accept a word into the text layer.
# Words below this threshold are likely OCR noise and are skipped.
MIN_CONF = 40


def _build_text_layer(
    image,
    dpi: int,
    font_path_str: str,
    pt_w: float,
    pt_h: float,
) -> BytesIO:
    """
    Use Tesseract word-level bounding boxes to build an invisible text layer.

    Each recognised word is placed in its own beginText() object at the exact
    pixel-accurate position Tesseract found it.  This approach:
      • Eliminates column-overlap (each word is independently positioned)
      • Handles multi-column Arabic layouts correctly
      • Allows confidence filtering to skip OCR noise
      • Gives mobile screen readers accurate tap-target regions per word
    """
    import arabic_reshaper
    from bidi.algorithm import get_display
    from reportlab.pdfgen import canvas
    import pytesseract

    _register_font(Path(font_path_str))

    # pixel → point scale factor
    scale = 72.0 / dpi

    # Get word-level data: text, bounding box (left, top, width, height),
    # confidence score, and block/paragraph/line numbers for grouping.
    data = pytesseract.image_to_data(
        image,
        lang="ara",
        config="--psm 6 --oem 1",
        output_type=pytesseract.Output.DICT,
    )

    text_buf = BytesIO()
    c = canvas.Canvas(text_buf, pagesize=(pt_w, pt_h))

    # Invisible text: alpha=0 means fully transparent but still selectable
    c.setFillColorRGB(0, 0, 0, alpha=0)

    n_words = len(data["text"])
    for i in range(n_words):
        word = data["text"][i]
        conf = int(data["conf"][i])

        # Skip empty strings and low-confidence garbage
        if not word.strip() or conf < MIN_CONF:
            continue

        # Tesseract bounding box in pixels
        x_px = data["left"][i]
        y_px = data["top"][i]
        h_px = data["height"][i]

        # Convert pixel coords → PDF points.
        # PDF y-axis is bottom-up; image y-axis is top-down.
        x_pt = x_px * scale
        # Align baseline to bottom of the bounding box
        y_pt = pt_h - (y_px + h_px) * scale

        # Font size proportional to bounding box height (clamped 6–32 pt)
        font_size = max(6.0, min(32.0, h_px * scale * 0.85))

        # Reshape Arabic ligatures + apply Unicode BiDi algorithm
        reshaped  = arabic_reshaper.reshape(word)
        bidi_word = get_display(reshaped)

        # Each word gets its own text object — no shared state, no overlap
        txt_obj = c.beginText(x_pt, y_pt)
        txt_obj.setFont(FONT_NAME, font_size)
        txt_obj.setFillColorRGB(0, 0, 0, alpha=0)  # keep transparent
        txt_obj.textOut(bidi_word)
        c.drawText(txt_obj)

    c.save()
    text_buf.seek(0)
    return text_buf


def _ocr_page(args):
    """
    OCR a single page image and return a bytes blob of a single-page PDF
    containing the original image + an invisible word-accurate Arabic text layer.

    Designed to be picklable so it runs safely in a worker process.
    """
    img_bytes, dpi, font_path_str = args

    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    image = Image.open(BytesIO(img_bytes))
    img_w, img_h = image.size
    pt_w = img_w * 72.0 / dpi
    pt_h = img_h * 72.0 / dpi

    # Build invisible word-level text layer
    text_buf = _build_text_layer(image, dpi, font_path_str, pt_w, pt_h)

    # Render original image as single-page PDF
    img_pdf_buf = BytesIO()
    image.save(img_pdf_buf, format="PDF", resolution=dpi)
    img_pdf_buf.seek(0)

    # Overlay the text layer onto the image page
    img_page  = PdfReader(img_pdf_buf).pages[0]
    text_page = PdfReader(text_buf).pages[0]
    img_page.merge_page(text_page)

    merged_buf = BytesIO()
    writer = PdfWriter()
    writer.add_page(img_page)
    writer.write(merged_buf)

    return merged_buf.getvalue()


# ---------------------------------------------------------------------------
# Main OCR entry-point for a single PDF
# ---------------------------------------------------------------------------

def ocr_pdf(
    pdf_path: Path,
    output_path: Path,
    dpi: int = 300,
    workers: int = 1,
    font_path: Path = FONT_PATH,
) -> None:
    """
    OCR all pages of *pdf_path* in parallel and write the result to *output_path*.
    Each page is processed by an independent worker process so all CPU cores
    are kept busy.
    """
    from pdf2image import convert_from_path
    from pypdf import PdfWriter

    images = convert_from_path(str(pdf_path), dpi=dpi)

    # Serialise images to bytes so they can be sent to worker processes
    page_args = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        page_args.append((buf.getvalue(), dpi, str(font_path)))

    merged_pages_bytes: list[bytes | None] = [None] * len(page_args)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_ocr_page, arg): idx
            for idx, arg in enumerate(page_args)
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=f"  OCR {pdf_path.name}",
            unit="page",
            leave=False,
        ):
            idx = future_to_idx[future]
            merged_pages_bytes[idx] = future.result()

    # Assemble final PDF
    final_writer = PdfWriter()
    from pypdf import PdfReader
    for page_bytes in merged_pages_bytes:
        reader = PdfReader(BytesIO(page_bytes))
        final_writer.add_page(reader.pages[0])

    # Embed document-level metadata for accessibility
    final_writer.add_metadata({
        "/Lang": "ar",
        "/Title": pdf_path.stem,
    })

    with open(output_path, "wb") as f:
        final_writer.write(f)


# ---------------------------------------------------------------------------
# Directory processing loop
# ---------------------------------------------------------------------------

def process_directory(
    directory: Path,
    output_dir: Path | None,
    dry_run: bool,
    dpi: int,
    workers: int,
) -> None:
    font_path = ensure_arabic_font()
    _register_font(font_path)          # verify font loads before processing

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    total     = len(pdf_files)
    needs_ocr = []
    already_ok= []
    errors    = []

    print(f"Scanning {total} PDF file(s) for extractable text…\n")

    for pdf in pdf_files:
        try:
            if has_extractable_text(pdf):
                already_ok.append(pdf)
                print(f"  [OK]  {pdf.name}")
            else:
                needs_ocr.append(pdf)
                print(f"  [OCR] {pdf.name}  ← needs OCR")
        except Exception as e:
            errors.append((pdf, str(e)))
            print(f"  [ERR] {pdf.name}: {e}")

    print(
        f"\nSummary: {len(already_ok)} already searchable, "
        f"{len(needs_ocr)} need OCR, {len(errors)} errors.\n"
    )

    if dry_run:
        print("Dry-run mode: no files were modified.")
        return

    if not needs_ocr:
        print("Nothing to do – all PDFs already have text layers.")
        return

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, pdf in enumerate(needs_ocr, 1):
        dest = (output_dir / pdf.name) if output_dir else pdf
        tmp  = None
        print(f"\n[{i}/{len(needs_ocr)}] Processing: {pdf.name}")
        try:
            if not output_dir:
                fd, tmp = tempfile.mkstemp(dir=pdf.parent, suffix=".tmp.pdf")
                os.close(fd)
                ocr_pdf(pdf, Path(tmp), dpi=dpi, workers=workers, font_path=font_path)
                os.replace(tmp, pdf)
                tmp = None
            else:
                ocr_pdf(pdf, dest, dpi=dpi, workers=workers, font_path=font_path)
            print(f"  ✓ Saved → {dest}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            errors.append((pdf, str(e)))
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    if errors:
        print(f"\n{len(errors)} file(s) had errors:")
        for pdf, msg in errors:
            print(f"  {pdf.name}: {msg}")
    else:
        print(f"\nAll done. {len(needs_ocr)} file(s) processed successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import multiprocessing
    ap = argparse.ArgumentParser(
        description="Make Arabic PDFs accessible by adding an OCR text layer."
    )
    ap.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing PDF files (default: current directory).",
    )
    ap.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Write processed files here instead of overwriting originals.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which files need OCR; do not modify anything.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI used when rasterising PDF pages for OCR (default: 300).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel OCR workers (default: all CPU cores).",
    )
    args = ap.parse_args()

    process_directory(
        directory  = Path(args.directory),
        output_dir = Path(args.output_dir) if args.output_dir else None,
        dry_run    = args.dry_run,
        dpi        = args.dpi,
        workers    = args.workers,
    )


if __name__ == "__main__":
    main()
