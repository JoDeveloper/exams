#!/usr/bin/env python3
"""
Arabic PDF Accessibility Processor
===================================
Makes Arabic PDFs accessible for screen readers (TalkBack / VoiceOver) by:
  1. Detecting whether each PDF already has a searchable text layer.
  2. Running Arabic OCR (Tesseract) on scanned/image-only PDFs and embedding
     the resulting text layer back into the file.

Requirements
------------
  sudo apt-get install tesseract-ocr tesseract-ocr-ara poppler-utils
  pip install pdfplumber pytesseract pdf2image pypdf reportlab \
              arabic-reshaper python-bidi

Usage
-----
  # Process all PDFs in the current directory (in-place):
  python3 make_accessible.py

  # Dry-run: only report which files need OCR, make no changes:
  python3 make_accessible.py --dry-run

  # Write processed files to a separate output directory:
  python3 make_accessible.py --output-dir ./accessible_pdfs
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy imports – fail fast with a clear message if a dep is missing
# ---------------------------------------------------------------------------
def _require(module, package=None):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        pkg = package or module
        sys.exit(f"Missing dependency: '{pkg}'.  Install with:  pip install {pkg}")


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
# OCR helpers
# ---------------------------------------------------------------------------

def ocr_pdf(pdf_path: Path, output_path: Path, dpi: int = 300) -> None:
    """
    Convert each page of a scanned PDF to an image, run Arabic OCR,
    and write a new PDF that embeds both the original page image and
    an invisible text layer so the file is both visually identical
    and screen-reader accessible.
    """
    pytesseract   = _require("pytesseract")
    pdf2image_mod = _require("pdf2image", "pdf2image")
    reportlab_canvas = _require("reportlab.pdfgen.canvas", "reportlab")
    reportlab_sizes  = _require("reportlab.lib.pagesizes",  "reportlab")
    arabic_reshaper  = _require("arabic_reshaper")
    bidi_algo        = _require("bidi.algorithm", "python-bidi")

    from io import BytesIO
    from pypdf import PdfWriter, PdfReader
    import arabic_reshaper
    from bidi.algorithm import get_display
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from pdf2image import convert_from_path

    images = convert_from_path(str(pdf_path), dpi=dpi)
    writer = PdfWriter()

    for i, image in enumerate(images):
        # --- OCR ---
        raw_text = pytesseract.image_to_string(
            image,
            lang="ara",
            config="--psm 6 --oem 1",
        )
        # Reshape Arabic glyphs and apply BiDi so the text reads correctly
        reshaped = arabic_reshaper.reshape(raw_text)
        bidi_text = get_display(reshaped)

        # --- Build an invisible text layer with ReportLab ---
        img_w, img_h = image.size          # pixels
        # Convert pixels → points (72 pt per inch)
        pt_w = img_w * 72 / dpi
        pt_h = img_h * 72 / dpi

        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=(pt_w, pt_h))
        c.setFillColorRGB(0, 0, 0, alpha=0)   # fully transparent
        c.setFont("Helvetica", 8)

        # Write text lines from top to bottom
        lines = bidi_text.splitlines()
        y = pt_h - 12
        for line in lines:
            if line.strip():
                c.drawString(8, y, line)
            y -= 10
            if y < 10:
                break

        c.save()
        packet.seek(0)

        # --- Merge image page + text layer ---
        # Save the original image as a single-page PDF
        img_pdf_buf = BytesIO()
        image.save(img_pdf_buf, format="PDF", resolution=dpi)
        img_pdf_buf.seek(0)

        img_page  = PdfReader(img_pdf_buf).pages[0]
        text_page = PdfReader(packet).pages[0]

        img_page.merge_page(text_page)   # overlay invisible text on image
        writer.add_page(img_page)

    with open(output_path, "wb") as f:
        writer.write(f)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_directory(
    directory: Path,
    output_dir: Path | None,
    dry_run: bool,
    dpi: int,
) -> None:
    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    total      = len(pdf_files)
    needs_ocr  = []
    already_ok = []
    errors     = []

    print(f"Scanning {total} PDF files for extractable text…\n")

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

    print(f"\nSummary: {len(already_ok)} already searchable, "
          f"{len(needs_ocr)} need OCR, {len(errors)} errors.\n")

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
        print(f"  [{i}/{len(needs_ocr)}] OCR → {dest.name} …", end=" ", flush=True)
        try:
            if not output_dir:
                # In-place: write to a temp file first, then replace
                fd, tmp = tempfile.mkstemp(dir=pdf.parent, suffix=".tmp.pdf")
                os.close(fd)
                ocr_pdf(pdf, Path(tmp), dpi=dpi)
                os.replace(tmp, pdf)
                tmp = None
            else:
                ocr_pdf(pdf, dest, dpi=dpi)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            errors.append((pdf, str(e)))
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    if errors:
        print(f"\n{len(errors)} file(s) encountered errors:")
        for pdf, msg in errors:
            print(f"  {pdf.name}: {msg}")
    else:
        print(f"\nAll done. {len(needs_ocr)} file(s) processed successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Make Arabic PDFs accessible by adding OCR text layers."
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
    args = ap.parse_args()

    process_directory(
        directory  = Path(args.directory),
        output_dir = Path(args.output_dir) if args.output_dir else None,
        dry_run    = args.dry_run,
        dpi        = args.dpi,
    )


if __name__ == "__main__":
    main()
