"""OCR the scanned (image-only) PDFs in data/raw/ into <name>.txt beside them.

Two of the five KMITL PDFs (the 2022 "guidebooks", which are actually the 2022 regulations)
contain no text layer, so pypdf extracts nothing. This script renders each page with
`pdftoppm` and runs `tesseract` (English), writing one page per form-feed (\\f) so citations
keep page numbers. The loader then uses the .txt instead of the PDF.

Requires the poppler-utils and tesseract-ocr system packages.

    python -m scripts.ocr_scanned                 # OCR every PDF with < 20 words of text
    python -m scripts.ocr_scanned --dpi 200
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader

from app.config import settings
from scripts.ocr_clean import clean_ocr_text

def needs_ocr(pdf: Path) -> bool:
    reader = PdfReader(str(pdf))
    return sum(len((p.extract_text() or "").split()) for p in reader.pages) < 20


def ocr_pdf(pdf: Path, dpi: int) -> list[str]:
    n = len(PdfReader(str(pdf)).pages)
    pages = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in range(1, n + 1):
            prefix = Path(tmp) / f"p{p}"
            subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-f", str(p), "-l", str(p), "-png", str(pdf), str(prefix)], check=True)
            img = next(Path(tmp).glob(f"p{p}*.png"))
            text = subprocess.run(["tesseract", str(img), "-", "-l", "eng", "--psm", "3"],
                                  check=True, capture_output=True, text=True).stdout
            img.unlink()
            pages.append(clean_ocr_text(text))
            print(f"  {pdf.name} page {p}/{n}")
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()
    for tool in ("pdftoppm", "tesseract"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found: install poppler-utils and tesseract-ocr")
    for pdf in sorted(settings.raw_dir.glob("*.pdf")):
        out = pdf.with_suffix(".txt")
        if out.exists() or not needs_ocr(pdf):
            continue
        pages = ocr_pdf(pdf, args.dpi)
        out.write_text("\f".join(pages) + "\n", encoding="utf-8")
        print(f"wrote {out.name}: {len(pages)} pages, {sum(len(p.split()) for p in pages)} words")


if __name__ == "__main__":
    main()
