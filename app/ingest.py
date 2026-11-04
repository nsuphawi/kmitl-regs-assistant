"""Load the raw corpus (PDF / .txt / .md) into per-page text, then chunk it."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Page:
    doc_id: str
    title: str
    page: int  # 1-based
    text: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    page_start: int
    page_end: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def cite(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        return f"{self.title}, {pages}"


def clean_text(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _title_from_path(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip()


def load_pages(raw_dir: Path) -> list[Page]:
    pages: list[Page] = []
    files = sorted(raw_dir.iterdir())
    txt_stems = {p.stem for p in files if p.suffix.lower() in {".txt", ".md"}}
    for path in files:
        if path.name.startswith("."):
            continue
        if path.suffix.lower() == ".pdf" and path.stem in txt_stems:
            continue  # scanned PDF: use its OCR text (scripts/ocr_scanned.py) instead
        doc_id, title = path.stem, _title_from_path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for i, p in enumerate(reader.pages, start=1):
                text = clean_text(p.extract_text() or "")
                if text:
                    pages.append(Page(doc_id, title, i, text))
        elif suffix in {".txt", ".md"}:
            # Form feed (\f) separates pages in plain-text files; otherwise one page.
            for i, part in enumerate(path.read_text(encoding="utf-8").split("\f"), start=1):
                text = clean_text(part)
                if text:
                    pages.append(Page(doc_id, title, i, text))
    return pages


def chunk_pages(pages: list[Page], chunk_words: int = 300, overlap: int = 50) -> list[Chunk]:
    """Sliding word window per document; chunks may span pages and remember both ends."""
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    by_doc: dict[str, list[Page]] = {}
    for p in pages:
        by_doc.setdefault(p.doc_id, []).append(p)

    chunks: list[Chunk] = []
    step = chunk_words - overlap
    for doc_id, doc_pages in by_doc.items():
        words: list[str] = []
        word_page: list[int] = []
        for p in doc_pages:
            ws = p.text.split()
            words.extend(ws)
            word_page.extend([p.page] * len(ws))
        title = doc_pages[0].title
        n = 0
        for start in range(0, max(len(words), 1), step):
            window = words[start : start + chunk_words]
            if not window:
                break
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{n:04d}",
                    doc_id=doc_id,
                    title=title,
                    page_start=word_page[start],
                    page_end=word_page[start + len(window) - 1],
                    text=" ".join(window),
                )
            )
            n += 1
            if start + chunk_words >= len(words):
                break
    return chunks
