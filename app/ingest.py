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


_ARTICLE_NUM = re.compile(r"\d{1,3}\.")
_CLAUSE_NUM = re.compile(r"\d{1,2}\.\d{1,2}(?:\.\d{1,2})?")


def clause_starts(words: list[str]) -> list[int]:
    """Word indices where a new Article / Chapter / numbered clause (e.g. 14.4) begins.

    Cross-references ("under Article 42.4 hereof") are not boundaries: an Article boundary needs
    "Article <n>." and a clause boundary needs the previous word to end a sentence or list item.
    """
    starts = [0]
    for i in range(1, len(words)):
        w, prev = words[i], words[i - 1]
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if w in ("Article", "ARTICLE", "Chapter", "CHAPTER") and (_ARTICLE_NUM.fullmatch(nxt) or (w.lower() == "chapter" and nxt.isdigit())):
            starts.append(i)
        elif _CLAUSE_NUM.fullmatch(w) and (prev[-1:] in ".;:" or prev.lower() in {"and", "or"}):
            starts.append(i)
    return starts


def _windows(lo: int, hi: int, size: int, overlap: int) -> list[tuple[int, int]]:
    out, step = [], size - overlap
    for s in range(lo, hi, step):
        out.append((s, min(s + size, hi)))
        if s + size >= hi:
            break
    return out


def chunk_pages(pages: list[Page], chunk_words: int = 300, overlap: int = 50, strategy: str = "window") -> list[Chunk]:
    """Split each document into chunks.

    window: fixed sliding word window (the naive baseline).
    clause: structure-aware. Split at Article / clause boundaries (regulations are numbered),
            pack whole clauses up to `chunk_words`, and prefix each chunk with its Article heading
            so a short clause like "14.4 ... not more than 22 credits" is not buried inside a
            300-word chunk about tuition fees. Over-long clauses fall back to windows.
    """
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    if strategy not in ("window", "clause"):
        raise ValueError(f"unknown chunk strategy {strategy!r}")
    by_doc: dict[str, list[Page]] = {}
    for p in pages:
        by_doc.setdefault(p.doc_id, []).append(p)

    chunks: list[Chunk] = []
    for doc_id, doc_pages in by_doc.items():
        words: list[str] = []
        word_page: list[int] = []
        for p in doc_pages:
            ws = p.text.split()
            words.extend(ws)
            word_page.extend([p.page] * len(ws))
        if not words:
            continue
        title = doc_pages[0].title

        spans: list[tuple[int, int, str]] = []  # (start, end, heading prefix)
        if strategy == "window":
            spans = [(a, b, "") for a, b in _windows(0, len(words), chunk_words, overlap)]
        else:
            bounds = clause_starts(words) + [len(words)]
            segments = list(zip(bounds[:-1], bounds[1:]))
            heading, cur = "", None  # cur = [start, end, heading]
            for a, b in segments:
                if words[a] in ("Article", "ARTICLE"):
                    nxt_clause = next((j for j in range(a + 2, min(b, a + 25)) if _CLAUSE_NUM.fullmatch(words[j])), min(b, a + 15))
                    heading = " ".join(words[a:nxt_clause])
                if b - a > chunk_words:  # over-long clause -> windows
                    if cur:
                        spans.append(tuple(cur)); cur = None
                    spans += [(x, y, heading) for x, y in _windows(a, b, chunk_words, overlap)]
                elif cur and (b - cur[0]) <= chunk_words:
                    cur[1] = b
                else:
                    if cur:
                        spans.append(tuple(cur))
                    cur = [a, b, heading]
            if cur:
                spans.append(tuple(cur))

        for n, (a, b, heading) in enumerate(spans):
            body = " ".join(words[a:b])
            text = f"[{heading}] {body}" if heading and not body.startswith(heading[:20]) else body
            chunks.append(Chunk(id=f"{doc_id}:{n:04d}", doc_id=doc_id, title=title,
                                page_start=word_page[a], page_end=word_page[b - 1], text=text))
    return chunks
