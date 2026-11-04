"""Remove the translation agency's stamp/signature blocks from OCR text.

The 2022 scans carry, on every page, a "Certified Correct Translation" stamp with a scribbled
signature, the translator's name, the agency name, phone/fax, URL and a date. OCR turns these into
garbage lines ("Fusmshuvagndon", "Tal. (66) 2007-5533-38 Fax, ...") that pollute chunks.

    python -m scripts.ocr_clean          # cleans data/raw/*.txt in place (idempotent)
"""
from __future__ import annotations

import re

from app.config import settings

STAMP_LINE = re.compile(
    r"(certified\s+correct\s+tran\w*|master\s*p\w*\s*tran\w*|masterp|kanyaret|lerdnun"
    r"|^\s*t[aeo]l[,.]?\s*\(\d+\)|\bfax\b)",
    re.I,
)
STAMP_DATE = re.compile(r"^\s*\d{1,2}\s+JUL\s+2023")
PAGE_NUM = re.compile(r"\s*-?\s*\d{1,3}\s*-?\.?\s*")
JUNK_TOKEN = re.compile(r"^[A-Za-z(){}\[\]]{1,20}$")  # lone garbled word (OCR'd signature)


def clean_ocr_text(text: str) -> str:
    out: list[str] = []
    after_stamp = 0  # lines remaining in which signature junk is dropped
    for line in text.split("\n"):
        s = line.strip()
        if STAMP_LINE.search(s):
            after_stamp = 6
            continue
        if after_stamp:
            after_stamp -= 1
            if not s or (JUNK_TOKEN.match(s) and s.lower() not in {"and", "or"}):
                continue
        if s == "Translation" or PAGE_NUM.fullmatch(line):
            continue
        out.append(STAMP_DATE.sub("", line))
    return "\n".join(out).strip()


def main() -> None:
    for path in sorted(settings.raw_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        pages = [clean_ocr_text(p) for p in raw.split("\f")]
        new = "\f".join(pages) + "\n"
        if new != raw:
            path.write_text(new, encoding="utf-8")
        print(f"{path.name}: {raw.count(chr(10)) - new.count(chr(10))} lines removed")


if __name__ == "__main__":
    main()
