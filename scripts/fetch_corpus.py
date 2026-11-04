"""Download the KMITL PDFs listed in data/sources.json into data/raw/ and write a SHA-256 manifest.

    python -m scripts.fetch_corpus            # skip files that already exist
    python -m scripts.fetch_corpus --force    # re-download

If a download fails (Drive sometimes demands a browser), open the "page" URL from
data/sources.json, download the PDF manually, and save it as data/raw/<name>.pdf.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from app.config import ROOT

SOURCES = ROOT / "data" / "sources.json"
URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"


def fetch(drive_id: str, dest: Path) -> None:
    req = urllib.request.Request(URL.format(id=drive_id), headers={"User-Agent": "kmitl-regs-assistant/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise RuntimeError("response is not a PDF (Drive may require a manual download)")
    dest.write_bytes(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for doc in json.loads(SOURCES.read_text(encoding="utf-8"))["documents"]:
        dest = raw / f"{doc['name']}.pdf"
        if dest.exists() and not args.force:
            print(f"skip  {dest.name} (exists)")
        else:
            try:
                fetch(doc["drive_id"], dest)
                print(f"ok    {dest.name} ({dest.stat().st_size // 1024} KB)")
            except Exception as exc:
                print(f"FAIL  {dest.name}: {exc}\n      download manually from {doc['page']}")
                continue
        manifest[dest.name] = {"sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "source": doc["page"]}
    (raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {raw / 'MANIFEST.json'} ({len(manifest)} files)")


if __name__ == "__main__":
    main()
