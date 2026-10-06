#!/usr/bin/env python3
"""
Reads a folder of PDFs and saves each one as clean text to data/raw/, in the
same format crawler.py uses, so build_index.py indexes them alongside the
crawled web pages.

Text is extracted page by page. Lines are regrouped into paragraphs (based on
the vertical gaps between them) so build_index.py can split long documents
into sensible chunks. Scanned PDFs with no text layer are skipped - there's
no OCR.

Each PDF is saved as data/raw/pdf-<name>.json. Re-running overwrites those
files; use --clean to also remove ones whose PDF no longer exists.

Usage:
    uv run pdf_crawler.py path/to/pdfs path/to/more-pdfs   # read both folders
    uv run pdf_crawler.py path/to/pdfs --recursive         # include subfolders
    uv run pdf_crawler.py path/to/pdfs path/to/more-pdfs --clean
                                                          # drop data/raw/pdf-*.json first
    uv run pdf_crawler.py path/to/pdfs path/to/more-pdfs --url-prefix https://example.com/docs/
                                                          # cite sources as hosted URLs instead of local file paths
"""
import argparse
import hashlib
import glob
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

import pdfplumber
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
FILE_PREFIX = "pdf-"
PARAGRAPH_GAP_RATIO = 1.5  # a gap this many times a line's height starts a new paragraph
MIN_TEXT_CHARS = 40  # same threshold crawler.py uses to skip near-empty pages


def find_pdfs(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")


def page_paragraphs(page) -> list[str]:
    """Group a page's text lines into paragraphs using the vertical gaps between them."""
    lines = page.extract_text_lines(strip=True)
    paragraphs: list[str] = []
    current: list[str] = []
    prev = None
    for line in lines:
        text = line["text"].strip()
        if not text:
            continue
        if prev is not None:
            height = max(prev["bottom"] - prev["top"], 1)
            gap = line["top"] - prev["bottom"]
            if gap > height * (PARAGRAPH_GAP_RATIO - 1) or line["top"] < prev["top"]:
                paragraphs.append(" ".join(current))
                current = []
        current.append(text)
        prev = line
    if current:
        paragraphs.append(" ".join(current))
    # Re-join words hyphenated across line breaks ("infor- mation" -> "information").
    return [re.sub(r"(\w)- (\w)", r"\1\2", p) for p in paragraphs]


def pdf_title(pdf, path: Path) -> str:
    title = (pdf.metadata or {}).get("Title")
    if isinstance(title, bytes):
        title = title.decode("utf-8", errors="ignore")
    title = (title or "").strip()
    # Metadata titles are often junk like "Microsoft Word - draft3.docx" or "untitled".
    if not title or title.lower() in {"untitled", "title"} or title.lower().startswith("microsoft word"):
        title = re.sub(r"[-_]+", " ", path.stem).strip()
    return title


def source_url(path: Path, folder: Path, url_prefix: str | None, multiple_folders: bool = False) -> str:
    if url_prefix:
        relative_path = path.relative_to(folder).as_posix()
        if multiple_folders:
            folder_slug = re.sub(r"[^a-z0-9]+", "-", folder.name.lower()).strip("-") or "folder"
            folder_hash = hashlib.sha256(str(folder.resolve()).encode("utf-8")).hexdigest()[:8]
            relative_path = f"{folder_slug}-{folder_hash}/{relative_path}"
        return url_prefix + quote(relative_path)
    return path.resolve().as_uri()


def output_name(path: Path, folder: Path, multiple_folders: bool = False) -> str:
    rel = path.relative_to(folder).with_suffix("").as_posix()
    if multiple_folders:
        folder_slug = re.sub(r"[^a-z0-9]+", "-", folder.name.lower()).strip("-") or "folder"
        folder_hash = hashlib.sha256(str(folder.resolve()).encode("utf-8")).hexdigest()[:8]
        rel = f"{folder_slug}-{folder_hash}/{rel}"
    slug = re.sub(r"[^a-z0-9]+", "-", rel.lower()).strip("-") or "document"
    return f"{FILE_PREFIX}{slug}.json"


def extract_pdf(
    path: Path, folder: Path, url_prefix: str | None, multiple_folders: bool = False
) -> dict | None:
    with pdfplumber.open(path) as pdf:
        paragraphs = []
        for page in pdf.pages:
            paragraphs.extend(page_paragraphs(page))
        title = pdf_title(pdf, path)

    text = "\n\n".join(paragraphs).strip()
    if len(text) < MIN_TEXT_CHARS:
        return None
    return {
        "url": source_url(path, folder, url_prefix, multiple_folders),
        "title": title,
        "text": text,
    }


def crawl_pdfs(
    folders: Path | list[Path],
    recursive: bool = False,
    url_prefix: str | None = None,
    clean: bool = False,
) -> list[dict]:
    os.makedirs(RAW_DIR, exist_ok=True)
    if clean:
        for old in glob.glob(os.path.join(RAW_DIR, f"{FILE_PREFIX}*.json")):
            os.remove(old)

    if isinstance(folders, Path):
        folders = [folders]
    folders = list(dict.fromkeys(folder.resolve() for folder in folders))
    pdfs = [(path, folder) for folder in folders for path in find_pdfs(folder, recursive)]
    if not pdfs:
        folder_list = ", ".join(str(folder) for folder in folders)
        raise SystemExit(f"No PDFs found in {folder_list}" + ("" if recursive else " (try --recursive?)"))

    pages = []
    written: dict[str, Path] = {}
    multiple_folders = len(folders) > 1
    for path, folder in tqdm(pdfs, desc="reading PDFs", unit="pdf"):
        try:
            page = extract_pdf(path, folder, url_prefix, multiple_folders)
        except Exception as e:  # corrupt/encrypted PDFs shouldn't stop the whole run
            tqdm.write(f"  ! failed to read {path}: {e}")
            continue
        if page is None:
            tqdm.write(f"  - skipped {path} (no extractable text - scanned image?)")
            continue

        name = output_name(path, folder, multiple_folders)
        if name in written:
            tqdm.write(f"  ! {path} and {written[name]} map to the same output file {name}; skipping {path}")
            continue
        written[name] = path

        with open(os.path.join(RAW_DIR, name), "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)
        pages.append(page)
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="+", help="One or more folders containing PDFs")
    parser.add_argument("--recursive", action="store_true", help="Also read PDFs in subfolders")
    parser.add_argument("--url-prefix", default=None, help="Cite each PDF as <prefix><relative path> instead of a local file:// path (multi-folder runs include a folder name)")
    parser.add_argument("--clean", action="store_true", help="Delete previously saved data/raw/pdf-*.json files before reading")
    args = parser.parse_args()

    for folder in args.folders:
        if not folder.is_dir():
            parser.error(f"{folder} is not a folder")

    folders = [folder.resolve() for folder in args.folders]
    pages = crawl_pdfs(folders, recursive=args.recursive, url_prefix=args.url_prefix, clean=args.clean)
    print(f"\nDone. Saved {len(pages)} PDFs to {os.path.abspath(RAW_DIR)}")
