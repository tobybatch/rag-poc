#!/usr/bin/env python3
"""
Reads a folder of Markdown files and saves each one to data/raw/ in the
same format used by crawler.py, so build_index.py indexes them alongside
the crawled web pages.

Markdown is kept intact so headings, lists, and code blocks remain useful
when the index splits documents into chunks.

Usage:
    uv run markdown_crawler.py path/to/markdown
    uv run markdown_crawler.py path/to/markdown --recursive
    uv run markdown_crawler.py path/to/markdown --clean
    uv run markdown_crawler.py path/to/markdown --url-prefix https://example.com/docs/
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
FILE_PREFIX = "markdown-"
MARKDOWN_SUFFIXES = {".md", ".markdown"}


def find_markdown_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in folder.glob(pattern)
        if path.is_file() and path.suffix.lower() in MARKDOWN_SUFFIXES
    )


def source_url(path: Path, folder: Path, url_prefix: str | None) -> str:
    if url_prefix:
        return url_prefix + quote(path.relative_to(folder).as_posix())
    return path.resolve().as_uri()


def output_name(path: Path, folder: Path) -> str:
    relative_path = path.relative_to(folder).with_suffix("").as_posix()
    slug = re.sub(r"[^a-z0-9]+", "-", relative_path.lower()).strip("-") or "document"
    return f"{FILE_PREFIX}{slug}.json"


def extract_markdown(path: Path, folder: Path, url_prefix: str | None) -> dict | None:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    heading = re.search(r"^#\s+(.+?)\s*#*\s*$", text, re.MULTILINE)
    title = heading.group(1).strip() if heading else re.sub(r"[-_]+", " ", path.stem).strip()
    return {"url": source_url(path, folder, url_prefix), "title": title, "text": text}


def crawl_markdown(
    folder: Path,
    recursive: bool = False,
    url_prefix: str | None = None,
    clean: bool = False,
) -> list[dict]:
    os.makedirs(RAW_DIR, exist_ok=True)
    if clean:
        for old in glob.glob(os.path.join(RAW_DIR, f"{FILE_PREFIX}*.json")):
            os.remove(old)

    markdown_files = find_markdown_files(folder, recursive)
    if not markdown_files:
        raise SystemExit(
            f"No Markdown files found in {folder}" + ("" if recursive else " (try --recursive?)")
        )

    pages = []
    written: dict[str, Path] = {}
    for path in tqdm(markdown_files, desc="reading Markdown", unit="file"):
        try:
            page = extract_markdown(path, folder, url_prefix)
        except (OSError, UnicodeError) as error:
            tqdm.write(f"  ! failed to read {path}: {error}")
            continue
        if page is None:
            tqdm.write(f"  - skipped {path} (empty file)")
            continue

        name = output_name(path, folder)
        if name in written:
            tqdm.write(f"  ! {path} and {written[name]} map to the same output file {name}; skipping {path}")
            continue
        written[name] = path

        with open(os.path.join(RAW_DIR, name), "w", encoding="utf-8") as output:
            json.dump(page, output, ensure_ascii=False, indent=2)
        pages.append(page)
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder containing Markdown files")
    parser.add_argument("--recursive", action="store_true", help="Also read Markdown files in subfolders")
    parser.add_argument("--url-prefix", default=None, help="Cite each file as <prefix><relative path> instead of a local file:// path")
    parser.add_argument("--clean", action="store_true", help="Delete previously saved data/raw/markdown-*.json files before reading")
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"{args.folder} is not a folder")

    pages = crawl_markdown(args.folder.resolve(), recursive=args.recursive, url_prefix=args.url_prefix, clean=args.clean)
    print(f"\nDone. Saved {len(pages)} Markdown files to {os.path.abspath(RAW_DIR)}")
