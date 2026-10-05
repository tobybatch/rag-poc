#!/usr/bin/env python3
"""
Reads every page saved by crawler.py / pdf_crawler.py in data/raw/, splits each into chunks,
embeds them with a local sentence-transformers model, and stores them in a
persistent local Chroma vector database at data/index/.

Run this after crawler.py and/or pdf_crawler.py, and again any time you
re-crawl / the playbook or PDFs change. It's a full rebuild each time - simple and fast enough at this
scale (a few hundred chunks), so there's no incremental-update logic to
worry about.

Usage:
    uv run build_index.py
"""
import glob
import json
import os
import re

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
COLLECTION_NAME = "playbook"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs locally - no API key needed

MAX_CHUNK_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text on '## heading' lines into (heading, body) pairs."""
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    for line in lines:
        m = re.match(r"^#{1,6}\s+(.*)", line.strip())
        if m:
            flush()
            current_heading = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if not sections:
        sections = [("", text)]
    return sections


def split_long_body(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]

    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) > max_chars and current:
            chunks.append(current)
            # carry a little overlap forward for context continuity
            tail = current[-overlap:]
            current = (tail + "\n\n" + para).strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_page(page: dict) -> list[dict]:
    chunks = []
    sections = split_into_sections(page["text"])
    for section_index, (heading, body) in enumerate(sections):
        for piece_index, piece in enumerate(split_long_body(body, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)):
            chunk_text = f"{heading}\n\n{piece}".strip() if heading else piece
            chunks.append(
                {
                    "text": chunk_text,
                    "url": page["url"],
                    "title": page["title"],
                    "heading": heading,
                    "chunk_id": f"{page['url']}#{section_index}-{piece_index}",
                }
            )
    return chunks


def build():
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not raw_files:
        raise SystemExit(f"No pages found in {RAW_DIR} - run crawler.py and/or pdf_crawler.py first.")

    all_chunks: list[dict] = []
    for path in raw_files:
        with open(path, encoding="utf-8") as f:
            page = json.load(f)
        all_chunks.extend(chunk_page(page))

    print(f"Loaded {len(raw_files)} pages -> {len(all_chunks)} chunks")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it, ~90MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    os.makedirs(INDEX_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=INDEX_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="embedding"):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"url": c["url"], "title": c["title"], "heading": c["heading"]} for c in batch],
        )

    print(f"\nIndexed {len(all_chunks)} chunks from {len(raw_files)} pages into {os.path.abspath(INDEX_DIR)}")


if __name__ == "__main__":
    build()
