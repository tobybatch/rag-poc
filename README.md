# Playbook RAG (Spike)

CLI tool to answer questions using dxw's Playbook (`playbook.dxw.com`) via RAG with Claude. Simple pipeline: crawl -> chunk & embed locally -> retrieve & query Claude.

## Quickstart

```bash
# Setup
cd playbook-rag
uv sync                        # creates .venv and installs locked dependencies

export ANTHROPIC_API_KEY=sk-ant-...   # on Windows: set ANTHROPIC_API_KEY=sk-ant-...
```

# Run

```bash
# 1. Crawl the whole site (takes a few minutes; add --max-pages 10 to test quickly first)
uv run crawler.py
#    optional: --base-url <url> to crawl another site, --xpath "<expr>" to only
#    save text from the matching part of each page (e.g. --xpath "//main")

# 1b. (optional) Add a folder of PDFs - saved alongside the crawled pages
uv run pdf_crawler.py path/to/pdfs
#    optional: --recursive for subfolders, --clean to drop previously added PDFs,
#    --url-prefix <url> to cite them as hosted URLs instead of local file paths

# 2. Chunk and embed everything into a local vector database
#    (first run downloads a small embedding model, ~90MB)
uv run build_index.py

# 3. Ask questions
uv run ask.py "How much holiday do I get?"

# or run it as an interactive chat:
uv run ask.py
```

* `MAX_CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS`: Chunking behavior.
* `TOP_K`: Number of context chunks pulled per query.
* `CLAUDE_MODEL`: Swap the underlying LLM.

## Project Structure

```text
crawler.py       -> Scrapes site into data/raw/
build_index.py   -> Chunks raw JSONs + embeds into Chroma
ask.py           -> CLI query / interactive chat interface
data/raw/        -> Saved JSON pages (6 sample pages pre-loaded)
data/index/      -> Local Chroma DB directory
```

## Tuning

A few things worth adjusting if answers feel off, in `build_index.py` /
`ask.py`:

- `MAX_CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS` - chunk size and overlap
- `TOP_K` in `ask.py` - how many chunks get pulled per question
- `CLAUDE_MODEL` in `ask.py` - swap for a different Claude model if you
  want

## Files

```
crawler.py       - crawls playbook.dxw.com -> data/raw/*.json
pdf_crawler.py   - extracts text from a folder of PDFs -> data/raw/pdf-*.json
build_index.py   - chunks + embeds data/raw/ -> data/index/ (Chroma DB)
ask.py           - retrieval + Claude Q&A
data/raw/        - one JSON file per crawled page (url, title, text)
data/index/      - the Chroma vector database (created by build_index.py)
scripts/seed_sample.py - the script that seeded the 6 sample pages (safe to ignore)
```
