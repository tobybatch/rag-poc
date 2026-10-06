# Playbook RAG (Spike)

CLI tool to answer questions using dxw's Playbook (`playbook.dxw.com`) via RAG. The documents, embeddings, and retrieval run locally; answers can use Claude or a local Ollama model.

## Quickstart

```bash
# Setup
cd playbook-rag
uv sync                        # creates .venv and installs locked dependencies

# Needed only for the default Claude provider:
export ANTHROPIC_API_KEY=sk-ant-...   # on Windows: set ANTHROPIC_API_KEY=sk-ant-...
```

# Run

```bash
# 1. Crawl the whole site (takes a few minutes; add --max-pages 10 to test quickly first)
uv run crawler.py
#    optional: --base-url <url> to crawl another site, --xpath "<expr>" to only
#    save text from the matching part of each page (e.g. --xpath "//main")

# 1b. (optional) Add a folder of PDFs - saved alongside the crawled pages
uv run pdf_crawler.py path/to/pdfs path/to/more-pdfs
#    --recursive reads subfolders of every directory; --clean drops previously added PDFs,
#    --url-prefix <url> to cite them as hosted URLs instead of local file paths

# 1c. (optional) Add a folder of Markdown files
uv run markdown_crawler.py path/to/markdown path/to/more-markdown
#    --recursive reads subfolders of every directory; --clean drops previously added Markdown,
#    --url-prefix <url> to cite them as hosted URLs instead of local file paths

# 2. Chunk and embed everything into a local vector database
#    (first run downloads a small embedding model, ~90MB)
uv run build_index.py

# 3. Ask questions
uv run ask.py "How much holiday do I get?"

# or run it as an interactive chat:
uv run ask.py

# Use a local Ollama model instead (no Anthropic API key or cloud LLM calls)
ollama pull llama3.2
ollama serve                     # if Ollama is not already running
uv run ask.py --provider ollama --model llama3.2 "How much holiday do I get?"
# or start an interactive local chat:
uv run ask.py --provider ollama --model llama3.2
```

* `MAX_CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS`: Chunking behavior.
* `TOP_K`: Number of context chunks pulled per query.
* `--provider ollama --model <name>`: Use a locally installed Ollama model.
* `OLLAMA_HOST`: Set a non-default local Ollama server URL.
* `--model <name>`: Select a different Claude or Ollama model.

For fully offline use, first download the Ollama model and the embedding model
(`all-MiniLM-L6-v2`) while online. Crawling websites also requires internet;
once the source documents, embedding model, and Ollama model are available
locally, Markdown/PDF ingestion, indexing, and question answering can run
without a cloud LLM. Ollama mode loads the embedding model from the local cache
only and contacts only the configured Ollama server (localhost by default).
Ollama and the embedding model both run on your machine.

## Project Structure

```text
crawler.py          -> Scrapes site into data/raw/
pdf_crawler.py      -> Extracts text from PDFs into data/raw/
markdown_crawler.py -> Saves Markdown files into data/raw/
build_index.py      -> Chunks raw JSONs + embeds into Chroma
ask.py              -> CLI query / interactive chat with Claude or Ollama
data/raw/           -> Saved JSON pages (6 sample pages pre-loaded)
data/index/         -> Local Chroma DB directory
```

## Tuning

A few things worth adjusting if answers feel off, in `build_index.py` /
`ask.py`:

- `MAX_CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS` - chunk size and overlap
- `TOP_K` in `ask.py` - how many chunks get pulled per question
- `--provider` / `--model` in `ask.py` - choose Claude or local Ollama

## Files

```
crawler.py             - crawls playbook.dxw.com -> data/raw/*.json
pdf_crawler.py         - extracts text from a folder of PDFs -> data/raw/pdf-*.json
markdown_crawler.py    - saves a folder of Markdown files -> data/raw/markdown-*.json
build_index.py         - chunks + embeds data/raw/ -> data/index/ (Chroma DB)
ask.py                 - retrieval + Claude Q&A
data/raw/              - one JSON file per crawled page (url, title, text)
data/index/            - the Chroma vector database (created by build_index.py)
scripts/seed_sample.py - the script that seeded the 6 sample pages (safe to ignore)
```
