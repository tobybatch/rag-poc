#!/usr/bin/env python3
"""
Ask questions about dxw's Playbook, grounded in the content indexed by
build_index.py. Answers can come from Claude or a local Ollama model.

Claude requires the ANTHROPIC_API_KEY environment variable. Ollama requires
a running local Ollama server and the selected model pulled in advance.

Usage:
    uv run ask.py                                # interactive Claude chat
    uv run ask.py "How much holiday do I get?"   # ask a single question and exit
    uv run ask.py --provider ollama --model llama3.2
"""
import argparse
import os
import sys

import chromadb
import requests
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
COLLECTION_NAME = "playbook"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-5"
OLLAMA_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"
TOP_K = 6

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a crawled resource, using only the excerpts provided to you below.

Rules:
- Answer using only the information in the provided excerpts. Do not use outside knowledge.
- If the excerpts don't contain enough information to answer, say so plainly rather than guessing.
- After your answer, list the source page(s) you used as a short "Sources:" list with their URLs.
- Be concise and direct."""


def load_retriever(local_files_only: bool = False):
    if not os.path.isdir(INDEX_DIR) or not os.listdir(INDEX_DIR):
        sys.exit(
            f"No index found at {INDEX_DIR}.\n"
            "Run 'uv run crawler.py' and then 'uv run build_index.py' first."
        )
    client = chromadb.PersistentClient(path=INDEX_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=local_files_only)
    return collection, model


def retrieve(collection, model, question: str, top_k: int = TOP_K):
    query_embedding = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "url": meta["url"], "title": meta["title"], "distance": dist})
    return hits


def build_context_block(hits) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[Excerpt {i} - from \"{hit['title']}\" ({hit['url']})]\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(
    provider,
    client,
    collection,
    embedding_model,
    question: str,
    llm_model: str,
    ollama_host: str,
) -> str:
    hits = retrieve(collection, embedding_model, question)
    context = build_context_block(hits)
    prompt = f"Data excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"

    if provider == "ollama":
        response = requests.post(
            f"{ollama_host.rstrip('/')}/api/chat",
            json={
                "model": llm_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    message = client.messages.create(
        model=llm_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("claude", "ollama"),
        default="claude",
        help="Answer provider (default: claude)",
    )
    parser.add_argument("--model", help="Model name (defaults to the provider's configured default)")
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", OLLAMA_HOST),
        help=f"Ollama server URL (default: {OLLAMA_HOST}, or OLLAMA_HOST)",
    )
    parser.add_argument("question", nargs="*", help="Question to ask; omit to start interactive chat")
    args = parser.parse_args()

    if args.provider == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set the ANTHROPIC_API_KEY environment variable to your Anthropic API key first.")

    collection, embedding_model = load_retriever(local_files_only=args.provider == "ollama")
    llm_model = args.model or (OLLAMA_MODEL if args.provider == "ollama" else CLAUDE_MODEL)
    client = None
    if args.provider == "claude":
        from anthropic import Anthropic

        client = Anthropic()

    def answer(question: str) -> str:
        try:
            return answer_question(
                args.provider, client, collection, embedding_model, question, llm_model, args.ollama_host
            )
        except requests.RequestException as error:
            if args.provider == "ollama":
                raise SystemExit(
                    f"Could not reach Ollama at {args.ollama_host}: {error}\n"
                    "Make sure Ollama is running and the model is available locally."
                ) from error
            raise

    if args.question:
        print(answer(" ".join(args.question)))
        return

    print("Ask questions. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        print()
        print(answer(question))
        print()


if __name__ == "__main__":
    main()
