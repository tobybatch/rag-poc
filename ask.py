#!/usr/bin/env python3
"""
Ask questions about dxw's Playbook, answered by Claude grounded in the
content indexed by build_index.py.

Requires the ANTHROPIC_API_KEY environment variable to be set to your
Anthropic API key.

Usage:
    uv run ask.py                                # interactive chat loop
    uv run ask.py "How much holiday do I get?"   # ask a single question and exit
"""
import os
import sys

import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
COLLECTION_NAME = "playbook"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-5"
TOP_K = 6

SYSTEM_PROMPT = """You are a helpful assistant answering questions about dxw's internal Playbook \
(playbook.dxw.com), using only the excerpts provided to you below.

Rules:
- Answer using only the information in the provided excerpts. Do not use outside knowledge.
- If the excerpts don't contain enough information to answer, say so plainly rather than guessing.
- After your answer, list the source page(s) you used as a short "Sources:" list with their URLs.
- Be concise and direct."""


def load_retriever():
    if not os.path.isdir(INDEX_DIR) or not os.listdir(INDEX_DIR):
        sys.exit(
            f"No index found at {INDEX_DIR}.\n"
            "Run 'uv run crawler.py' and then 'uv run build_index.py' first."
        )
    client = chromadb.PersistentClient(path=INDEX_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
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


def answer_question(client, collection, model, question: str) -> str:
    hits = retrieve(collection, model, question)
    context = build_context_block(hits)

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Playbook excerpts:\n\n{context}\n\n---\n\nQuestion: {question}",
            }
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set the ANTHROPIC_API_KEY environment variable to your Anthropic API key first.")

    collection, model = load_retriever()
    client = Anthropic()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(answer_question(client, collection, model, question))
        return

    print("Ask questions about dxw's Playbook. Type 'quit' to exit.\n")
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
        print(answer_question(client, collection, model, question))
        print()


if __name__ == "__main__":
    main()
