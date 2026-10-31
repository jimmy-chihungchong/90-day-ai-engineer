"""Quick terminal test: python cli.py <folder> "your question" [--answer] [-k 6]"""
import argparse
import sys

from semsearch.config import get_settings
from semsearch.embeddings import Embedder, Index
from semsearch.llm import LLMTimeout, answer, suggest_alternatives
from semsearch.loaders import load_folder

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("query")
ap.add_argument("-k", type=int, default=6)
ap.add_argument("--answer", action="store_true", help="also ask the chat LLM to answer from the hits")
ap.add_argument("--provider", default=None)
a = ap.parse_args()

s = get_settings(a.provider)
chunks = load_folder(a.folder)
print(f"{len(chunks)} chunks from {a.folder} | embed={s.embed_model}")
embedder = Embedder(s)
index = Index.build(chunks, embedder, progress=lambda f: print(f"  embedding {f:.0%}", end="\r"))
hits = index.search(embedder.embed([a.query], "query")[0], a.k)
for n, (score, c) in enumerate(hits, 1):
    print(f"\n[{n}] {score:.3f}  {c.source} | {c.title}\n    {c.text[:240].replace(chr(10), ' ')}")
if a.answer:
    try:
        print("\n--- ANSWER ---\n" + answer(s, a.query, hits))
    except LLMTimeout:
        print(f"\n!! {s.chat_model} timed out after {s.timeout}s. Try: {suggest_alternatives(s)}")
