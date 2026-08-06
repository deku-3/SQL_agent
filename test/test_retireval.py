import ctypes
_d = r"C:\Users\AdityaKumar\AppData\Local\Programs\Python310"
ctypes.CDLL(_d + r"\vcruntime140.dll")
ctypes.CDLL(_d + r"\vcruntime140_1.dll")
ctypes.CDLL(_d + r"\msvcp140.dll")
print("runtime preloaded")

import json, random
from SQL_agent.src.config import DEV_JSON
from SQL_agent.src.vectorstore import db_docs_store

dev = json.load(open(DEV_JSON, encoding="utf-8"))
sample = random.sample(dev, 100)

top1 = top3 = 0
misses = []
for ex in sample:
    hits = db_docs_store.similarity_search(ex["question"], k=3)
    ids = [h.metadata["db_id"] for h in hits]
    if ids[0] == ex["db_id"]: top1 += 1
    if ex["db_id"] in ids: top3 += 1
    else: misses.append((ex["question"], ex["db_id"], ids))

print(f"Top-1 routing accuracy: {top1}%")
print(f"Top-3 routing accuracy: {top3}%")
print("\nSample misses:")
for q, gold, got in misses[:5]:
    print(f"  Q: {q}\n  gold: {gold}  got: {got}\n")