import ctypes
_d = r"C:\Users\AdityaKumar\AppData\Local\Programs\Python310"
ctypes.CDLL(_d + r"\vcruntime140.dll")
ctypes.CDLL(_d + r"\vcruntime140_1.dll")
ctypes.CDLL(_d + r"\msvcp140.dll")
print("runtime preloaded")
print("script started")

import json
from langchain_core.documents import Document
print("imports 1 done")

from src.vectorstore import examples_store
print("vectorstore loaded")

from src.config import TRAIN_JSON   # or your inline path
print("config loaded:", TRAIN_JSON)

def main():
    print("opening train json...")
    train = json.load(open(TRAIN_JSON, encoding="utf-8"))
    print("loaded", len(train), "examples")

    # train = train[:20]   # small test first!

    docs = [Document(
                page_content=ex["question"],
                metadata={"sql": ex["query"], "db_id": ex["db_id"]},
            ) for ex in train]
    print("docs built:", len(docs))

    B = 500
    for i in range(0, len(docs), B):
        print("adding batch...")
        examples_store.add_documents(docs[i:i+B])
        print(f"  {min(i+B, len(docs))}/{len(docs)}")
    print("✅ Examples bank ingested")

if __name__ == "__main__":
    main()