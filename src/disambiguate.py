# disambiguate.py
# Stage 2 of routing: LLM picks the right db from top-k candidates
# Run standalone to eval: python disambiguate.py

# --- DLL preload ---
import ctypes
_d = r"C:\Users\AdityaKumar\AppData\Local\Programs\Python310"
ctypes.CDLL(_d + r"\vcruntime140.dll")
ctypes.CDLL(_d + r"\vcruntime140_1.dll")
ctypes.CDLL(_d + r"\msvcp140.dll")

import json
import random
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

from src.config import DEV_JSON, spider_db_uri
from src.vectorstore import db_docs_store

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------- stage 1: candidate retrieval ----------

def route_docs(q: str, k=5):
    hits = db_docs_store.similarity_search(q, k=k)
    return [h.metadata["db_id"] for h in hits]


# ---------- stage 2: LLM disambiguation ----------

# LINK-1 FIX: the judge previously saw schemas WITHOUT any data values
# (sample_rows_in_table_info=0, added to dodge a dirty-date crash).
# That made twin databases (flight_2 vs flight_4) a coin flip: their table
# structures look alike, and questions that mention literal values
# ("APG", "CVO", "Aberdeen") gave the judge nothing to grab onto.
# Now we TRY to include 3 sample rows per table; if that database's dirty
# data crashes the sample fetch (the ValueError we hit), we fall back to
# the bare schema instead of failing.

@lru_cache(maxsize=200)
def get_schema(db_id: str) -> str:
    try:
        sdb = SQLDatabase.from_uri(spider_db_uri(db_id),
                                   sample_rows_in_table_info=3)
        return sdb.get_table_info()          # schema + 3 sample rows per table
    except Exception:
        pass                                  # dirty data -> fall back below
    try:
        sdb = SQLDatabase.from_uri(spider_db_uri(db_id),
                                   sample_rows_in_table_info=0)
        return sdb.get_table_info()          # bare schema, no samples
    except Exception as e:
        return f"(schema unavailable for {db_id}: {e})"


def disambiguate(question: str, candidates: list[str]) -> str:
    if len(candidates) == 1:
        return candidates[0]

    blocks = "\n\n".join(
        f"=== DATABASE: {c} ===\n{get_schema(c)}"
        for c in candidates
    )
    prompt = f"""A user asked: "{question}"

Below are the schemas (with sample rows where available) of {len(candidates)}
candidate databases. Exactly one is right. If the question mentions specific
values (names, codes, years), check which database's sample data plausibly
contains such values.

{blocks}

Which database can answer this question? Reply with ONLY the database name
exactly as written after 'DATABASE:'. No other text."""

    answer = llm.invoke(prompt).content.strip()
    answer = answer.replace("DATABASE:", "").strip()   # model sometimes echoes the label

    if answer in candidates:
        return answer
    for c in candidates:                     # tolerate minor formatting
        if c.lower() in answer.lower():
            return c
    return candidates[0]                     # fallback: rank-1


def route_two_stage(question: str) -> str:
    return disambiguate(question, route_docs(question, k=5))


# ---------- standalone eval ----------

def main():
    dev = json.load(open(DEV_JSON, encoding="utf-8"))
    random.seed(42)                          # same exam every run
    sample = random.sample(dev, 100)[:40]    # first 40 of the canonical 100

    n = len(sample)
    correct = 0
    fixable_misses = []
    ceiling_misses = []

    for i, ex in enumerate(sample):
        q = ex["question"]
        candidates = route_docs(q, k=5)
        choice = disambiguate(q, candidates)

        if choice == ex["db_id"]:
            correct += 1
        elif ex["db_id"] in candidates:
            fixable_misses.append((q, ex["db_id"], choice, candidates))
        else:
            ceiling_misses.append((q, ex["db_id"], candidates))

        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{n}")

    print(f"\nTwo-stage routing accuracy: {correct}/{n} = {100*correct/n:.0f}%")
    print(f"LLM picked wrong despite gold in candidates: {len(fixable_misses)}/{n}")
    print(f"Gold never made the candidates (retrieval ceiling): {len(ceiling_misses)}/{n}")

    print("\n--- LLM mistakes (fixable) ---")
    for q, gold, chose, cands in fixable_misses[:5]:
        print(f"Q: {q}\n   gold: {gold}   LLM chose: {chose}   candidates: {cands}\n")

    print("--- Retrieval ceiling misses ---")
    for q, gold, cands in ceiling_misses[:5]:
        print(f"Q: {q}\n   gold: {gold}   candidates: {cands}\n")


if __name__ == "__main__":
    main()