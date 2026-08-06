# eval_agent.py
# End-to-end evaluation of the SQL agent against Spider dev.
# For each question: run the full agent, execute BOTH the agent's SQL and the
# gold SQL, compare result sets. Every failure is attributed so we know WHERE
# to spend Phase 4 effort (routing vs SQL vs execution).
#
# Run: python eval_agent.py

# --- DLL preload (needed on this machine for chromadb) ---
import ctypes
_d = r"C:\Users\AdityaKumar\AppData\Local\Programs\Python310"
ctypes.CDLL(_d + r"\vcruntime140.dll")
ctypes.CDLL(_d + r"\vcruntime140_1.dll")
ctypes.CDLL(_d + r"\msvcp140.dll")

import json
import random
import sqlite3
from pathlib import Path

from config import DEV_JSON, SPIDER_DB_DIR
# Import the MEMORY-FREE graph (each question independent) + state builder.
from sql_ag import graph

N = 40                      # questions to evaluate (seed-42, comparable across runs)
SEED = 42


# ---------------------------------------------------------------
# Direct SQLite execution (bypasses langchain formatting) so we can
# compare raw rows as sets. We run gold vs agent SQL ourselves.
# ---------------------------------------------------------------

def run_sql(db_id: str, sql: str, timeout=15):
    """Execute SQL on a Spider db, return set of row-tuples or ('ERROR', msg)."""
    path = SPIDER_DB_DIR / db_id / f"{db_id}.sqlite"
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
        con.text_factory = lambda b: b.decode("utf-8", "replace")  # dodge bad encodings
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        # normalize: set of tuples, stringified, order-independent
        return {tuple(str(c) for c in row) for row in rows}
    except Exception as e:
        return ("ERROR", str(e))


def results_match(agent_res, gold_res) -> bool:
    """Compare as ORDER-INDEPENDENT sets. (Spider's standard exec-accuracy is
    order-sensitive only when the question says 'ordered by'; set comparison is
    a slightly lenient but fair proxy.)"""
    if isinstance(agent_res, tuple) or isinstance(gold_res, tuple):
        return False        # one of them errored
    return agent_res == gold_res


# ---------------------------------------------------------------
# Run the agent on one question, capture its SQL + routed db.
# ---------------------------------------------------------------

def run_agent(question: str):
    initial = {
        "question": question,
        "messages": [("user", question)],
        "error": "no",
        "attempts": 0,
        "query": "",
        "result": "",
        "db_id": "",
        "candidates": [],
        "reroutes": 0,
        "banned_dbs": [],
    }
    final = None
    for event in graph.stream(initial, stream_mode="values"):
        final = event
    return final.get("db_id", ""), final.get("query", "")


# ---------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------

def main():
    dev = json.load(open(DEV_JSON, encoding="utf-8"))
    random.seed(SEED)
    sample = random.sample(dev, N)

    passed = 0
    # failure buckets for attribution
    routing_miss = []     # agent routed to wrong db
    agent_sql_error = []  # agent SQL crashed
    wrong_result = []     # ran fine, right db, but rows differ from gold
    gold_error = []       # gold SQL itself errored (benchmark quirk - exclude)

    for i, ex in enumerate(sample):
        q = ex["question"]
        gold_db = ex["db_id"]
        gold_sql = ex["query"]

        print(f"\n[{i+1}/{N}] {q}")
        try:
            agent_db, agent_sql = run_agent(q)
        except Exception as e:
            agent_db, agent_sql = "", f"(agent crashed: {e})"

        gold_res = run_sql(gold_db, gold_sql)
        # agent executes against the db IT chose (routing miss then naturally fails)
        agent_res = run_sql(agent_db, agent_sql) if agent_db else ("ERROR", "no db")

        # gold itself broken? exclude from scoring (Spider has a few)
        if isinstance(gold_res, tuple):
            gold_error.append((q, gold_db, gold_res[1]))
            print(f"   ⚠️  gold SQL errored - excluded")
            continue

        if results_match(agent_res, gold_res):
            passed += 1
            print(f"   ✅ PASS  (db: {agent_db})")
        else:
            if agent_db != gold_db:
                routing_miss.append((q, gold_db, agent_db))
                print(f"   ❌ ROUTING  gold={gold_db} agent={agent_db}")
            elif isinstance(agent_res, tuple):
                agent_sql_error.append((q, agent_db, agent_res[1]))
                print(f"   ❌ SQL ERROR  {agent_res[1][:80]}")
            else:
                wrong_result.append((q, agent_db, agent_sql, gold_sql))
                print(f"   ❌ WRONG RESULT  (right db, rows differ)")

    scored = N - len(gold_error)
    print("\n" + "=" * 60)
    print(f"EXECUTION ACCURACY: {passed}/{scored} = "
          f"{100*passed/scored:.0f}%   (excluded {len(gold_error)} broken-gold)")
    print("=" * 60)
    print(f"Failure attribution:")
    print(f"  routing miss (wrong db):        {len(routing_miss)}")
    print(f"  agent SQL error (crashed):      {len(agent_sql_error)}")
    print(f"  wrong result (right db, bad SQL):{len(wrong_result)}")
    print(f"  [excluded] broken gold SQL:     {len(gold_error)}")

    print("\n--- ROUTING misses ---")
    for q, g, a in routing_miss[:6]:
        print(f"  {q}\n     gold={g}  agent={a}")

    print("\n--- WRONG RESULTS (right db, SQL logic off) ---")
    for q, db, asql, gsql in wrong_result[:6]:
        print(f"  {q}  [{db}]")
        print(f"     agent: {asql[:100]}")
        print(f"     gold:  {gsql[:100]}")

    print("\n--- AGENT SQL ERRORS ---")
    for q, db, err in agent_sql_error[:6]:
        print(f"  {q}  [{db}]\n     {err[:100]}")


if __name__ == "__main__":
    main()