# sql_ag.py
# Spider SQL agent:
#   gate -> rewrite -> retrieve -> pick_db -> write_query -> execute -> answer
#   (+ re-route loop when the chosen database turns out to be wrong)
# Run: python sql_ag.py

# --- DLL preload (needed on this machine for chromadb) ---
import ctypes
_d = r"C:\Users\AdityaKumar\AppData\Local\Programs\Python310"
ctypes.CDLL(_d + r"\vcruntime140.dll")
ctypes.CDLL(_d + r"\vcruntime140_1.dll")
ctypes.CDLL(_d + r"\msvcp140.dll")

import re
from dotenv import load_dotenv
load_dotenv()          # loads OPENAI_API_KEY + LANGFUSE_* keys
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, MessagesState, END
from src.config import CHECKPOINT_DB
import sqlite3
# --- LANGFUSE TRACING -------------------------------------------------
from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()
# ---------------------------------------------------------------------

from src.vectorstore import db_docs_store, examples_store
from src.config import spider_db_uri

# routing layer (eval-proven: 81% two-stage on dev sample)
from src.disambiguate import route_docs, disambiguate, get_schema


# ---------------------------------------------------------------
# Token-budget / abuse knobs
# ---------------------------------------------------------------

MAX_RESULT_CHARS = 4000    # cap on query-result text entering LLM prompts
MESSAGE_WINDOW = 8         # how many recent messages write_query sees
MAX_QUESTION_CHARS = 500   # reject oversized questions BEFORE any LLM call
MAX_REROUTES = 1           # how many times one question may switch database


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class SQLQuery(BaseModel):
    """Generate a SQL query to answer the user's question."""
    reasoning: str = Field(description="Brief explanation of how this query answers the question")
    query: str = Field(description="A syntactically correct SQLite query. SELECT statements only.")


# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------

class AgentState(MessagesState):
    question: str
    query: str
    result: str
    error: str
    attempts: int
    db_id: str            # the routed database
    candidates: list      # candidates from stage 1
    reroutes: int         # how many times we've switched database this question
    banned_dbs: list      # databases proven wrong for this question


# ---------------------------------------------------------------
# Dynamic database connections (one per Spider db, cached)
# ---------------------------------------------------------------

_conns = {}

def get_db(db_id: str) -> SQLDatabase:
    if db_id not in _conns:
        _conns[db_id] = SQLDatabase.from_uri(
            spider_db_uri(db_id), sample_rows_in_table_info=0)
    return _conns[db_id]


# ---------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------

def gate(state: AgentState):
    q = state["question"]
    if len(q) > MAX_QUESTION_CHARS:
        print(f"--- REJECTED: question too long ({len(q)} chars) ---")
        return {
            "error": "rejected",
            "messages": [("assistant",
                f"Your question is too long ({len(q)} characters). "
                f"Please keep it under {MAX_QUESTION_CHARS} characters.")],
        }
    return {"error": "no"}


def route_after_gate(state: AgentState):
    return "reject" if state["error"] == "rejected" else "rewrite"


REWRITE_PROMPT = """Given the conversation so far and a new user question, rewrite the
new question as a single fully standalone question that needs no prior context.
Resolve references like "them", "those", "it", "each one" using the conversation,
including concrete values from previous query results when they are what the
user refers to.
If the new question is about a DIFFERENT subject than the conversation, do NOT
graft the old subject onto it - return the new question unchanged.
If the question is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation so far:
{history}

New question: {question}
Standalone question:"""


def _history(state, n=6) -> str:
    prior = state["messages"][:-1]
    return "\n".join(
        f"{getattr(m, 'type', 'msg')}: {getattr(m, 'content', m)}" for m in prior[-n:])


def rewrite(state: AgentState):
    if not state["messages"][:-1]:
        return {}                     # first turn, nothing to resolve
    print("--- Rewriting follow-up into standalone question ---")
    standalone = llm.invoke(REWRITE_PROMPT.format(
        history=_history(state), question=state["question"])).content.strip()
    if not standalone or len(standalone) > 300:
        return {}                     # distrust a garbage rewrite
    if standalone != state["question"]:
        print(f"    rewritten: {standalone}")
    return {"question": standalone}


# ---------------------------------------------------------------
# ROUTING with LLM TOPIC-SHIFT CLASSIFIER
#
# Blind sticky routing (reuse the thread's db forever) answered "how many
# singers?" from world_1 by counting countries. Score-based heuristics also
# failed: retrieval is noisy enough that the old db lingers in the top-5 even
# for an unrelated question.
#
# So we ask a cheap LLM the question code cannot answer reliably:
# "is this a follow-up about the same subject, or a new subject?"
#   FOLLOWUP -> stay on the current db (skips the expensive judge call)
#   NEW      -> full re-route across all candidates
# Anything unparseable falls through to re-routing (fail toward the safe path).
# ---------------------------------------------------------------

FOLLOWUP_PROMPT = """Conversation so far:
{history}

New user message: "{question}"

Is this new message a follow-up about the SAME subject as the conversation, or
does it start a NEW subject that would need different data?
Answer with exactly one word: FOLLOWUP or NEW."""


def retrieve(state: AgentState):
    print("--- Routing: retrieving candidates ---")
    current = state.get("db_id")
    banned = state.get("banned_dbs") or []

    # A) re-route triggered by a proven-wrong database (see execute_query)
    if state.get("error") == "wrong_db":
        fresh = [d for d in route_docs(state["question"], k=6) if d not in banned]
        print(f"    (re-routing away from {banned}) candidates: {fresh}")
        return {"candidates": fresh, "error": "no"}

    # B) mid-conversation: ask whether this is a follow-up or a new subject
    if current and state["messages"][:-1]:
        verdict = llm.invoke(FOLLOWUP_PROMPT.format(
            history=_history(state, 4), question=state["question"])
        ).content.strip().upper()
        if "FOLLOWUP" in verdict:
            print(f"    (follow-up -> staying on {current})")
            return {"candidates": [current]}
        print(f"    (new subject -> re-routing away from {current})")

    # C) first turn, or a new subject
    fresh = [d for d in route_docs(state["question"], k=5) if d not in banned]
    return {"candidates": fresh}


def pick_db(state: AgentState):
    cands = state["candidates"]
    if not cands:
        print("--- Routing: no candidates left ---")
        return {"db_id": ""}
    if len(cands) == 1:
        print(f"--- Routing: single candidate, skipping disambiguation ({cands[0]}) ---")
        return {"db_id": cands[0]}
    print("--- Routing: disambiguating ---")
    db_id = disambiguate(state["question"], cands)
    print(f"    chose: {db_id}")
    return {"db_id": db_id}


def write_query(state: AgentState):
    print("--- Writing query ---")
    schema = get_schema(state["db_id"])

    ex_hits = examples_store.similarity_search(state["question"], k=2)
    examples = "\n\n".join(
        f"Question: {h.page_content}\nSQL: {h.metadata['sql']}" for h in ex_hits)

    system = f"""You are a SQL expert working with a SQLite database.

Database schema:
{schema}

Similar example queries (from different databases - use for SQL patterns only):
{examples}

Rules:
- SELECT statements only. Never INSERT, UPDATE, DELETE, DROP, or ALTER.
- Only use tables and columns from the schema above.
- Pay attention to which column belongs to which table.
- Prefer returning human-readable columns (names/titles) over internal IDs;
  JOIN to reference tables to get names when needed.
- Text values in the data may have inconsistent whitespace or casing.
  When filtering on specific text values (codes, names), compare defensively,
  e.g. WHERE TRIM(col) = 'APG' or LOWER(TRIM(col)) = LOWER('value').
- CRITICAL: if the question asks about entities that DO NOT EXIST in this
  schema, you MUST output exactly: SELECT 'SCHEMA_MISMATCH'
  Never substitute a different table as a stand-in. Counting countries when
  asked about singers is a serious error - outputting SCHEMA_MISMATCH is the
  correct behaviour, and another database will be tried instead.
- Unless the question asks for a complete list or is an aggregate
  (COUNT/SUM/AVG/...).

The question to answer (already resolved to standalone form):
{state['question']}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("placeholder", "{messages}"),
    ])
    chain = prompt | llm.with_structured_output(SQLQuery)
    solution = chain.invoke({"messages": state["messages"][-MESSAGE_WINDOW:]})
    return {
        "query": solution.query,
        "attempts": state["attempts"] + 1,
        "messages": [("assistant", f"Generated SQL [{state['db_id']}]: {solution.query}")],
    }


FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "truncate", "pragma")

# a missing table/column is a DETERMINISTIC signal that we're on the wrong
# database - no heuristics needed, SQLite itself told us
WRONG_DB_SIGNALS = ("no such table", "no such column")


def extract_literal_filters(query: str):
    pairs = []
    base = r"(?:\w+\()*\s*(\w+(?:\.\w+)?)\s*\)*\s*(?:=|LIKE)\s*'([^']+)'"
    pairs += re.findall(base, query, flags=re.IGNORECASE)
    for col, vals in re.findall(
            r"(?:\w+\()*\s*(\w+(?:\.\w+)?)\s*\)*\s+IN\s*\(([^)]+)\)",
            query, flags=re.IGNORECASE):
        pairs += [(col, v) for v in re.findall(r"'([^']+)'", vals)]
    return [(c.split(".")[-1], v.strip("%")) for c, v in pairs]


def probe_values(db_id: str, query: str, filters: list) -> str:
    """For each (column, value), find what's ACTUALLY stored near that value."""
    db = get_db(db_id)
    tables = re.findall(r"(?:FROM|JOIN)\s+(\w+)", query, flags=re.IGNORECASE)
    evidence = []
    for col, val in filters:
        safe_val = val.replace("'", "''")
        for table in set(tables):
            try:
                found = db.run(
                    f"SELECT DISTINCT \"{col}\" FROM \"{table}\" "
                    f"WHERE \"{col}\" LIKE '%{safe_val}%' LIMIT 5")
                if found and str(found).strip() not in ("", "[]"):
                    evidence.append(
                        f"- You filtered {col} = '{val}'. Matching stored values "
                        f"in {table}.{col}: {found}")
                else:
                    sample = db.run(
                        f"SELECT DISTINCT \"{col}\" FROM \"{table}\" LIMIT 5")
                    evidence.append(
                        f"- You filtered {col} = '{val}' but NO stored value in "
                        f"{table}.{col} contains '{val}'. Sample of actual values "
                        f"there: {sample}. The value may be in a different column.")
            except Exception:
                continue
    return "\n".join(evidence)


def _wrong_db(state: AgentState, why: str):
    """Mark the current database as wrong and ask the graph to re-route."""
    db = state["db_id"]
    banned = list(state.get("banned_dbs") or []) + [db]
    print(f"--- WRONG DATABASE ({db}): {why} -> re-routing ---")
    return {
        "error": "wrong_db",
        "reroutes": (state.get("reroutes") or 0) + 1,
        "banned_dbs": banned,
        "attempts": 0,          # fresh retry budget on the new database
        "messages": [("user",
            f"The database '{db}' cannot answer this question ({why}). "
            f"Ignore it; a different database will be used.")],
    }


def execute_query(state: AgentState):
    print("--- Executing query ---")
    query = state["query"]

    lowered = query.lower()
    if any(word in lowered for word in FORBIDDEN):
        print(f"--- BLOCKED forbidden query: {query} ---")
        return {"error": "yes",
                "messages": [("user", "That query contains a forbidden operation. "
                                      "Write a SELECT-only query.")]}

    # the writer declared this schema cannot answer the question
    if "SCHEMA_MISMATCH" in query:
        if (state.get("reroutes") or 0) < MAX_REROUTES:
            return _wrong_db(state, "schema mismatch declared by the SQL writer")
        return {"result": "", "error": "no",
                "messages": [("assistant",
                    "I couldn't find a database that answers that question.")]}

    try:
        print(query)
        result = get_db(state["db_id"]).run(query)

        if len(result) > MAX_RESULT_CHARS:
            result = (result[:MAX_RESULT_CHARS]
                      + f"\n...(truncated - full result was {len(result)} chars)")

        print(f"Result: {result[:200]}")

        # TIER 2: empty result + literal filters -> probe, retry informed
        if (str(result).strip() in ("", "[]", "()")
                and state["attempts"] < 2):
            filters = extract_literal_filters(query)
            if filters:
                print("--- Empty result: probing actual stored values ---")
                evidence = probe_values(state["db_id"], query, filters)
                if evidence:
                    print(evidence)
                    return {
                        "result": result,
                        "error": "empty_suspicious",
                        "messages": [("user",
                            "Your query returned ZERO rows. I checked the "
                            "database for the values you filtered on:\n"
                            f"{evidence}\n"
                            "Rewrite the query using the ACTUAL stored values "
                            "or the correct column.")],
                    }

        return {"result": result, "error": "no"}

    except Exception as e:
        msg = str(e).lower()
        # DETERMINISTIC wrong-database detection - but only AFTER the model has
        # had a chance to fix a hallucinated table/column name on this db.
        # (Re-routing on the first "no such table" discards correct databases
        #  over recoverable typos: it cost 3 points on the eval.)
        if (any(sig in msg for sig in WRONG_DB_SIGNALS)
                and state["attempts"] >= 2                       # ← NEW
                and (state.get("reroutes") or 0) < MAX_REROUTES):
            return _wrong_db(state, str(e).split("\n")[0])

        print(f"--- Query failed: {e} ---")
        return {"error": "yes",
                "messages": [("user", f"The query failed with this error: {e}\n"
                                      f"Rewrite the query to fix it.")]}


def answer(state: AgentState):
    print("--- Answering ---")
    if not state.get("db_id"):
        return {"messages": [("assistant",
            "I couldn't find a database that answers that question.")]}

    result = state["result"]
    if result is None or str(result).strip() in ("", "[]", "()"):
        result = "(empty - no rows matched)"

    prompt = f"""Answer the user's question using the query result below.

Question: {state['question']}
SQL query used: {state['query']}
Query result: {result}

Instructions:
- Answer directly and concretely, stating the ACTUAL values from the result.
- Do not tell the user to "refer to the query result" - the values ARE your answer.
- Format numbers and currency cleanly for a human reader (e.g. $596,462) even
  if the raw result contains odd symbols or encoding artifacts.
- If the result is empty, say that no matching data was found.
- If the result was truncated or long, summarize it and show the first several entries.
- Only if the user asked to modify data: explain you can only read data, never modify it."""
    response = llm.invoke(prompt)
    return {"messages": [("assistant", response.content)]}


MAX_ATTEMPTS = 3

def route_after_execute(state: AgentState):
    err = state["error"]
    if err == "wrong_db":
        return "retrieve"                          # switch database, start over
    if err == "no":
        return "answer"
    if err == "empty_suspicious":
        if state["attempts"] < MAX_ATTEMPTS:
            return "write_query"
        return "answer"
    if state["attempts"] >= MAX_ATTEMPTS:
        print("--- Max attempts reached, giving up ---")
        return "give_up"
    return "write_query"


def give_up(state: AgentState):
    return {"messages": [("assistant",
        "I couldn't generate a working query after several attempts. "
        "Last error context is above. Try rephrasing the question.")]}


# ---------------------------------------------------------------
# Graph
# ---------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("gate", gate)
workflow.add_node("rewrite", rewrite)
workflow.add_node("retrieve", retrieve)
workflow.add_node("pick_db", pick_db)
workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("answer", answer)
workflow.add_node("give_up", give_up)

workflow.set_entry_point("gate")
workflow.add_conditional_edges("gate", route_after_gate,
    {"reject": END, "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "pick_db")
workflow.add_edge("pick_db", "write_query")
workflow.add_edge("write_query", "execute_query")
workflow.add_conditional_edges("execute_query", route_after_execute,
    {"answer": "answer",
     "write_query": "write_query",
     "retrieve": "retrieve",        # NEW: wrong database -> pick another
     "give_up": "give_up"})
workflow.add_edge("answer", END)
workflow.add_edge("give_up", END)

graph = workflow.compile().with_config({"callbacks": [langfuse_handler]})


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

def ask(question: str):
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
    final_state = None
    for event in graph.stream(initial, stream_mode="values"):
        final_state = event
    print("\n=== ANSWER ===")
    print(final_state["messages"][-1].content)
    print("=" * 50, "\n")
    return final_state


if __name__ == "__main__":
    ask("How many singers do we have?")
    ask("Which countries have a population over 100 million?")
    ask("why " * 400)                     # gate test: rejected pre-LLM

    from langfuse import get_client
    get_client().flush()


# ---------------------------------------------------------------
# Checkpointered graph for the Streamlit chat app
# ---------------------------------------------------------------


_ckpt_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
memory = SqliteSaver(_ckpt_conn)

chat_graph = workflow.compile(checkpointer=memory).with_config(
    {"callbacks": [langfuse_handler]})


def delete_conversation(thread_id: str):
    memory.delete_thread(thread_id)