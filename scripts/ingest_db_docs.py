import json, sqlite3
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src.vectorstore import db_docs_store
from dotenv import load_dotenv
from src.config import TABLES_JSON,SPIDER_DB_DIR
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_sample_values(db_id: str, max_vals=12) -> str:
    """Grab a few distinct values from text columns to enrich the doc."""
    path = SPIDER_DB_DIR / db_id / f"{db_id}.sqlite"
    vals = []
    try:
        con = sqlite3.connect(path)
        cur = con.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for t in tables:
            cols = cur.execute(f'PRAGMA table_info("{t}")').fetchall()
            text_cols = [c[1] for c in cols if "CHAR" in (c[2] or "").upper()
                         or "TEXT" in (c[2] or "").upper()][:2]
            for col in text_cols:
                try:
                    rows = cur.execute(
                        f'SELECT DISTINCT "{col}" FROM "{t}" LIMIT 3').fetchall()
                    vals += [str(r[0]) for r in rows if r[0]]
                except Exception:
                    pass
        con.close()
    except Exception as e:
        print(f"  (values skipped for {db_id}: {e})")
    return ", ".join(vals[:max_vals])

def schema_text(db: dict) -> str:
    """Turn one tables.json entry into readable table: col, col lines."""
    lines = []
    for t_idx, t_name in enumerate(db["table_names_original"]):
        cols = [c_name for (tbl_i, c_name) in db["column_names_original"]
                if tbl_i == t_idx]
        lines.append(f"{t_name}: {', '.join(cols)}")
    return "\n".join(lines)

PROMPT = """Given this database schema and sample values, write a retrieval
document in EXACTLY this format (no extra text):

Database: {db_id}
Domain: <one sentence: what real-world domain this covers>
Contains: <one or two sentences summarizing entities and their attributes>
Example questions this answers: <4 short natural questions a user might ask>
Key columns: <10-15 meaningful column names, SKIP id/key columns>
Example values: <the provided sample values, lightly cleaned>

Schema:
{schema}

Sample values: {values}"""

def main():
    tables = json.load(open(TABLES_JSON, encoding="utf-8"))
    tables = tables[3:]
    docs = []
    for i, db in enumerate(tables):
        db_id = db["db_id"]
        print(f"[{i+1}/{len(tables)}] {db_id}")
        values = get_sample_values(db_id)
        response = llm.invoke(PROMPT.format(
            db_id=db_id, schema=schema_text(db), values=values))
        docs.append(Document(
            page_content=response.content,
            metadata={"db_id": db_id},
        ))
    db_docs_store.add_documents(docs)
    print(f"✅ Ingested {len(docs)} database docs")

if __name__ == "__main__":
    main()