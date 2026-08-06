# thread_registry.py
import sqlite3
from datetime import datetime

DB = "checkpoints.db"   # same file the checkpointer uses — one db, tidy

def _conn():
    return sqlite3.connect(DB, check_same_thread=False)


def init_registry():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT)""")

def register_thread(thread_id: str, title: str):
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO chat_threads VALUES (?,?,?)",
                  (thread_id, title[:60], datetime.now().isoformat()))

def list_threads():
    with _conn() as c:
        rows = c.execute("""SELECT thread_id, title FROM chat_threads
                            ORDER BY created_at DESC""").fetchall()
    return rows   # [(thread_id, title), ...]

def delete_thread_record(thread_id: str):
    with _conn() as c:
        c.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))


def delete_conversation(thread_id: str):
    with _conn() as c:  # the same sqlite3 connection the saver uses
        c.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        c.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))