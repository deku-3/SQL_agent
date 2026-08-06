# app.py
# Streamlit chat UI for the Spider multi-database SQL agent
# Run: streamlit run app.py

import uuid
import streamlit as st
from SQL_agent.src.sql_ag import chat_graph as graph, delete_conversation
from SQL_agent.src.thread_registry import init_registry, register_thread, list_threads, delete_thread_record
from langchain_openai import ChatOpenAI
from langfuse import get_client

_title_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def make_title(question: str) -> str:
    """3-4 word chat title from the first question; falls back to raw text."""
    try:
        resp = _title_llm.invoke(
            "Summarize this database question as a 3-4 word title. "
            "Reply with ONLY the title - no quotes, no punctuation at the end.\n\n"
            f"Question: {question}"
        )
        title = resp.content.strip().strip('"').strip()
        # guard: model rambled or returned junk -> fall back to the question
        if 0 < len(title) <= 40:
            return title
    except Exception:
        pass
    return question[:40]

st.title("💬 SQL Agent")
init_registry()

# ---------- session bootstrapping ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.is_new = True          # not registered yet (no messages)

config = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "metadata": {
        "langfuse_session_id": st.session_state.thread_id,   # groups all turns of a chat
        "langfuse_tags": ["spider-agent", "streamlit"],      # separates app traffic from eval runs
    },
}

# ---------- delete confirmation dialog ----------
@st.dialog("Delete chat?")
def confirm_delete(tid: str, title: str):
    st.write(f'Delete **"{title}"**? This cannot be undone.')
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, delete", type="primary", use_container_width=True):
            delete_thread_record(tid)
            delete_conversation(tid)
            if tid == st.session_state.thread_id:
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.is_new = True
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()          # closes the dialog

# ---------- SIDEBAR ----------
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.is_new = True
        st.rerun()

    st.divider()
    st.caption("Previous chats")

    for tid, title in list_threads():
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"▶ {title}" if tid == st.session_state.thread_id else title
            if st.button(label, key=f"open_{tid}", use_container_width=True):
                st.session_state.thread_id = tid
                st.session_state.is_new = False
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{tid}", use_container_width=True):
                confirm_delete(tid, title)

# ---------- render history from the checkpointer ----------
state = graph.get_state(config)
if state.values:
    msgs = state.values["messages"]
    for i, msg in enumerate(msgs):
        if msg.type == "human" and (
            msg.content.startswith("The query failed")
            or msg.content.startswith("That query contains a forbidden")
            ):
            continue
        if msg.type == "human":
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif msg.type == "ai":
            # "Generated SQL" (old format) or "Generated SQL [db_id]" (new format)
            if msg.content.startswith("Generated SQL"):
                continue                                  # skip as a bubble...
            with st.chat_message("assistant"):
                st.markdown(msg.content)
                # ...but recover the SQL breadcrumbs, attached to this answer:
                j = i - 1
                sql_blocks = []
                while j >= 0 and msgs[j].type == "ai" and msgs[j].content.startswith("Generated SQL"):
                    text = msgs[j].content
                    # strip "Generated SQL [db]: " or "Generated SQL: " prefix
                    text = text.split(":", 1)[1].strip() if ":" in text else text
                    # pull the db label out of "Generated SQL [db_id]" if present
                    header = msgs[j].content.split(":", 1)[0]     # "Generated SQL [flight_2]"
                    db_label = header.replace("Generated SQL", "").strip(" []")
                    sql_blocks.append((db_label, text))
                    j -= 1
                if sql_blocks:
                    db_label = sql_blocks[-1][0]          # oldest attempt's db (same for all)
                    title = f"🔍 SQL used — database: {db_label}" if db_label else "🔍 SQL used"
                    with st.expander(title):
                        for _, sql in reversed(sql_blocks):
                            st.code(sql, language="sql")

# ---------- chat input ----------
if question := st.chat_input("Ask a question about any database..."):
    # first message of a new thread → register it with the question as title
    if st.session_state.get("is_new", False):
        register_thread(st.session_state.thread_id, make_title(question))
        st.session_state.is_new = False

    # ---- live node-progress panel (replaces the generic spinner) ----
    NODE_LABELS = {
        "gate":          "🔍 Checking your question...",
        "rewrite":       "✍️ Understanding your question...",
        "retrieve":      "📚 Finding candidate databases...",
        "pick_db":       "🎯 Selecting the right database...",
        "write_query":   "🧠 Writing SQL...",
        "execute_query": "▶️ Running the query...",
        "answer":        "💬 Composing the answer...",
        "give_up":       "😞 Couldn't find a working query",
    }

    # NOTE: deliberately NO "db_id" / "candidates" keys here.
    # On a fresh thread they don't exist yet -> retrieve() routes normally.
    # On an existing thread the checkpointer keeps the saved db_id ->
    # sticky routing keeps follow-ups on the same database.
    # (Passing db_id="" would OVERWRITE the saved value and break this.)
    initial = {
        "question": question,
        "messages": [("user", question)],
        "error": "no",
        "attempts": 0,
        "query": "",
        "result": "",
    }

    with st.status("Working...", expanded=True) as status:
        for event in graph.stream(initial, config, stream_mode="updates"):
            node = list(event.keys())[0]                 # which node just finished
            st.write(NODE_LABELS.get(node, node))
            # enrich with context where we have it:
            update = event[node] or {}
            if node == "pick_db" and update.get("db_id"):
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;→ database: **{update['db_id']}**")
            if node == "write_query" and update.get("attempts"):
                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;→ attempt {update['attempts']}")
            if node == "execute_query" and update.get("error") == "empty_suspicious":
                st.write("&nbsp;&nbsp;&nbsp;&nbsp;→ 🔎 empty result — checking actual stored values...")
        status.update(label="Done ✅", state="complete", expanded=False)
    get_client().flush()
    st.rerun()   # history loop redraws everything from state