# 🕸️ Text-to-SQL Agent

**Ask questions in plain English. Get answers from the right database — out of 166 of them.**

A LangGraph agent that routes natural-language questions across the full [Spider](https://yale-lily.github.io/spider) benchmark's databases, writes SQLite queries, executes them safely, and — when a query comes back empty — *investigates the actual data* before retrying.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-orange)
![OpenAI](https://img.shields.io/badge/gpt--4o--mini-LLM-black)

---

## 📊 Measured Results (Spider dev set, n = 100)

| Stage | Accuracy |
|---|---|
| **Database routing** (two-stage: embedding top-5 → LLM disambiguation) | **81%** |
| **End-to-end execution accuracy** (right answer from the right data) | **70%** |

Failure decomposition of routing errors: 12% judge errors (dominated by near-twin
databases like `flight_2` vs `flight_4` that can both answer the same question),
7% retrieval ceiling (gold DB never reached top-5 — largely questions whose
vocabulary is inherently ambiguous, e.g. *"find the total number of players"*
→ women's tennis).

> Every number he<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 560" font-family="Segoe UI, -apple-system, Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#8b949e"/>
    </marker>
    <marker id="arrowAmber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#d29922"/>
    </marker>
    <marker id="arrowGreen" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#3fb950"/>
    </marker>
    <marker id="arrowRed" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#f85149"/>
    </marker>
  </defs>

  <!-- background -->
  <rect width="1180" height="560" rx="14" fill="#0d1117"/>
  <rect x="1" y="1" width="1178" height="558" rx="13" fill="none" stroke="#21262d" stroke-width="2"/>

  <!-- title -->
  <text x="46" y="52" fill="#e6edf3" font-size="21" font-weight="700" letter-spacing="0.5">TEXT-TO-SQL AGENT</text>
  <text x="46" y="74" fill="#8b949e" font-size="12.5" letter-spacing="2.5">LANGGRAPH STATE MACHINE · ROUTE → GENERATE → EXECUTE → SELF-CORRECT</text>
  <line x1="46" y1="90" x2="1134" y2="90" stroke="#21262d" stroke-width="1"/>

  <!-- ============ ROW 1: main pipeline ============ -->

  <!-- user question pill -->
  <rect x="46" y="140" width="118" height="46" rx="23" fill="#161b22" stroke="#30363d" stroke-width="1.5"/>
  <text x="105" y="160" fill="#8b949e" font-size="11" text-anchor="middle">user</text>
  <text x="105" y="176" fill="#e6edf3" font-size="13" font-weight="600" text-anchor="middle">question</text>

  <!-- rewrite -->
  <rect x="204" y="128" width="150" height="70" rx="10" fill="#161b22" stroke="#388bfd" stroke-width="1.5"/>
  <text x="279" y="154" fill="#e6edf3" font-size="14" font-weight="600" text-anchor="middle">rewrite</text>
  <text x="279" y="172" fill="#8b949e" font-size="11" text-anchor="middle">resolve follow-ups</text>
  <text x="279" y="186" fill="#8b949e" font-size="11" text-anchor="middle">into standalone form</text>

  <!-- retrieve -->
  <rect x="394" y="128" width="150" height="70" rx="10" fill="#161b22" stroke="#388bfd" stroke-width="1.5"/>
  <text x="469" y="154" fill="#e6edf3" font-size="14" font-weight="600" text-anchor="middle">retrieve</text>
  <text x="469" y="172" fill="#8b949e" font-size="11" text-anchor="middle">top-5 databases</text>
  <text x="469" y="186" fill="#8b949e" font-size="11" text-anchor="middle">by vector similarity</text>

  <!-- pick_db -->
  <rect x="584" y="128" width="150" height="70" rx="10" fill="#161b22" stroke="#388bfd" stroke-width="1.5"/>
  <text x="659" y="154" fill="#e6edf3" font-size="14" font-weight="600" text-anchor="middle">pick_db</text>
  <text x="659" y="172" fill="#8b949e" font-size="11" text-anchor="middle">LLM judge over</text>
  <text x="659" y="186" fill="#8b949e" font-size="11" text-anchor="middle">candidate schemas</text>

  <!-- write_query -->
  <rect x="774" y="128" width="150" height="70" rx="10" fill="#1c2128" stroke="#a371f7" stroke-width="1.5"/>
  <text x="849" y="154" fill="#e6edf3" font-size="14" font-weight="600" text-anchor="middle">write_query</text>
  <text x="849" y="172" fill="#8b949e" font-size="11" text-anchor="middle">schema + few-shot</text>
  <text x="849" y="186" fill="#8b949e" font-size="11" text-anchor="middle">structured SQL out</text>

  <!-- main row arrows -->
  <line x1="164" y1="163" x2="200" y2="163" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="354" y1="163" x2="390" y2="163" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="544" y1="163" x2="580" y2="163" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="734" y1="163" x2="770" y2="163" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- routing accuracy tag -->
  <rect x="394" y="212" width="340" height="24" rx="12" fill="none" stroke="#21262d" stroke-width="1"/>
  <text x="564" y="228" fill="#8b949e" font-size="11" text-anchor="middle">two-stage routing · measured 81% on Spider dev (n=100)</text>

  <!-- ============ ROW 2: execute + branches ============ -->

  <!-- write_query down to execute_query -->
  <line x1="849" y1="198" x2="849" y2="290" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- execute_query (hexagon-ish, the decision heart) -->
  <path d="M 774 294 L 924 294 L 944 331 L 924 368 L 774 368 L 754 331 Z"
        fill="#1c2128" stroke="#d29922" stroke-width="1.8"/>
  <text x="849" y="322" fill="#e6edf3" font-size="14" font-weight="600" text-anchor="middle">execute_query</text>
  <text x="849" y="340" fill="#8b949e" font-size="11" text-anchor="middle">SELECT-only gate · runs SQL</text>
  <text x="849" y="354" fill="#8b949e" font-size="11" text-anchor="middle">result truncation</text>

  <!-- probe sidecar -->
  <rect x="980" y="286" width="154" height="90" rx="10" fill="#161b22" stroke="#d29922" stroke-width="1.2" stroke-dasharray="5 4"/>
  <text x="1057" y="310" fill="#d29922" font-size="12" font-weight="600" text-anchor="middle">🔎 evidence probe</text>
  <text x="1057" y="328" fill="#8b949e" font-size="10.5" text-anchor="middle">on suspicious empties:</text>
  <text x="1057" y="342" fill="#8b949e" font-size="10.5" text-anchor="middle">query the DB for what is</text>
  <text x="1057" y="356" fill="#8b949e" font-size="10.5" text-anchor="middle">ACTUALLY stored near each</text>
  <text x="1057" y="370" fill="#8b949e" font-size="10.5" text-anchor="middle">filtered value</text>
  <line x1="944" y1="331" x2="976" y2="331" stroke="#d29922" stroke-width="1.2" stroke-dasharray="5 4"/>

  <!-- retry loop: execute -> write_query (left arc, amber) -->
  <path d="M 754 331 C 660 331 660 231 770 178" fill="none" stroke="#d29922" stroke-width="1.6" marker-end="url(#arrowAmber)"/>
  <rect x="560" y="272" width="176" height="40" rx="8" fill="#0d1117" stroke="#d29922" stroke-width="1"/>
  <text x="648" y="289" fill="#d29922" font-size="11" font-weight="600" text-anchor="middle">retry · max 3 attempts</text>
  <text x="648" y="303" fill="#8b949e" font-size="10.5" text-anchor="middle">error text or probe evidence fed back</text>

  <!-- ok path: execute -> answer (green, down-left) -->
  <path d="M 800 368 C 700 430 560 445 470 449" fill="none" stroke="#3fb950" stroke-width="1.8" marker-end="url(#arrowGreen)"/>
  <text x="672" y="416" fill="#3fb950" font-size="11.5" font-weight="600">ok</text>

  <!-- give up path: execute -> give_up (red, down-right) -->
  <path d="M 898 368 C 940 405 960 425 968 442" fill="none" stroke="#f85149" stroke-width="1.6" marker-end="url(#arrowRed)"/>
  <text x="948" y="408" fill="#f85149" font-size="11.5" font-weight="600">attempts ≥ 3</text>

  <!-- ============ ROW 3: terminals ============ -->

  <!-- answer -->
  <rect x="316" y="426" width="150" height="66" rx="10" fill="#12261a" stroke="#3fb950" stroke-width="1.6"/>
  <text x="391" y="452" fill="#e6edf3" font-size="14" font-weight="600" text-anchor="middle">answer</text>
  <text x="391" y="470" fill="#8b949e" font-size="11" text-anchor="middle">concrete values from</text>
  <text x="391" y="483" fill="#8b949e" font-size="11" text-anchor="middle">actual result rows</text>

  <!-- give_up -->
  <rect x="900" y="446" width="140" height="56" rx="10" fill="#291415" stroke="#f85149" stroke-width="1.6"/>
  <text x="970" y="472" fill="#e6edf3" font-size="14" font-weight="600" text-anchor="middle">give_up</text>
  <text x="970" y="489" fill="#8b949e" font-size="11" text-anchor="middle">honest failure message</text>

  <!-- END pill -->
  <rect x="120" y="436" width="96" height="46" rx="23" fill="#161b22" stroke="#30363d" stroke-width="1.5"/>
  <text x="168" y="464" fill="#e6edf3" font-size="13" font-weight="600" text-anchor="middle">END</text>

  <!-- answer -> END -->
  <line x1="312" y1="459" x2="222" y2="459" stroke="#8b949e" stroke-width="1.5" marker-end="url(#arrow)"/>
  <!-- give_up -> END (long bottom line) -->
  <path d="M 900 486 C 600 530 300 520 190 486" fill="none" stroke="#8b949e" stroke-width="1.2" stroke-dasharray="4 4" marker-end="url(#arrow)"/>

  <!-- footer legend -->
  <text x="46" y="536" fill="#484f58" font-size="10.5">blue = routing · purple = generation · amber = execution + self-correction · green/red = terminal paths</text>
  <text x="1134" y="536" fill="#484f58" font-size="10.5" text-anchor="end">end-to-end execution accuracy: 70% (Spider dev, n=100)</text>
</svg>
re is reproducible: the eval harness lives in `test.py`.

---

## 🧠 Architecture
<img width="300" height="142" alt="architecture" src="https://github.com/user-attachments/assets/4a5205a3-9a3a-47c8-9754-97c9f5d323be" />


### The pipeline

1. **Rewrite** — follow-up questions (*"how many flights does each of **them** operate?"*) are rewritten into standalone questions using conversation history, so routing and SQL generation never see dangling pronouns.
2. **Route** — the question is embedded and matched against LLM-generated *database profiles* (domain, contents, example questions per DB) in ChromaDB → top-5 candidates.
3. **Disambiguate** — an LLM judge picks the right database from the candidates' full schemas.
4. **Write SQL** — gpt-4o-mini with structured output (Pydantic), the live schema, and similar solved examples retrieved from a 7,000-pair question→SQL bank.
5. **Execute** — behind a safety gate (SELECT-only, forbidden-keyword block, result truncation).
6. **Self-correct** — errors feed back verbatim for up to 3 attempts.

---

## 🔎 The interesting part: evidence-based retry

Most text-to-SQL agents retry on *errors*. This one also investigates *suspicious successes* — a query that runs fine but returns **zero rows** when it filtered on string literals.

Instead of guessing, the agent probes the database for what is *actually stored* near each filtered value and feeds back **facts**:

```
Your query returned ZERO rows. I checked the database for the values you filtered on:
- You filtered City = 'Aberdeen'. Matching stored values in airports.City:
  [('Aberdeen, MD',)]
Rewrite the query using the ACTUAL stored values or the correct column.
```

**Why this exists:** during evaluation, a perfectly correct query kept returning
empty. Investigation revealed Spider's `flight_2` stores airport codes with a
leading space (`' APG'`, not `'APG'`) — dirty data that defeats even the
benchmark's own gold queries. Literal string matching is the #1 real-world
text-to-SQL failure mode; this agent handles it structurally.

---

## 🛡️ Production-minded details

- **Safety gate** — SELECT-only enforcement before any query touches a database
- **Token-budget controls** — query results truncated before entering prompts; message history windowed in long chats; sticky routing skips redundant disambiguation calls on follow-ups
- **Honest empties** — an empty result is treated as suspicious *once*, then accepted as the true answer (no infinite self-doubt loops)
- **Two graphs, one workflow** — a memory-free graph for reproducible evals, a SQLite-checkpointed graph for the multi-turn chat app (`app.py`, Streamlit)

---

## 🗂️ Repository map

| File | What it does |
|---|---|
| `sql_ag.py` | The agent: LangGraph nodes, evidence-based retry, both compiled graphs |
| `disambiguate.py` | Two-stage routing (vector top-5 + LLM judge) |
| `vectorstore.py` | ChromaDB collections: database profiles + example bank |
| `ingest_db_docs.py` | Generates & embeds an LLM-written profile per database |
| `ingest_example.py` | Embeds 7,000 Spider question→SQL pairs for few-shot retrieval |
| `app.py` | Streamlit chat frontend (multi-turn, checkpointed) |
| `test.py` / `test_retireval.py` | Eval harnesses: routing accuracy & end-to-end execution accuracy |
| `config.py` | Paths to the Spider dataset |

---

## 🚀 Setup

```bash
git clone https://github.com/deku-3/SQL_agent.git
cd SQL_agent
pip install -r requirements.txt
```

1. Download the [Spider dataset](https://yale-lily.github.io/spider) and point `config.py` at it.
2. Create `.env` with your key:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. Build the retrieval layer (one-time, a few cents of API cost):
   ```bash
   python ingest_db_docs.py     # LLM-written profile per database
   python ingest_example.py     # 7,000-example few-shot bank
   ```
4. Ask questions:
   ```bash
   python sql_ag.py             # smoke tests in console
   streamlit run app.py         # chat UI
   ```

---

## 🧭 Roadmap

- [ ] Failure-bucket analysis of the 30% end-to-end misses
- [ ] Hybrid retrieval (BM25 + vectors) to attack the 7% routing ceiling
- [ ] Cache-aligned prompting (static schema prefix → ~40% input-cost cut on multi-turn chats)
- [ ] FastAPI + Docker deployment; MCP server wrapper so other agents can use this as a tool
- [ ] LangSmith tracing for one-click failure autopsies

---

## 📝 Lessons learned the hard way

- **Dirty data beats correct SQL** — a single leading space defeated both my agent and Spider's own gold queries. Value grounding > literal matching.
- **Decompose your error before fixing anything** — routing misses and generation misses need different medicine; measuring them separately (81% vs 70%) showed exactly where accuracy leaks.
- **Silent process death + no traceback = look below Python** — this project also survived a broken system `MSVCP140.dll` that crashed every vector-store write on the machine. Event Viewer, not print statements, found it.

---

*Built as a deep-dive into RAG, agent architecture, and evaluation discipline. Questions and issues welcome.*
