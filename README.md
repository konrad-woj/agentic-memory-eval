# agentic-memory-eval

Benchmarks two memory backends for an LLM agent that manages client accounts: a **keyword-overlap in-memory baseline** versus a **Cognee knowledge-graph** backend. Both agents share identical LLM, prompt, and tool-call logic — only the memory layer differs.

The core question: does a knowledge graph that automatically extracts entities and relationships from raw facts lead to better agent decisions compared to a simple keyword-overlap retriever — and at what latency cost?

## Why Cognee

Several graph-memory frameworks were evaluated (Graphiti, FalkorDB GraphRAG SDK, Cognee). Cognee was chosen as the KG backend because:

- **Zero infrastructure by default** — SQLite + LanceDB + Kuzu are all file-based; `pip install cognee` is the entire setup
- **Fully offline** — runs with Ollama, no cloud API keys required
- **Auto-extracts structure** — the ECL pipeline (Extract, Cognify, Load) discovers entities and relationships from raw text without manual schema definition; a flat key-value store can't answer "which of our healthcare clients have expiring contracts?" without you writing that join logic
- **Swappable backends** — graph DB, vector DB, and relational layer can be replaced independently without changing agent code

The baseline (`BaselineAgent`) uses a Python dict and keyword overlap scoring — representative of what you'd get with `LangGraph PostgresStore` using simple document retrieval, without pgvector or relationship extraction.

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                  Three-Layer Eval Harness                    │
│       Retrieval (F1) + Action (pass/fail) + Answer           │
└──────────┬───────────────────┬──────────────────┬───────────┘
           │                   │                  │
  ┌────────▼───────┐  ┌────────▼──────┐  ┌───────▼────────┐
  │   Cognee KG    │  │    Vector     │  │   InMemory     │
  │   Agent        │  │   Agent       │  │   Baseline     │
  └────────┬───────┘  └────────┬──────┘  └───────┬────────┘
           │   shared tools:   │                  │
           │   send_email      │                  │
           │   create_ticket   │                  │
           │   schedule_meeting│                  │
           │   escalate        │                  │
           │   update_crm      │                  │
           │   no_action       │                  │
           │                   │                  │
  ┌────────▼───────────────────▼──────────────────▼────────┐
  │                   Ollama (llama3.1)                     │
  └─────────────────────────────────────────────────────────┘
```

```text
agentic-memory-eval/
├── agents.py                   # BaselineAgent + VectorAgent + CogneeAgent (shared base)
├── .env.template               # copy to .env and fill in keys
├── apps/eval/
│   ├── scenarios.py            # 9 eval scenarios, 18 tasks, ground truth
│   ├── metrics.py              # deterministic scoring: retrieval / action / answer
│   ├── run_eval.py             # main evaluation harness (CLI)
│   ├── deepeval_suite.py       # LLM-as-judge layer (DeepEval + Gemini/Ollama/OpenAI)
│   └── profiler.py             # latency comparison
```

### Agents

| Agent | Memory backend | Store | Retrieve |
| --- | --- | --- | --- |
| `BaselineAgent` | Python dict | O(1) append | keyword overlap scoring |
| `VectorAgent` | In-memory numpy array | sentence-transformers encode | cosine similarity (`BAAI/bge-large-en-v1.5`) |
| `CogneeAgent` | Cognee knowledge graph (LanceDB vectors) | `cognee.add` + `cognee.cognify` | `SearchType.GRAPH_COMPLETION` |

`VectorAgent` is the key addition: it isolates what dense embeddings alone contribute versus the graph structure, using `BAAI/bge-large-en-v1.5` (top-tier MTEB retrieval, ~1.3 GB download on first run, configurable via `VECTOR_EMBEDDING_MODEL` env var).

All agents expose the same interface: `store_facts(facts)`, `retrieve(query)`, `execute_task(instruction)`, `reset()`.

### Evaluation metrics

Three layers, all deterministic unless the optional LLM judge is enabled:

| Layer | What it measures | Method |
| --- | --- | --- |
| **Retrieval** | Did memory return the right facts? | Token-overlap recall/precision/F1 |
| **Action** | Did the agent call the right tools with the right args? | Hard pass/fail per assertion |
| **Answer** | Was the final response correct? | Keyword coverage + term overlap; optional LLM judge |

The action layer is the headline metric. It checks:

- Was the right tool called? (e.g. `create_ticket`, not `send_email`)
- Were the args correct? (e.g. `priority=P1`, not `P3`)
- Were forbidden tools NOT called? (e.g. no escalation for low-priority issues)
- Was `no_action` chosen when no action was needed?

The optional `deepeval_suite.py` adds LLM-as-judge **Correctness** (GEval) and **Faithfulness** metrics on top.

### Scenarios

| Scenario | Tasks | Difficulty | What it tests |
| --- | --- | --- | --- |
| `basic_recall_action` | 2 | single-hop | Direct fact → single tool call; ignores distractor facts |
| `multi_hop_routing` | 2 | multi-hop | Chain facts across entities to derive priority + assignee |
| `temporal_updates` | 2 | temporal | Prefer newest fact (updated contact/ticket) over older one |
| `cross_entity_decisions` | 2 | cross-entity | Synthesise across 3 clients → different actions per client |
| `negative_distractor` | 1 | negative | Correctly do nothing when conditions are not met |
| `ambiguous_crm_update` | 2 | ambiguous | Apply multiple policies from underspecified instruction |
| `churn_risk_triage` | 2 | multi-hop | Connect satisfaction + ticket count + policy to flag at-risk accounts |
| `account_upgrade_chain` | 3 | multi-hop | Apply chained upgrade policies triggered by a funding milestone |
| `sla_incident_response` | 2 | temporal | Derive SLA tier from incident timestamps; apply correct follow-up policy |

## Setup (Mac)

```bash
# 1. Ollama
brew install ollama && ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2. Python environment
pip install uv          # or: python3 -m venv .venv && source .venv/bin/activate
uv sync                 # installs all dependencies from pyproject.toml

# 3. Config
cp .env.template .env
# Edit .env — add GOOGLE_API_KEY if using Gemini as judge
```

### `.env` reference

`.env.template` is included in the repo — copy it to `.env` and adjust as needed:

```env
# Agent LLM (runs locally via Ollama)
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text

# Cognee config
LLM_API_KEY=ollama
LLM_PROVIDER=ollama

# DeepEval judge — pick one:
# Option A: Gemini (recommended — free tier, good quality)
GOOGLE_API_KEY=your-gemini-api-key
DEEPEVAL_JUDGE=gemini
DEEPEVAL_JUDGE_MODEL=gemini-2.5-flash

# Option B: Ollama (fully offline, lower judge quality)
# DEEPEVAL_JUDGE=ollama
# DEEPEVAL_JUDGE_MODEL=llama3.1:8b

# Option C: OpenAI (best quality, costs money)
# OPENAI_API_KEY=sk-...
# DEEPEVAL_JUDGE=openai
# DEEPEVAL_JUDGE_MODEL=gpt-4o-mini
```

### Judge options

| Judge | Quality | Cost | Requirement |
| --- | --- | --- | --- |
| Gemini Flash (recommended) | Excellent | Free tier | `GOOGLE_API_KEY` in `.env` |
| Ollama | Decent | Free | Already running |
| GPT-4o-mini | Best | ~$0.01/eval | `OPENAI_API_KEY` in `.env` |

## Usage

All commands run from the **project root**. The `PYTHONPATH=.` prefix is required so `apps/eval/` scripts can import `agents.py` from the root.

### Run the full evaluation

```bash
PYTHONPATH=. python apps/eval/run_eval.py
```

Filter by scenario name or agent:

```bash
# Only the temporal scenario
PYTHONPATH=. python apps/eval/run_eval.py --scenario temporal

# Only the baseline agent (no Cognee or model download required)
PYTHONPATH=. python apps/eval/run_eval.py --agent baseline

# Only the vector agent (downloads BAAI/bge-large-en-v1.5 on first run, ~1.3 GB)
PYTHONPATH=. python apps/eval/run_eval.py --agent vector

# Combine filters
PYTHONPATH=. python apps/eval/run_eval.py --scenario multi_hop --agent cognee
```

Results are printed as Rich tables and saved to `eval_results/results_<timestamp>.json`.

### Latency profiler

```bash
PYTHONPATH=. python apps/eval/profiler.py
```

Compares store, retrieve, and `execute_task` latencies between both backends across a fixed set of facts and queries.

### LLM-as-judge (DeepEval)

```bash
PYTHONPATH=. python apps/eval/deepeval_suite.py
# override the judge backend at runtime
PYTHONPATH=. python apps/eval/deepeval_suite.py --judge gemini
PYTHONPATH=. python apps/eval/deepeval_suite.py --judge ollama
```

Runs GEval correctness and faithfulness metrics on every task for both agents.

## Example output

```text
Cognee KG vs InMemory Baseline
LLM: llama3.1:8b | Scenarios: 9 | Tasks: 18

─────────────────────────────────────────────────────────────────
Cognee KG
  basic_recall_action (Cognee KG)
  Stored 4 facts in 12.3s
  Task 1: It's May 16, 2025. Check if any contract renewals nee...
    ✅ [positive_args] Email Acme contact about renewal
    ✅ [negative] Correctly avoided escalate
    Retrieval: 100% | Contains: 100% | 4.2s
  ...

ACTION CHECKS (deterministic)
 Metric             Cognee KG   Baseline
 Action Pass Rate   82%         61%
 Total Checks       38          38
 Passed             31          23
 ...
```

## Adding scenarios

Add a `Scenario` to `SCENARIOS` in [apps/eval/scenarios.py](apps/eval/scenarios.py). Each task requires:

- `instruction` — the agent's input
- `expect_calls` — list of `ToolAssertion(tool_name, required_args, description)` that must be called
- `expect_not_called` — list of tool names that must not appear
- `required_facts` — facts the retriever must surface (used for recall scoring)
- `expected_answer_contains` — keywords the answer must include
- `difficulty` — one of `SINGLE_HOP`, `MULTI_HOP`, `TEMPORAL`, `CROSS_ENTITY`, `NEGATIVE`, `AMBIGUOUS`

```python
Task(
    instruction="Handle the outstanding HealthFirst issue.",
    expect_calls=[
        ToolAssertion("create_ticket", {"priority": "P1", "assignee": "marcus"}, "P1 to on-call"),
        ToolAssertion("send_email", {"to": "dr.chen@healthfirst.com"}, "Notify client"),
    ],
    expect_not_called=["escalate"],
    expected_answer_contains=["healthfirst", "critical", "marcus"],
    required_facts=["HealthFirst reported critical bug...", "Current on-call: Marcus..."],
    difficulty=Difficulty.MULTI_HOP,
)
```
