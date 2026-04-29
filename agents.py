"""
agents.py — Two LangGraph agent variants sharing a common base.

Only the memory backend differs. Everything else (LLM, tools, prompt, parsing) is shared.
"""

import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VECTOR_EMBEDDING_MODEL = os.getenv("VECTOR_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

TOOL_PROMPT = """You have these tools. Call them by writing TOOL_CALL: tool(args) on its own line.

TOOLS:
- send_email(to, subject, body) — Send an email
- create_ticket(title, priority, assignee) — Create a ticket. Priority: P1/P2/P3
- schedule_meeting(attendees, topic, date) — Schedule a meeting
- update_crm(company, field, value) — Update a CRM record
- escalate(issue, level) — Escalate an issue. Level: VP/director/manager
- no_action() — Explicitly do nothing

RULES:
- You MUST call at least one tool per task (even no_action if nothing is needed)
- Format: TOOL_CALL: tool_name(arg1=value1, arg2=value2)
- After tool calls, give a one-sentence summary
"""

TASK_PROMPT = """You are an AI agent managing client accounts.

{tool_prompt}

CONTEXT FROM MEMORY:
{context}

TASK: {instruction}

Think step by step, then call the appropriate tool(s)."""


# ═══════════════════════════════════════════════════════════════════════
# Action logging
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ActionRecord:
    tool_name: str
    args: dict[str, str]
    timestamp: float = field(default_factory=time.time)


class ActionLog:
    def __init__(self):
        self.actions: list[ActionRecord] = []

    def record(self, tool_name: str, **kwargs):
        self.actions.append(ActionRecord(
            tool_name=tool_name,
            args={k: str(v).lower() for k, v in kwargs.items()},
        ))

    def clear(self):
        self.actions.clear()

    def as_dicts(self) -> list[dict]:
        return [{"tool": a.tool_name, "args": a.args} for a in self.actions]


# ═══════════════════════════════════════════════════════════════════════
# Tool-call parsing
# ═══════════════════════════════════════════════════════════════════════

_TOOL_RE = re.compile(r"TOOL_CALL:\s*(\w+)\(([^)]*)\)", re.IGNORECASE)


def parse_tool_calls(response: str, log: ActionLog) -> list[dict]:
    """Extract TOOL_CALL lines, log them, return structured list."""
    calls = []
    for match in _TOOL_RE.finditer(response):
        tool = match.group(1).lower()
        args = {}
        for pair in match.group(2).split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                args[k.strip().lower()] = v.strip().strip("'\"").lower()
        log.record(tool, **args)
        calls.append({"tool": tool, "args": args})
    return calls


# ═══════════════════════════════════════════════════════════════════════
# Base agent (shared logic)
# ═══════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """Common interface for both memory backends."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.action_log = ActionLog()
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        return self._llm

    @abstractmethod
    async def store_facts(self, facts: list[str]) -> dict[str, Any]:
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        ...

    @abstractmethod
    async def reset(self) -> None:
        ...

    async def execute_task(self, instruction: str) -> dict[str, Any]:
        self.action_log.clear()
        t0 = time.time()

        retrieved = await self.retrieve(instruction)
        context = "\n".join(f"- {r}" for r in retrieved) if retrieved else "No relevant information found."

        prompt = TASK_PROMPT.format(
            tool_prompt=TOOL_PROMPT,
            context=context,
            instruction=instruction,
        )

        from langchain_core.messages import HumanMessage
        response = (await self._get_llm().ainvoke([HumanMessage(content=prompt)])).content
        tool_calls = parse_tool_calls(response, self.action_log)

        return {
            "answer": response,
            "tool_calls": tool_calls,
            "action_log": self.action_log.as_dicts(),
            "retrieved_context": retrieved,
            "latency_s": time.time() - t0,
        }


# ═══════════════════════════════════════════════════════════════════════
# Baseline: keyword-overlap memory
# ═══════════════════════════════════════════════════════════════════════

class BaselineAgent(BaseAgent):

    def __init__(self, user_id: str = "default"):
        super().__init__(user_id)
        self._memories: dict[str, str] = {}

    async def store_facts(self, facts: list[str]) -> dict[str, Any]:
        t0 = time.time()
        for fact in facts:
            self._memories[uuid.uuid4().hex[:8]] = fact
        return {"stored": len(facts), "latency_s": time.time() - t0}

    async def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        query_words = set(query.lower().split())
        scored = [
            (len(query_words & set(text.lower().split())), text)
            for text in self._memories.values()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:top_k] if score > 0]

    async def reset(self):
        self._memories.clear()
        self.action_log.clear()


# ═══════════════════════════════════════════════════════════════════════
# Vector: dense embedding retrieval (sentence-transformers, no graph)
# ═══════════════════════════════════════════════════════════════════════

class VectorAgent(BaseAgent):
    """Ablation: dense vector retrieval without graph structure.

    Answers the question: does Cognee win because of its graph structure, or
    simply because it uses better retrieval than keyword overlap?

    Same embedding quality as a modern retrieval system (BAAI/bge-large-en-v1.5,
    top MTEB retrieval, ~1.3 GB download on first run), but no entity extraction,
    no relationship graph, no multi-hop traversal.
    """

    def __init__(self, user_id: str = "default"):
        super().__init__(user_id)
        self._model = None
        self._facts: list[str] = []
        self._embeddings = None  # numpy array shape (n_facts, dim), L2-normalised

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(VECTOR_EMBEDDING_MODEL)
        return self._model

    async def store_facts(self, facts: list[str]) -> dict[str, Any]:
        import numpy as np
        t0 = time.time()
        self._facts = list(facts)
        self._embeddings = self._get_model().encode(
            self._facts, normalize_embeddings=True, show_progress_bar=False,
        )
        return {"stored": len(facts), "latency_s": time.time() - t0}

    async def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        if not self._facts or self._embeddings is None:
            return []
        import numpy as np
        q_emb = self._get_model().encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
        )
        # Cosine similarity = dot product of L2-normalised vectors
        scores = (self._embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self._facts[i] for i in top_idx if scores[i] > 0.25]

    async def reset(self) -> None:
        self._facts.clear()
        self._embeddings = None
        self.action_log.clear()


# ═══════════════════════════════════════════════════════════════════════
# Cognee: knowledge-graph memory
# ═══════════════════════════════════════════════════════════════════════

class CogneeAgent(BaseAgent):

    def __init__(self, user_id: str = "default"):
        super().__init__(user_id)
        self._initialized = False

    async def _init_cognee(self):
        if self._initialized:
            return
        import cognee
        cognee.config.set_llm_config({
            "llm_provider": "ollama",
            "llm_model": OLLAMA_MODEL,
            "llm_endpoint": OLLAMA_BASE_URL,
            "llm_api_key": "ollama",
        })
        cognee.config.set_vector_db_config({"vector_db_provider": "lancedb"})
        await cognee.prune.prune_data()
        await cognee.prune.prune_system()
        self._initialized = True

    async def store_facts(self, facts: list[str]) -> dict[str, Any]:
        import cognee
        await self._init_cognee()
        t0 = time.time()
        await cognee.add("\n".join(facts), dataset_name=f"user_{self.user_id}")
        await cognee.cognify()
        return {"stored": len(facts), "latency_s": time.time() - t0}

    async def retrieve(self, query: str, top_k: int = 10) -> list[str]:
        import cognee
        from cognee import SearchType
        try:
            results = await cognee.search(query_text=query, query_type=SearchType.GRAPH_COMPLETION)
            return [str(r) for r in (results or [])[:top_k]]
        except Exception as e:
            print(f"  [cognee retrieve error: {e}]")
            return []

    async def reset(self):
        import cognee
        try:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system()
        except Exception:
            pass
        self._initialized = False
        self.action_log.clear()
