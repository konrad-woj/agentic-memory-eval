"""
metrics.py — Three-layer evaluation metrics.

1. RETRIEVAL:  token-overlap recall/precision/F1 against required facts     (deterministic)
2. ACTION:     positive checks (tool+args) + negative checks (must_not_call) (deterministic)
3. ANSWER:     keyword contains + term overlap heuristic                    (deterministic)
               + optional DeepEval LLM-as-judge (Gemini/Ollama/OpenAI)      (LLM-judged)
"""

import os
import re
from dataclasses import dataclass, field

from scenarios import Difficulty, Task, ToolAssertion

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> set[str]:
    return set(text.lower().split())


# ════════════════════════════════════════════════════════════════════════
# 1. RETRIEVAL SCORING
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalScore:
    recall: float
    precision: float
    f1: float
    matched: list[str]
    missed: list[str]


def score_retrieval(retrieved: list[str], required_facts: list[str], threshold: float = 0.3) -> RetrievalScore:
    if not required_facts:
        return RetrievalScore(1.0, 1.0, 1.0, [], [])

    ret_token_sets = [_tokenize(r) for r in retrieved]
    req_token_sets = [_tokenize(r) for r in required_facts]

    matched, missed = [], []
    for req_text, req_tokens in zip(required_facts, req_token_sets):
        if not req_tokens:
            matched.append(req_text)
            continue
        found = any(len(req_tokens & rt) / len(req_tokens) >= threshold for rt in ret_token_sets)
        (matched if found else missed).append(req_text)

    relevant_retrieved = 0
    for rt in ret_token_sets:
        if any(len(rq & rt) / max(len(rq), 1) >= threshold for rq in req_token_sets):
            relevant_retrieved += 1

    recall = len(matched) / len(required_facts)
    precision = relevant_retrieved / len(retrieved) if retrieved else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return RetrievalScore(recall=recall, precision=precision, f1=f1, matched=matched, missed=missed)


# ════════════════════════════════════════════════════════════════════════
# 2. ACTION SCORING
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ActionCheck:
    passed: bool
    check_type: str   # "positive_call", "positive_args", "negative"
    description: str
    reason: str = ""


@dataclass
class ActionScore:
    total_checks: int
    passed_checks: int
    pass_rate: float
    checks: list[ActionCheck]
    # Breakdown
    positive_passed: int
    positive_failed: int
    negative_passed: int
    negative_failed: int


def score_actions(
    actual_actions: list[dict],
    expect_calls: list[ToolAssertion],
    expect_not_called: list[str],
) -> ActionScore:
    """
    Score agent actions with clean separation of positive and negative checks.

    Positive: each ToolAssertion in expect_calls must match an actual call + args.
    Negative: each tool in expect_not_called must NOT appear in actual calls.
    """
    checks: list[ActionCheck] = []
    pos_pass = pos_fail = neg_pass = neg_fail = 0

    calls_by_tool: dict[str, list[dict]] = {}
    for a in actual_actions:
        calls_by_tool.setdefault(a["tool"], []).append(a)
    actual_tool_names = set(calls_by_tool)

    # ── Positive checks ──
    for assertion in expect_calls:
        matching = calls_by_tool.get(assertion.tool_name, [])
        if not matching:
            checks.append(ActionCheck(
                passed=False, check_type="positive_call",
                description=assertion.description or f"Call {assertion.tool_name}",
                reason=f"'{assertion.tool_name}' never called. Actual: {sorted(actual_tool_names)}",
            ))
            pos_fail += 1
            continue

        if not assertion.required_args:
            checks.append(ActionCheck(
                passed=True, check_type="positive_call",
                description=assertion.description or f"Call {assertion.tool_name}",
            ))
            pos_pass += 1
            continue

        # Check args: at least one matching call must have all required k=v
        arg_ok = False
        for call in matching:
            actual_args = call.get("args", {})
            if all(
                exp_v.lower() in actual_args.get(exp_k.lower(), "")
                or actual_args.get(exp_k.lower(), "") in exp_v.lower()
                for exp_k, exp_v in assertion.required_args.items()
            ):
                arg_ok = True
                break

        if arg_ok:
            checks.append(ActionCheck(
                passed=True, check_type="positive_args",
                description=assertion.description or f"{assertion.tool_name} with correct args",
            ))
            pos_pass += 1
        else:
            actual_arg_strs = [c.get("args", {}) for c in matching]
            checks.append(ActionCheck(
                passed=False, check_type="positive_args",
                description=assertion.description or f"{assertion.tool_name} with correct args",
                reason=f"Args mismatch. Expected {assertion.required_args}, got {actual_arg_strs}",
            ))
            pos_fail += 1

    # ── Negative checks ──
    for banned_tool in expect_not_called:
        if banned_tool.lower() in actual_tool_names:
            checks.append(ActionCheck(
                passed=False, check_type="negative",
                description=f"Must NOT call {banned_tool}",
                reason=f"Forbidden tool '{banned_tool}' was called",
            ))
            neg_fail += 1
        else:
            checks.append(ActionCheck(
                passed=True, check_type="negative",
                description=f"Correctly avoided {banned_tool}",
            ))
            neg_pass += 1

    total = len(checks)
    passed = pos_pass + neg_pass

    return ActionScore(
        total_checks=total,
        passed_checks=passed,
        pass_rate=passed / total if total > 0 else 1.0,
        checks=checks,
        positive_passed=pos_pass,
        positive_failed=pos_fail,
        negative_passed=neg_pass,
        negative_failed=neg_fail,
    )


# ════════════════════════════════════════════════════════════════════════
# 3. ANSWER SCORING
# ════════════════════════════════════════════════════════════════════════

_STOPS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for",
    "and", "or", "that", "this", "with", "it", "its", "by", "on", "at",
    "be", "has", "had", "not", "no", "but", "from", "as", "we", "our",
})


@dataclass
class AnswerScore:
    correctness: float
    contains_score: float
    method: str
    explanation: str


def score_answer_heuristic(answer: str, expected: str, must_contain: list[str]) -> AnswerScore:
    answer_lower = answer.lower()

    found = sum(1 for term in must_contain if term.lower() in answer_lower)
    contains_score = found / len(must_contain) if must_contain else 1.0

    exp_terms = {w for w in re.findall(r"[\w$@.]+", expected.lower()) if w not in _STOPS and len(w) > 1}
    ans_terms = {w for w in re.findall(r"[\w$@.]+", answer_lower) if w not in _STOPS and len(w) > 1}
    overlap = len(exp_terms & ans_terms) / len(exp_terms) if exp_terms else 1.0

    correctness = 0.4 * contains_score + 0.6 * overlap

    return AnswerScore(
        correctness=correctness,
        contains_score=contains_score,
        method="heuristic",
        explanation=f"Contains {found}/{len(must_contain)} keywords. Term overlap: {overlap:.0%}",
    )


# ════════════════════════════════════════════════════════════════════════
# DeepEval judge factory
# ════════════════════════════════════════════════════════════════════════

def get_deepeval_judge():
    """Build a DeepEval judge model from DEEPEVAL_JUDGE env var."""
    backend = os.getenv("DEEPEVAL_JUDGE", "ollama")
    model_name = os.getenv("DEEPEVAL_JUDGE_MODEL", "")

    if backend == "gemini":
        from deepeval.models import GeminiModel
        return GeminiModel(model=model_name or "gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

    if backend == "openai":
        return model_name or "gpt-4o-mini"

    # Default: Ollama
    from deepeval.models import DeepEvalBaseLLM

    class OllamaJudge(DeepEvalBaseLLM):
        def __init__(self):
            self.model = model_name or os.getenv("LLM_MODEL", "llama3.1:8b")
            self._client = None

        def load_model(self):
            if not self._client:
                from langchain_ollama import ChatOllama
                self._client = ChatOllama(model=self.model, base_url=OLLAMA_BASE_URL, temperature=0)
            return self._client

        def generate(self, prompt: str, schema=None) -> str:
            llm = self.load_model()
            if schema:
                return llm.with_structured_output(schema).invoke(prompt)
            return llm.invoke(prompt).content

        async def a_generate(self, prompt: str, schema=None) -> str:
            llm = self.load_model()
            if schema:
                return await llm.with_structured_output(schema).ainvoke(prompt)
            return (await llm.ainvoke(prompt)).content

        def get_model_name(self) -> str:
            return f"ollama/{self.model}"

    return OllamaJudge()


# ════════════════════════════════════════════════════════════════════════
# Aggregate dataclasses
# ════════════════════════════════════════════════════════════════════════

@dataclass
class TaskResult:
    scenario: str
    instruction: str
    difficulty: str
    retrieval: RetrievalScore
    action: ActionScore
    answer: AnswerScore
    latency_s: float
    raw_answer: str
    tool_calls: list[dict]


@dataclass
class E2EScore:
    agent_name: str
    scenario_name: str
    avg_retrieval_recall: float
    action_pass_rate: float
    total_action_checks: int
    passed_action_checks: int
    positive_failed: int
    negative_failed: int
    avg_correctness: float
    avg_contains: float
    avg_latency_s: float
    store_latency_s: float


def aggregate_e2e(agent_name: str, scenario_name: str, results: list[TaskResult], store_latency: float) -> E2EScore:
    n = len(results) or 1
    return E2EScore(
        agent_name=agent_name,
        scenario_name=scenario_name,
        avg_retrieval_recall=sum(r.retrieval.recall for r in results) / n,
        action_pass_rate=sum(r.action.pass_rate for r in results) / n,
        total_action_checks=sum(r.action.total_checks for r in results),
        passed_action_checks=sum(r.action.passed_checks for r in results),
        positive_failed=sum(r.action.positive_failed for r in results),
        negative_failed=sum(r.action.negative_failed for r in results),
        avg_correctness=sum(r.answer.correctness for r in results) / n,
        avg_contains=sum(r.answer.contains_score for r in results) / n,
        avg_latency_s=sum(r.latency_s for r in results) / n,
        store_latency_s=store_latency,
    )
