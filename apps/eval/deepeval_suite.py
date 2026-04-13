"""
deepeval_suite.py — LLM-as-judge evaluation layer (Gemini / Ollama / OpenAI).

Runs on top of the deterministic action checks. Adds nuanced correctness
and faithfulness scoring via DeepEval metrics.

Usage:
    python deepeval_suite.py                  # uses DEEPEVAL_JUDGE from .env
    python deepeval_suite.py --judge gemini   # override
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

from metrics import get_deepeval_judge


async def run(judge_override: str | None = None):
    from deepeval import evaluate
    from deepeval.metrics import GEval, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    from agents import BaselineAgent, CogneeAgent
    from scenarios import SCENARIOS

    if judge_override:
        os.environ["DEEPEVAL_JUDGE"] = judge_override

    judge = get_deepeval_judge()
    label = judge if isinstance(judge, str) else judge.get_model_name()
    print(f"Judge: {label}\n")

    correctness = GEval(
        name="Correctness",
        criteria="Score 1.0 if the output correctly addresses the task. 0.5 for partial. 0 for wrong or missing action.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5, model=judge,
    )
    faithfulness = FaithfulnessMetric(threshold=0.5, model=judge, include_reason=True)

    for name, cls in [("Cognee KG", CogneeAgent), ("Baseline", BaselineAgent)]:
        print(f"\n{'='*50}\n{name}\n{'='*50}")
        cases = []
        for scenario in SCENARIOS:
            agent = cls(user_id=f"de_{scenario.name}")
            await agent.store_facts(scenario.seed_facts)
            for task in scenario.tasks:
                r = await agent.execute_task(task.instruction)
                cases.append(LLMTestCase(
                    input=task.instruction, actual_output=r["answer"],
                    expected_output=task.expected_answer, retrieval_context=r["retrieved_context"],
                    context=task.required_facts,
                ))
            await agent.reset()
        print(f"Scoring {len(cases)} tasks...")
        evaluate(test_cases=cases, metrics=[correctness, faithfulness])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--judge", choices=["gemini", "ollama", "openai"], default=None)
    asyncio.run(run(p.parse_args().judge))
