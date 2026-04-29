"""
run_eval.py — Three-layer evaluation harness.

Usage:
    python run_eval.py                       # all scenarios, both agents
    python run_eval.py --scenario temporal   # filter by name
    python run_eval.py --agent baseline      # one agent only
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from agents import BaselineAgent, CogneeAgent, VectorAgent
from metrics import (
    E2EScore, TaskResult,
    aggregate_e2e, score_actions, score_answer_heuristic, score_retrieval,
)
from scenarios import SCENARIOS, Scenario, get_task_count


async def evaluate_task(agent, task, scenario_name: str = "") -> TaskResult:
    result = await agent.execute_task(task.instruction)
    answer = result["answer"]
    action_log = result["action_log"]
    return TaskResult(
        scenario=scenario_name,
        instruction=task.instruction,
        difficulty=task.difficulty.value,
        retrieval=score_retrieval(result["retrieved_context"], task.required_facts),
        action=score_actions(action_log, task.expect_calls, task.expect_not_called),
        answer=score_answer_heuristic(answer, task.expected_answer, task.expected_answer_contains),
        latency_s=result["latency_s"],
        raw_answer=answer,
        tool_calls=action_log,
    )


async def evaluate_scenario(
    scenario: Scenario, agent, agent_name: str, console,
) -> tuple[E2EScore, list[TaskResult]]:
    console.print(f"\n  [bold cyan]{scenario.name}[/bold cyan] ({agent_name})")

    store_result = await agent.store_facts(scenario.seed_facts)
    store_lat = store_result["latency_s"]
    console.print(f"  Stored {len(scenario.seed_facts)} facts in {store_lat:.1f}s")

    task_results = []
    for i, task in enumerate(scenario.tasks):
        console.print(f"\n  Task {i+1}: {task.instruction[:65]}...")
        tr = await evaluate_task(agent, task, scenario.name)
        task_results.append(tr)

        a = tr.action
        color = "green" if a.pass_rate == 1.0 else ("yellow" if a.pass_rate >= 0.5 else "red")
        console.print(f"    [{color}]ACTIONS: {a.passed_checks}/{a.total_checks} ({a.pass_rate:.0%})[/{color}]")
        for c in a.checks:
            icon = "✅" if c.passed else "❌"
            console.print(f"      {icon} [{c.check_type}] {c.description}")
            if not c.passed:
                console.print(f"         [dim red]{c.reason}[/dim red]")

        console.print(f"    Retrieval: {tr.retrieval.recall:.0%} | Contains: {tr.answer.contains_score:.0%} | {tr.latency_s:.1f}s")

    return aggregate_e2e(agent_name, scenario.name, task_results, store_lat), task_results


def print_report(all_e2e: dict[str, list[E2EScore]], all_tasks: dict[str, list[TaskResult]]):
    from rich.console import Console
    from rich.table import Table
    console = Console()

    console.print("\n" + "=" * 75)
    console.print("[bold white on blue]  EVALUATION REPORT  [/bold white on blue]")
    console.print("=" * 75)

    # ── Action pass rate (headline) ──
    t = Table(title="ACTION CHECKS (deterministic)")
    t.add_column("Metric", style="bold")
    for name in all_e2e:
        t.add_column(name, justify="center")

    metrics = [
        ("Action Pass Rate", lambda ss: f"{sum(s.action_pass_rate for s in ss)/len(ss):.0%}"),
        ("Total Checks", lambda ss: str(sum(s.total_action_checks for s in ss))),
        ("Passed", lambda ss: str(sum(s.passed_action_checks for s in ss))),
        ("Positive Failed", lambda ss: str(sum(s.positive_failed for s in ss))),
        ("Negative Failed", lambda ss: str(sum(s.negative_failed for s in ss))),
    ]
    for label, fn in metrics:
        t.add_row(label, *[fn(scores) for scores in all_e2e.values()])
    console.print(t)

    # ── Retrieval + Answer ──
    t2 = Table(title="RETRIEVAL & ANSWER")
    t2.add_column("Metric", style="bold")
    for name in all_e2e:
        t2.add_column(name, justify="center")

    for label, attr, fmt in [
        ("Retrieval Recall", "avg_retrieval_recall", "{:.0%}"),
        ("Answer Correctness", "avg_correctness", "{:.0%}"),
        ("Contains Score", "avg_contains", "{:.0%}"),
        ("Avg Latency", "avg_latency_s", "{:.1f}s"),
        ("Store Latency", "store_latency_s", "{:.1f}s"),
    ]:
        row = [label]
        for scores in all_e2e.values():
            val = sum(getattr(s, attr) for s in scores) / max(len(scores), 1)
            row.append(fmt.format(val))
        t2.add_row(*row)
    console.print(t2)

    # ── By difficulty ──
    t3 = Table(title="ACTION PASS RATE BY DIFFICULTY")
    t3.add_column("Difficulty", style="bold")
    for name in all_e2e:
        t3.add_column(name, justify="center")

    diffs = sorted({tr.difficulty for trs in all_tasks.values() for tr in trs})
    for diff in diffs:
        row = [diff]
        for trs in all_tasks.values():
            matching = [tr for tr in trs if tr.difficulty == diff]
            rate = sum(tr.action.pass_rate for tr in matching) / len(matching) if matching else 0
            row.append(f"{rate:.0%}")
        t3.add_row(*row)
    console.print(t3)

    # ── Per scenario ──
    t4 = Table(title="PER-SCENARIO BREAKDOWN")
    t4.add_column("Scenario")
    t4.add_column("Agent", style="dim")
    t4.add_column("Action %", justify="center")
    t4.add_column("Retrieval", justify="center")
    t4.add_column("Latency", justify="center")

    for agent_name, scores in all_e2e.items():
        for s in scores:
            t4.add_row(s.scenario_name, agent_name, f"{s.action_pass_rate:.0%}",
                       f"{s.avg_retrieval_recall:.0%}", f"{s.avg_latency_s:.1f}s")
    console.print(t4)


def save_results(all_tasks: dict[str, list[TaskResult]]):
    Path("eval_results").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"eval_results/results_{ts}.json"

    out = {
        agent: [
            {
                "scenario": tr.scenario,
                "instruction": tr.instruction,
                "difficulty": tr.difficulty,
                "retrieval_recall": tr.retrieval.recall,
                "action_pass_rate": tr.action.pass_rate,
                "action_checks": [
                    {"passed": c.passed, "type": c.check_type, "desc": c.description, "reason": c.reason}
                    for c in tr.action.checks
                ],
                "answer_correctness": tr.answer.correctness,
                "answer_contains": tr.answer.contains_score,
                "tool_calls": tr.tool_calls,
                "latency_s": tr.latency_s,
            }
            for tr in trs
        ]
        for agent, trs in all_tasks.items()
    }

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


async def main():
    parser = argparse.ArgumentParser(description="Cognee KG vs InMemory baseline evaluation")
    parser.add_argument("--scenario", type=str, default=None, help="Filter scenarios by name substring")
    parser.add_argument("--agent", choices=["cognee", "vector", "baseline"], default=None, help="Run only one agent")
    args = parser.parse_args()

    from rich.console import Console
    console = Console()

    model = os.getenv("LLM_MODEL", "llama3.1:8b")
    console.print(f"[bold]Cognee KG vs InMemory Baseline[/bold]")
    console.print(f"LLM: {model} | Scenarios: {len(SCENARIOS)} | Tasks: {get_task_count()}")

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if args.scenario.lower() in s.name.lower()]
        if not scenarios:
            console.print(f"[red]No scenario matching '{args.scenario}'[/red]")
            return

    agents = {}
    if args.agent in (None, "cognee"):
        agents["Cognee KG"] = CogneeAgent
    if args.agent in (None, "vector"):
        agents["Vector (ablation)"] = VectorAgent
    if args.agent in (None, "baseline"):
        agents["Baseline"] = BaselineAgent

    all_e2e: dict[str, list[E2EScore]] = {n: [] for n in agents}
    all_tasks: dict[str, list[TaskResult]] = {n: [] for n in agents}

    for agent_name, agent_cls in agents.items():
        console.print(f"\n{'─'*65}")
        console.print(f"[bold green]{agent_name}[/bold green]")
        for scenario in scenarios:
            agent = agent_cls(user_id=f"eval_{scenario.name}")
            try:
                e2e, trs = await evaluate_scenario(scenario, agent, agent_name, console)
                all_e2e[agent_name].append(e2e)
                all_tasks[agent_name].extend(trs)
            except Exception as e:
                console.print(f"  [red]ERROR: {e}[/red]")
                import traceback; traceback.print_exc()
            finally:
                await agent.reset()

    print_report(all_e2e, all_tasks)
    save_results(all_tasks)


if __name__ == "__main__":
    asyncio.run(main())
