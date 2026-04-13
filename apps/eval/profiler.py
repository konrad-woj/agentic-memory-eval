"""
profiler.py — Latency comparison between memory backends.

Usage: python profiler.py
"""

import asyncio
import time

from dotenv import load_dotenv
load_dotenv()


async def main():
    from rich.console import Console
    from rich.table import Table
    from agents import BaselineAgent, CogneeAgent

    console = Console()

    facts = [
        "Elena works as a data engineer at Acme Corp.",
        "Acme Corp signed a $1.2M contract with HealthFirst in January 2025.",
        "HealthFirst is a healthcare company in Boston.",
        "Our team has 15 engineers as of March 2025.",
    ]
    queries = ["Where does Elena work?", "What is the HealthFirst contract value?", "How many engineers?"]

    console.print("[bold]Latency Profile[/bold]\n")
    rows = []

    for name, cls in [("Baseline", BaselineAgent), ("Cognee KG", CogneeAgent)]:
        console.print(f"  Profiling {name}...")
        agent = cls(user_id="profiler")
        try:
            t0 = time.time()
            await agent.store_facts(facts)
            store = time.time() - t0

            ret_lats = []
            for q in queries:
                t0 = time.time()
                await agent.retrieve(q)
                ret_lats.append(time.time() - t0)

            ans_lats = []
            for q in queries:
                t0 = time.time()
                await agent.execute_task(q)
                ans_lats.append(time.time() - t0)

            rows.append((name, store, sum(ret_lats)/len(ret_lats), sum(ans_lats)/len(ans_lats)))
        except Exception as e:
            console.print(f"    [red]{e}[/red]")
        finally:
            await agent.reset()

    if len(rows) < 2:
        return

    t = Table(title="Latency (seconds)")
    t.add_column("Operation", style="bold")
    for r in rows:
        t.add_column(r[0], justify="center")

    t.add_row("Store facts", *[f"{r[1]:.2f}" for r in rows])
    t.add_row("Avg retrieve", *[f"{r[2]:.3f}" for r in rows])
    t.add_row("Avg execute_task", *[f"{r[3]:.2f}" for r in rows])

    if rows[0][1] > 0:
        t.add_row("Store overhead", "1.0x", f"{rows[1][1]/rows[0][1]:.0f}x")
    console.print(t)


if __name__ == "__main__":
    asyncio.run(main())
