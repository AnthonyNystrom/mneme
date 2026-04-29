"""Use case: agent memory.

LLM-driven agents need to remember prior decisions so they're consistent
on similar inputs. mneme provides 'task description embedding -> plan'
lookup, with confidence-gated staleness.

Run:
    python examples/use_cases/agent_memory.py
"""

from __future__ import annotations

import time

from _embedder import TokenBagEmbedder

from mneme import MemoryStore, SemanticCache


def run_agent_loop(task: str) -> str:
    """Pretend this is a multi-step LLM agent: planning, tool calls, verification.
    Costs about 1s of wall time."""
    time.sleep(1.0)
    return f"plan-for[{task[:40]}]: 1) understand 2) decompose 3) execute"


def execute_task(cache: SemanticCache, task: str, agent_id: str) -> tuple[str, str, float]:
    """Cache-aware agent execution. Returns (plan, layer, latency_ms)."""
    t0 = time.monotonic()
    namespace = f"agent:{agent_id}"
    hit = cache.get(task, namespace=namespace)
    if hit is not None and hit.confidence >= 0.7:        # confidence gate
        return hit.response, hit.layer, (time.monotonic() - t0) * 1000

    plan = run_agent_loop(task)
    cache.put(task, plan, namespace=namespace)
    return plan, "miss", (time.monotonic() - t0) * 1000


def main() -> None:
    with SemanticCache(store=MemoryStore(), embedder=TokenBagEmbedder(), similarity_threshold=0.5) as cache:
        # Two agents, different memory partitions.
        tasks_for_alice = [
            "Summarize the latest pull request",
            "Summarize the latest pull request",          # exact replay
            "Summarize the latest pull request now",      # paraphrase, shares words
            "Refactor the payment module",                # different task
        ]
        tasks_for_bob = [
            "Summarize the latest pull request",          # bob hasn't seen this; miss
            "Summarize the latest pull request briefly",  # bob's paraphrase, hits bob's cache
        ]

        print("=== Agent: alice ===")
        for task in tasks_for_alice:
            _plan, layer, ms = execute_task(cache, task, agent_id="alice")
            print(f"  {layer:8s}  {ms:7.1f} ms  task={task!r:50s}")

        print("\n=== Agent: bob ===")
        for task in tasks_for_bob:
            _plan, layer, ms = execute_task(cache, task, agent_id="bob")
            print(f"  {layer:8s}  {ms:7.1f} ms  task={task!r:50s}")

        print("\n=== Per-agent memory size ===")
        for ns in cache.list_namespaces():
            print(f"  {ns}: {cache.stats(namespace=ns).entries} cached plans")


if __name__ == "__main__":
    main()
