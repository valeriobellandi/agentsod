"""Run the GridOps-SoD experiment.

  python -m experiments.gridops.run --backend mock --n 5 --out results/gridops/mock.jsonl
  python -m experiments.gridops.run --backend openai --base-url http://localhost:11434/v1 \
      --model qwen2.5:7b --n 40 --workers 4 --out results/gridops/qwen2.5-7b.jsonl

Output: one JSON line per (episode, condition). The run is resumable: episodes
already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .harness import run_episode
from .llm import MockChat, OpenAIChat
from .scenarios import KINDS, make_episodes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["openai", "mock"], default="openai")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--api-key-env", default="LLM_API_KEY", help="environment variable holding the key")
    ap.add_argument("--n", type=int, default=40, help="episodes per kind")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    llm = MockChat() if a.backend == "mock" else OpenAIChat(
        a.base_url, a.model, os.environ.get(a.api_key_env, "none"), a.temperature)
    kinds = set(a.kinds.split(","))
    eps = [e for e in make_episodes(a.n, a.seed) if e.kind in kinds]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["eid"] for l in out.read_text().splitlines() if l.strip()}
    todo = [e for e in eps if e.eid not in done]
    print(f"{len(eps)} episodes, {len(done)} already done, {len(todo)} to run with {llm.model}", flush=True)
    lock, t0, n = threading.Lock(), time.time(), 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool, out.open("a") as f:
        futs = {pool.submit(run_episode, llm, e): e for e in todo}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                rows = fut.result()
            except Exception as exc:
                print(f"  {e.eid}: failed ({exc})", file=sys.stderr, flush=True)
                continue
            with lock:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)} episodes, {el / 60:.1f} min, ~{el / n * (len(todo) - n) / 60:.0f} min left",
                          flush=True)
    print(f"done: {getattr(llm, 'calls', 0)} LLM calls -> {out}")


if __name__ == "__main__":
    main()
