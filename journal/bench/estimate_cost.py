"""Estimate tokens, cost and time of the Stage 2 LLM runs before spending anything.

It runs the real pipeline with a stand-in model (gridops/fake_llm.py) that counts every
call and the size of every prompt, then applies the prices in a providers file.
Real models may take more turns than the stand-in: --turn-factor scales the number of
calls (default 1.5). After a smoke run with real models, --calibrate turns.jsonl
replaces the assumed tokens per turn and latency with measured ones.

  python bench/estimate_cost.py --providers config/providers.example.json
  python bench/estimate_cost.py --providers ... --calibrate results/stage2_llm/turns.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gridops import run as R  # noqa: E402
from gridops.fake_llm import FakeEndpoint  # noqa: E402

CHARS_PER_TOKEN = 3.6   # typical for English text and JSON with these tokenizers


class Counting(FakeEndpoint):
    def __init__(self, fam: str) -> None:
        super().__init__(fam)
        self.fam = fam
        self.n = 0
        self.prompt_tokens = 0

    def chat(self, messages, tools=None):
        self.n += 1
        self.prompt_tokens += int(len(json.dumps(messages)) + len(json.dumps(tools or []))) // CHARS_PER_TOKEN
        return super().chat(messages, tools)


def calibrate(path: str, cfg: dict) -> None:
    by = defaultdict(lambda: {"pt": [], "ct": [], "s": []})
    model_to_fam = {c["model"]: f for f, c in cfg.items() if not f.startswith("_") and f != "judge"}
    model_to_fam.update({c.get("local_ollama"): f for f, c in cfg.items()
                         if not f.startswith("_") and f != "judge"})
    for line in open(path):
        r = json.loads(line)
        fam = model_to_fam.get(r.get("model"))
        u = r.get("usage") or {}
        if fam and u.get("completion_tokens"):
            by[fam]["ct"].append(u["completion_tokens"])
            by[fam]["s"].append(r["s"])
    for fam, v in by.items():
        if v["ct"]:
            cfg[fam]["out_tokens_per_turn"] = statistics.mean(v["ct"])
            cfg[fam]["latency_s"] = statistics.median(v["s"])
            print(f"calibrated {fam}: {cfg[fam]['out_tokens_per_turn']:.0f} out tokens/turn, "
                  f"{cfg[fam]['latency_s']:.2f} s/turn over {len(v['ct'])} turns")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default=str(ROOT / "config" / "providers.example.json"))
    ap.add_argument("--configs", default="B0,B1,B2,L2,L3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-benign", type=int, default=36)
    ap.add_argument("--n-attack", type=int, default=6)
    ap.add_argument("--turn-factor", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=4, help="(config, seed) runs in parallel")
    ap.add_argument("--calibrate")
    ap.add_argument("--a5", action="store_true",
                    help="estimate the A5 experiment (bench/a5_llm.py) instead of the main runs")
    ap.add_argument("--episodes", type=int, default=30, help="A5 episodes per instance")
    a = ap.parse_args()
    cfg = json.loads(open(a.providers).read())
    if a.calibrate:
        calibrate(a.calibrate, cfg)

    # one seed per configuration with the stand-in model; scale by the number of seeds
    rows, n_tasks = [], 0
    for c in a.configs.split(","):
        eps = {f: Counting(f) for f in ("M1", "M2", "M3")}
        judge = Counting("judge")
        if a.a5:
            out = R.run_a5_llm(c, 1, eps, episodes=a.episodes)
            n_tasks += 2 * len(out) * a.seeds   # each episode: one background task and P's task
        else:
            out = R.run_llm_config(c, 1, eps, judge=judge, n_benign=a.n_benign, n_attack=a.n_attack)
            n_tasks += len(out) * a.seeds
        for fam, ep in list(eps.items()) + [("judge", judge)]:
            if ep.n == 0:
                continue
            p = cfg[fam]
            calls = ep.n * a.turn_factor * a.seeds if fam != "judge" else ep.n * a.seeds
            pin = ep.prompt_tokens * (a.turn_factor if fam != "judge" else 1) * a.seeds
            pout = calls * p["out_tokens_per_turn"]
            usd = pin / 1e6 * p["price_in"] + pout / 1e6 * p["price_out"]
            rows.append({"config": c, "fam": fam, "provider": p["provider"], "model": p["model"],
                         "calls": calls, "in_tok": pin, "out_tok": pout, "usd": usd,
                         "hours": calls * p["latency_s"] / 3600})
    what = (f"A5, {a.episodes} episodes per instance" if a.a5 else
            f"{a.n_benign} benign and {a.n_attack} x 5 attacks per family")
    print(f"\nTasks: {n_tasks} ({a.configs}; {a.seeds} seeds; {what}); turn factor {a.turn_factor}\n")
    print(f"{'config':6} {'family':6} {'provider':10} {'calls':>8} {'M tok in':>9} {'M tok out':>9} {'USD':>7}")
    for r in rows:
        print(f"{r['config']:6} {r['fam']:6} {r['provider']:10} {r['calls']:8.0f} "
              f"{r['in_tok'] / 1e6:9.2f} {r['out_tok'] / 1e6:9.2f} {r['usd']:7.2f}")
    tot = sum(r["usd"] for r in rows)
    hours = sum(r["hours"] for r in rows)
    print(f"\nTotal: {sum(r['in_tok'] for r in rows) / 1e6:.1f} M input tokens, "
          f"{sum(r['out_tok'] for r in rows) / 1e6:.1f} M output tokens, about {tot:.0f} USD")
    print(f"Sequential model time about {hours:.1f} h; with {a.workers} parallel runs about "
          f"{hours / a.workers:.1f} h (calls inside one run are sequential)")
    by_p = defaultdict(float)
    for r in rows:
        by_p[r["provider"]] += r["usd"]
    print("By provider: " + ", ".join(f"{k} {v:.2f} USD" for k, v in by_p.items()))


if __name__ == "__main__":
    main()
