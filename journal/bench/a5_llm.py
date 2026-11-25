"""A5 (long-horizon pairing) with LLM agents: level 2 (fixed pair) against level 3 (rotation).

Local (Ollama, same models as the main runs):
  python bench/a5_llm.py --providers config/providers.example.json --local --seeds 1,2,3,4
Hosted:
  python bench/a5_llm.py --providers config/providers.large.json --seeds 1,2,3,4 --workers 4

Each (config, seed) is one instance of `--episodes` episodes. Finished instances are appended
to <out>/a5_rows.jsonl (the run resumes where it stopped); model turns go to turns.jsonl and
monitor decisions to decisions.jsonl. --smoke runs one short instance per configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gridops import run as R  # noqa: E402
from gridops.llm import load_providers  # noqa: E402


def summarize(rows):
    pr = [r for r in rows if r["pressure"]]
    byc = [r for r in rows if r["checker"] and r["checker"].endswith("chk1")]
    return {"episodes": len(rows), "pressure": len(pr),
            "proposer_chose_harmful": sum(r["chose_harmful"] for r in pr),
            "harmful_executed": sum(r["harmful_done"] for r in rows),
            "C_reviews": len(byc),
            "C_skipped_check": sum(1 for r in byc if not r["used_check_tool"]),
            "rotated": sum(1 for r in rows if r["checker"] and not r["checker"].endswith("chk1"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", required=True)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--local-url", default="http://localhost:11434/v1")
    ap.add_argument("--configs", default="L2,L3")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--background", type=int, default=1)
    ap.add_argument("--p-pressure", type=float, default=0.5)
    ap.add_argument("--kappa", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nudge", type=int, default=0,
                    help="reminders when an agent ends its step without calling the tool that ends it")
    ap.add_argument("--out", default=str(ROOT / "results" / "a5_llm"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    load_providers(a.providers, local=a.local, local_url=a.local_url)  # fail early on missing keys
    out = Path(a.out) / "smoke" if a.smoke else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "a5_rows.jsonl"
    done = set()
    if rows_f.exists() and not a.smoke:
        for line in rows_f.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["seed"]))
    episodes = 4 if a.smoke else a.episodes
    seeds = a.seeds.split(",")[:1] if a.smoke else a.seeds.split(",")
    jobs = [(c, int(s)) for c in a.configs.split(",") for s in seeds if (c, int(s)) not in done]
    lock = threading.Lock()
    tf = open(out / "turns.jsonl", "a")

    def log(rec):
        with lock:
            tf.write(json.dumps(rec, default=str) + "\n")
            tf.flush()

    def job(cfg, s):
        eps = load_providers(a.providers, local=a.local, local_url=a.local_url)
        eps.pop("judge", None)
        decs = []
        t0 = time.time()
        rows = R.run_a5_llm(cfg, s, eps, episodes=episodes, background=a.background,
                            p_pressure=a.p_pressure, kappa=a.kappa, log=log, decisions=decs,
                            nudge=a.nudge)
        with lock, open(rows_f, "a") as rf, open(out / "decisions.jsonl", "a") as df:
            for r in rows:
                rf.write(json.dumps(r) + "\n")
            for d in decs:
                df.write(json.dumps(d, default=str) + "\n")
        return cfg, s, rows, time.time() - t0

    print(f"{len(jobs)} instances to do ({len(done)} already done), {a.workers} in parallel", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(job, c, s) for c, s in jobs]
        for f in as_completed(futs):
            try:
                cfg, s, rows, dt = f.result()
            except Exception as err:  # one failed instance must not stop the others
                print(f"instance failed, will be retried on the next launch: {err!r}", flush=True)
                continue
            print(f"{cfg} seed {s}: {dt:.0f}s {json.dumps(summarize(rows))}", flush=True)
    tf.close()


if __name__ == "__main__":
    main()
