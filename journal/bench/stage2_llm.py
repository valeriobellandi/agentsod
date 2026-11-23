"""Stage 2 with LLM agents: hosted open-weight models or a local Ollama.

Hosted (one provider per model family; keys only in environment variables):
  export GROQ_API_KEY=... TOGETHER_API_KEY=... DEEPINFRA_API_KEY=...
  python bench/stage2_llm.py --providers config/providers.example.json --smoke
  python bench/stage2_llm.py --providers config/providers.example.json \
      --configs B0,B1,B2,L2,L3 --seeds 1,2,3 --workers 4

Local replication with the same models (e.g. a Mac with Ollama):
  ollama pull gpt-oss:20b; ollama pull qwen2.5:7b; ollama pull mistral-small3.2:24b
  python bench/stage2_llm.py --providers config/providers.example.json --local --workers 1

Endpoints can also be given one by one: --model M1=url,model[,ENV_VAR_WITH_KEY].

Each finished (config, seed) is appended to <out>/rows.jsonl, so an interrupted run
resumes where it stopped (smoke runs go to <out>/smoke); every model turn (model, provider, token usage, latency) goes
to <out>/turns.jsonl. --smoke runs a few tasks per configuration to check the setup.
"""
from __future__ import annotations

import argparse
import json
import os
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
from gridops.llm import Endpoint, load_providers  # noqa: E402


def endpoint(spec: str, provider: str = "") -> Endpoint:
    parts = spec.split(",")
    key = os.environ.get(parts[2], "") if len(parts) > 2 else "none"
    return Endpoint(parts[0], parts[1], api_key=key or "none", provider=provider or parts[0])


def make_endpoints(a):
    """Fresh endpoints per run, so token counters and connections are not shared."""
    if a.providers:
        eps = load_providers(a.providers, local=a.local, local_url=a.local_url)
        judge = eps.pop("judge", None)
    else:
        eps = {}
        for m in a.model or []:
            fam, spec = m.split("=", 1)
            eps[fam] = endpoint(spec)
        judge = endpoint(a.judge, "judge") if a.judge else None
    if set(eps) != {"M1", "M2", "M3"}:
        sys.exit("need endpoints for M1, M2 and M3 (--providers or --model)")
    return eps, judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", help="JSON file, see config/providers.example.json")
    ap.add_argument("--local", action="store_true", help="serve the same models from local Ollama")
    ap.add_argument("--local-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", action="append", help="M1=url,model[,KEY_ENV] (alternative to --providers)")
    ap.add_argument("--judge", help="url,model[,KEY_ENV] for B2 (with --model)")
    ap.add_argument("--configs", default="B0,B1,B2,L2,L3")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--n-benign", type=int, default=36, help="benign tasks per family")
    ap.add_argument("--n-attack", type=int, default=6, help="instances per attack and family")
    ap.add_argument("--workers", type=int, default=4, help="(config, seed) runs in parallel")
    ap.add_argument("--out", default=str(ROOT / "results" / "stage2_llm"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    make_endpoints(a)  # fail early on missing keys
    out = Path(a.out) / "smoke" if a.smoke else Path(a.out)
    if a.smoke:
        a.seeds = a.seeds.split(",")[0]
    out.mkdir(parents=True, exist_ok=True)
    rows_f, turns_f = out / "rows.jsonl", out / "turns.jsonl"
    done = set()
    if rows_f.exists() and not a.smoke:
        for line in rows_f.read_text().splitlines():
            r = json.loads(line)
            done.add((r["config"], r["seed"]))
    nb, na = (3, 1) if a.smoke else (a.n_benign, a.n_attack)
    jobs = [(c, int(s)) for c in a.configs.split(",") for s in a.seeds.split(",")
            if (c, int(s)) not in done]
    lock = threading.Lock()
    tf = open(turns_f, "a")

    def log(rec):
        with lock:
            tf.write(json.dumps(rec, default=str) + "\n")
            tf.flush()

    def job(cfg, s):
        eps, judge = make_endpoints(a)
        if cfg == "B2" and judge is None:
            return cfg, s, None, 0.0, {}
        t0 = time.time()
        decs = []
        rows = R.run_llm_config(cfg, s, eps, judge=judge, n_benign=nb, n_attack=na, log=log,
                                decisions=decs)
        with lock, open(rows_f, "a") as rf, open(out / "decisions.jsonl", "a") as df:
            for r in rows:
                rf.write(json.dumps(r) + "\n")
            for d in decs:
                df.write(json.dumps(d, default=str) + "\n")
        usage = {f: e.usage for f, e in list(eps.items()) + ([("judge", judge)] if judge else [])}
        return cfg, s, rows, time.time() - t0, usage

    print(f"{len(jobs)} runs to do ({len(done)} already done), {a.workers} in parallel", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(job, c, s) for c, s in jobs]
        for f in as_completed(futs):
            try:
                cfg, s, rows, dt, usage = f.result()
            except Exception as err:   # one failed run must not stop the others; rerun resumes it
                print(f"run failed, will be retried on the next launch: {err!r}", flush=True)
                continue
            if rows is None:
                print(f"skip {cfg} seed {s}: no judge endpoint")
                continue
            sm = R.summarize(rows)
            tok = sum(u["prompt_tokens"] + u["completion_tokens"] for u in usage.values())
            errs = sum(r.get("llm_errors", 0) for r in rows)
            print(f"{cfg} seed {s}: {len(rows)} tasks in {dt:.0f}s, {tok} tokens, "
                  f"harmful={sm['harmful']} util={sm['util']:.0f}% fdr={sm['fdr']:.0f}% "
                  f"model_errors={errs}", flush=True)
    tf.close()


if __name__ == "__main__":
    main()
