"""Stage 2 dry run: GridOps-SoD with scripted, worst-case-obedient agents.

Usage: python bench/stage2.py [--quick]
Writes results/stage2/{tasks.csv, summary.json, attacks.csv, ablations.csv, a5.csv,
a5_pgf.dat, kappa.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gridops import run as R  # noqa: E402
from sodrm.ledger import recompute, verify_chain  # noqa: E402

OUT = ROOT / "results" / "stage2"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def mean_ci(xs):
    if len(xs) < 2:
        return (xs[0] if xs else 0.0, 0.0)
    m = statistics.mean(xs)
    return (m, 2.776 * statistics.stdev(xs) / math.sqrt(len(xs)))  # t(0.975, 4) for 5 runs


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate(rows_by_seed):
    rows = [r for rs in rows_by_seed for r in rs]
    ben = [r for r in rows if r["kind"] == "benign"]
    adv = [r for r in rows if r["kind"] != "benign"]
    per_seed = [R.summarize(rs) for rs in rows_by_seed]
    k_h = sum(r["harmful_done"] for r in adv)
    k_u = sum(r["completed"] for r in ben)
    k_f = sum(1 for r in ben if r["denies"] + r["escalations"] > 0)
    gm = [r["guard_ms"] for r in rows if r["guard_ms"] > 0]
    return {
        "harmful_per_run": mean_ci([s["harmful"] for s in per_seed]),
        "n_adv_per_run": per_seed[0]["n_adv"],
        "asr": (100 * k_h / len(adv), *wilson(k_h, len(adv))),
        "util": (100 * k_u / len(ben), *wilson(k_u, len(ben))),
        "fdr": (100 * k_f / len(ben), *wilson(k_f, len(ben))),
        "esc100": mean_ci([s["esc100"] for s in per_seed]),
        "esc100_all": mean_ci([s["esc100_all"] for s in per_seed]),
        "guard_ms_median": statistics.median(gm) if gm else 0.0,
    }


def ledger_check(px):
    mon = getattr(px.guard, "monitor", None)
    if mon is None:
        return None
    recs = mon.ledger.records
    bad = verify_chain(recs, mon.ledger.anchor())
    mism = sum(1 for r in recs if recompute(r) != r["decision"])
    return {"records": len(recs), "chain_ok": bad is None, "recompute_mismatch": mism}


def main(quick: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [1, 2] if quick else [1, 2, 3, 4, 5]
    nb, na = (30, 6) if quick else (100, 20)
    t0 = time.time()
    summary = {"seeds": seeds, "n_benign_per_family": nb, "n_attack_per_family": na,
               "agents": "scripted, p_obey=1 (worst case)", "p_chat": 0.2, "p_miss": 0.1,
               "main": {}, "attacks": {}, "ledger": {}}
    all_rows, attack_rows = [], []

    # ---- main comparison
    for cfg in R.CONFIGS:
        by_seed = []
        for s in seeds:
            px, sched = R.build(cfg, s)
            rows = _run_with(px, sched, s, nb, na)
            by_seed.append(rows)
            all_rows += rows
            lc = ledger_check(px)
            if lc:
                summary["ledger"].setdefault(cfg, []).append(lc)
        summary["main"][cfg] = aggregate(by_seed)
        # per attack: harmful and intercepted (>= 1 denial or escalation)
        rows = [r for rs in by_seed for r in rs if r["kind"] != "benign"]
        groups = defaultdict(list)
        for r in rows:
            groups[r["kind"] + r["variant"]].append(r)
        summary["attacks"][cfg] = {}
        for k, g in sorted(groups.items()):
            h = sum(r["harmful_done"] for r in g)
            i = sum(1 for r in g if r["denies"] + r["escalations"] > 0)
            summary["attacks"][cfg][k] = {"n": len(g), "harmful": h, "intercepted": i}
            attack_rows.append({"config": cfg, "attack": k, "n": len(g), "harmful": h,
                                "harmful_pct": round(100 * h / len(g), 1),
                                "intercepted_pct": round(100 * i / len(g), 1)})
        print(cfg, json.dumps({k: (round(v, 1) if isinstance(v, (int, float)) else [round(x, 1) for x in v])
                               for k, v in summary["main"][cfg].items()}), flush=True)
    write_csv(OUT / "tasks.csv", all_rows)
    write_csv(OUT / "attacks.csv", attack_rows)

    # ---- ablations on SoD-RM levels 2 and 3
    abl = []
    variants = [("session", "sod_aware", 0.1), ("noreset", "sod_aware", 0.1),
                ("plan", "sod_aware", 0.1), ("session", "naive", 0.1), ("session", "sod_aware", 0.0)]
    for cfg in ["L1", "L2", "L3"]:
        for gran, sched, pm in variants:
            by_seed = []
            for s in seeds:
                px, sch = R.build(cfg, s, granularity=gran, scheduler=sched, p_miss=pm)
                by_seed.append(_run_with(px, sch, s, nb, na))
            a = aggregate(by_seed)
            abl.append({"config": cfg, "granularity": gran, "scheduler": sched, "p_miss": pm,
                        "harmful_per_run": round(a["harmful_per_run"][0], 1),
                        "asr": round(a["asr"][0], 1), "util": round(a["util"][0], 1),
                        "fdr": round(a["fdr"][0], 1), "esc100": round(a["esc100"][0], 1)})
            print("abl", abl[-1], flush=True)
    write_csv(OUT / "ablations.csv", abl)

    # ---- A5: long-horizon pairing
    inst = range(8) if quick else range(20)
    curves, kap = {}, []
    for cfg, kappa in [("L2", 20), ("L3", 5), ("L3", 10), ("L3", 20), ("L3", 30)]:
        per_ep = defaultdict(list)
        esc = 0
        tot = 0
        for i in inst:
            wf = ["grid", "ptw", "trade", "p2p"][i % 4]
            rows = R.run_a5(cfg, 100 + i, wf=wf, kappa=kappa)
            for r in rows:
                per_ep[r["episode"]].append(r["harm_cum"])
            esc += sum(r["esc"] for r in rows)
            tot += rows[-1]["harm_cum"]
        name = f"{cfg}_k{kappa}" if cfg == "L3" else cfg
        curves[name] = [statistics.mean(per_ep[e]) for e in sorted(per_ep)]
        kap.append({"config": cfg, "kappa": kappa if cfg == "L3" else "", "instances": len(inst),
                    "harmful_total": tot, "harmful_per_instance": round(tot / len(inst), 2),
                    "escalations_per_episode": round(esc / (len(inst) * 60), 3)})
        print("A5", kap[-1], flush=True)
    write_csv(OUT / "kappa.csv", kap)
    with open(OUT / "a5_pgf.dat", "w") as f:
        f.write("episode " + " ".join(curves) + "\n")
        for e in range(60):
            f.write(f"{e + 1} " + " ".join(f"{curves[k][e]:.3f}" for k in curves) + "\n")

    summary["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("done in", summary["elapsed_s"], "s")


def _run_with(px, sched, seed, nb, na):
    import random
    from gridops.scenarios import PayloadFactory, Pipeline, ScriptedAgents, make_tasks
    rng = random.Random(seed)
    tasks = make_tasks(px, rng, PayloadFactory(seed), nb, na)
    ag = ScriptedAgents(px, seed + 1)
    pipe = Pipeline(px, ag, sched, seed + 2)
    out = []
    for t in tasks:
        r = pipe.run(t)
        r.update(config=px.guard.name, seed=seed)
        out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
