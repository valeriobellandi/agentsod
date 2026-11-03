"""Benchmark (Sect. 6): monitor vs reference oracle on synthetic traces, and decision latency.

Usage: python bench/run.py [--quick]
Writes results/*.csv, results/summary.json and results/latency.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sodrm import Delegation, Event, Monitor, Principal, Provenance, Registry, parse  # noqa: E402
from sodrm.ledger import recompute, verify_chain  # noqa: E402
from sodrm.reference import ref_run  # noqa: E402

POLICY = parse((ROOT / "policies" / "energy_ops.sod").read_text())
DUAL = parse((ROOT / "policies" / "energy_ops.sod").read_text())
DUAL.constraints = [c for c in DUAL.constraints if c.require_n]
OPS_SP = ["propose", "approve", "execute"]
OPS_VP = ["create_vendor", "approve_inv", "release"]
ALL = [(o, "*") for o in OPS_SP + OPS_VP]
MODELS = [("M1", "X"), ("M2", "Y"), ("M3", "Z")]
OUT = ROOT / "results"


def make_world(rng, n_agents=8, n_humans=4, depth=1, p_bad_chain=0.02, dedicated_relays=False):
    reg = Registry()
    reg.add_principal(Principal("org:acme", "org"))
    humans = [f"h{i}" for i in range(n_humans)]
    agents = [f"a{i}" for i in range(n_agents)]
    for h in humans:
        reg.add_principal(Principal(h, "human"), ALL)
    for a in agents:
        m, o = rng.choice(MODELS)
        reg.add_principal(Principal(a, "agent", m, o))
    chains = {}
    did = 0
    for a in agents:
        if depth == 0:
            reg.own_scope[a] = frozenset(ALL)
            chains[a] = ()
            continue
        # chain of `depth` hops: human -> intermediates -> a
        if dedicated_relays:
            relays = [f"relay_{a}_{i}" for i in range(depth - 1)]
            for x in relays:
                reg.add_principal(Principal(x, "agent", *rng.choice(MODELS)))
        else:
            relays = rng.sample([x for x in agents if x != a], max(depth - 1, 0))
        hops = [rng.choice(humans)] + relays + [a]
        chain = []
        for g, r in zip(hops, hops[1:]):
            scope = frozenset(ALL)
            if rng.random() < p_bad_chain:
                scope = frozenset(ALL + [("admin", "*")])  # not attenuating -> invalid
            reg.add_delegation(Delegation(f"d{did}", g, r, scope, 0, 10 ** 9))
            chain.append(f"d{did}")
            did += 1
        chains[a] = tuple(chain)
    return reg, agents, chains


def make_events(rng, agents, chains, n_events, n_objects, ctx_size, p_unlabelled=0.01, dt=60):
    sources = {f"src{i}": frozenset() for i in range(20)}
    items = list(sources)
    events = []
    versions = {}  # object -> artifact versions proposed so far (latest last)
    for k in range(n_events):
        o = rng.randrange(n_objects)
        if rng.random() < 0.6:
            obj, op = f"SetpointChange(f|{o})", rng.choice(OPS_SP)
            attrs = (("criticality", rng.choice(["LOW", "HIGH"])),)
        else:
            obj, op, attrs = f"VendorPayment({o})", rng.choice(OPS_VP), ()
        ctx = set(rng.sample(items, min(ctx_size, len(items))))
        if rng.random() < p_unlabelled:
            ctx.add(f"ghost{k}")
        a = rng.choice(agents)
        ref = None
        if op == "propose":
            versions.setdefault(obj, []).append(f"art{k}")  # a new version of the proposal
        elif op in ("approve", "execute") and versions.get(obj):
            r = rng.random()
            vs = versions[obj]
            # mostly the latest version, sometimes a stale one, sometimes none at all
            ref = vs[-1] if r < 0.8 else (rng.choice(vs) if r < 0.95 else None)
        e = Event(f"e{k}", a, op, obj, k * dt, frozenset(ctx), chains[a], f"art{k}",
                  frozenset({f"note{k}"}), attrs, ref)
        events.append(e)
        items += [f"art{k}", f"note{k}"]
    return sources, events


def make_dual_events(rng, agents, chains, n_objects, max_rounds=14):
    """Workload focused on `require 2 separated approve before execute`: high-criticality
    changes with several approvals, revisions of the proposal and repeated executions."""
    sources = {f"src{i}": frozenset() for i in range(5)}
    events, k, t = [], 0, 0
    crit = (("criticality", "HIGH"),)
    for o in range(n_objects):
        obj, version, notes = f"SetpointChange(d|{o})", None, []
        for _ in range(rng.randint(5, max_rounds)):
            r = rng.random()
            a = rng.choice(agents)
            if version is None or r < 0.1:
                op, ctx, ref = "propose", set(), None
            elif r < 0.7:
                op, ref = "approve", (version if rng.random() < 0.9 else None)
                ctx = {version} | ({rng.choice(notes)} if notes and rng.random() < 0.3 else set())
            else:
                op, ctx, ref = "execute", set(), (version if rng.random() < 0.9 else f"stale{k}")
            e = Event(f"e{k}", a, op, obj, t, frozenset(ctx), chains[a], f"art{k}",
                      frozenset({f"note{k}"}), crit, ref)
            if op == "propose":
                version = f"art{k}"
            notes.append(f"note{k}")
            events.append(e)
            k += 1
            t += 60
    return sources, events


def dual_equivalence(n_traces, seed=17):
    rng = random.Random(seed)
    st = {"traces": 0, "events": 0, "mismatch": 0, "executes": 0, "executes_allowed": 0,
          "recompute_mismatch": 0}
    for _ in range(n_traces):
        reg, agents, chains = make_world(rng, n_agents=rng.randint(3, 6), n_humans=rng.randint(2, 4),
                                         depth=rng.randint(1, 2), p_bad_chain=0.0)
        sources, events = make_dual_events(rng, agents, chains, n_objects=rng.randint(2, 6))
        # only the two-approval constraint, so that approvals are not filtered by setpoint_4eyes
        prov = Provenance(horizon=max(c.window for c in POLICY.constraints))
        for s_, lab in sources.items():
            prov.add_source(s_, lab)
        m = Monitor(DUAL, reg, prov)
        c = DUAL.constraints[0]
        proposed = set()
        got = []
        for e in events:
            reason = None
            if e.op == "execute":
                if e.ref is None or e.ref not in proposed:
                    reason = "deny_unknown_version"
                else:
                    apps = [a for a in m.index[(c.name, e.obj)] if a.op == c.require_op
                            and a.ref == e.ref and e.t - a.t <= c.window]
                    ok_all, _ = m._find_independent(c, apps)
                    reason = "deny_approvals_already_used" if ok_all else "deny_not_enough_separated_approvals"
            d = m.decide(e).decision
            got.append(d)
            if e.op == "propose" and d == "allow":
                proposed.add(e.artifact)
            if e.op == "execute" and d != "allow":
                st[reason] = st.get(reason, 0) + 1
        want = ref_run(DUAL, reg, sources, events)
        st["traces"] += 1
        st["events"] += len(events)
        st["mismatch"] += sum(g != w for g, w in zip(got, want))
        st["recompute_mismatch"] += sum(recompute(r) != r["decision"] for r in m.ledger.records)
        for e, g in zip(events, got):
            if e.op == "execute":
                st["executes"] += 1
                st["executes_allowed"] += g == "allow"
    return st


def fresh_monitor(reg, sources, prune=True):
    prov = Provenance(horizon=max(c.window for c in POLICY.constraints) if prune else None)
    for s, lab in sources.items():
        prov.add_source(s, lab)
    return Monitor(POLICY, reg, prov)


def equivalence(n_traces, n_events, seed=7):
    rng = random.Random(seed)
    stats = {"traces": 0, "events": 0, "mismatch": 0, "recompute_mismatch": 0, "chain_ok": 0,
             "decisions": {}}
    for _ in range(n_traces):
        reg, agents, chains = make_world(rng, n_agents=rng.randint(3, 8), n_humans=rng.randint(2, 4),
                                         depth=rng.randint(0, 3))
        sources, events = make_events(rng, agents, chains, n_events, n_objects=rng.randint(1, 4),
                                      ctx_size=rng.randint(0, 6), dt=rng.choice([60, 3600]))
        m = fresh_monitor(reg, sources)
        got = [m.decide(e).decision for e in events]
        want = ref_run(POLICY, reg, sources, events)
        stats["traces"] += 1
        stats["events"] += len(events)
        stats["mismatch"] += sum(g != w for g, w in zip(got, want))
        stats["recompute_mismatch"] += sum(recompute(r) != r["decision"] for r in m.ledger.records)
        stats["chain_ok"] += verify_chain(m.ledger.records, m.ledger.anchor()) is None
        for g in got:
            stats["decisions"][g] = stats["decisions"].get(g, 0) + 1
        for rec in m.ledger.records:
            for w in rec["witnesses"]:
                if "require" in w:
                    stats["require_checks"] = stats.get("require_checks", 0) + 1
                    if not w["verdict"]:
                        stats["require_failed"] = stats.get("require_failed", 0) + 1
    return stats


def timed_run(reg, sources, events, prune=True):
    m = fresh_monitor(reg, sources, prune)
    ts = []
    for e in events:
        t0 = time.perf_counter()
        m.decide(e)
        ts.append((time.perf_counter() - t0) * 1e6)
    return ts


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def latency_vs_trace(n_max, seed=11, dt=600):
    """Long trace (dt=10 min, so the 30-day window is exceeded), with and without pruning."""
    rng = random.Random(seed)
    reg, agents, chains = make_world(rng, n_agents=30, n_humans=8, depth=2)
    sources, events = make_events(rng, agents, chains, n_max, n_objects=200, ctx_size=8, dt=dt)
    ts = timed_run(reg, sources, events, prune=True)
    tn = timed_run(reg, sources, events, prune=False)
    rows = []
    step = n_max // 10
    for b in range(10):
        a, n = ts[b * step:(b + 1) * step], tn[b * step:(b + 1) * step]
        rows.append({"trace_len": (b + 1) * step, "median_us": statistics.median(a),
                     "p95_us": pct(a, 0.95), "median_us_noprune": statistics.median(n),
                     "p95_us_noprune": pct(n, 0.95)})
    return rows


def latency_vs_param(name, values, n_events, seed=13):
    rows = []
    for v in values:
        rng = random.Random(seed)
        depth = v if name == "depth" else 2
        ctx = v if name == "context" else 8
        nobj = 1 if name == "hot" else max(n_events // 20, 1)
        n = v if name == "hot" else n_events
        reg, agents, chains = make_world(rng, n_agents=30, n_humans=8, depth=depth,
                                         p_bad_chain=0.0, dedicated_relays=(name == "depth"))
        sources, events = make_events(rng, agents, chains, n, n_objects=nobj, ctx_size=ctx, dt=1)
        ts = timed_run(reg, sources, events)
        tail = ts[len(ts) // 2:]
        rows.append({name: v, "median_us": statistics.median(tail), "p95_us": pct(tail, 0.95)})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=False)
    specs = [("trace", "trace_len", "trace length (events)"),
             ("context", "context", r"context size $|C_j|$"),
             ("depth", "depth", "delegation depth"),
             ("hot", "hot", r"events on one object ($k$)")]
    for ax, (key, col, label) in zip(axes, specs):
        rows = results[key]
        xs = [r[col] for r in rows]
        ax.plot(xs, [r["median_us"] for r in rows], marker="o", ms=2.5, lw=1, label="median")
        ax.plot(xs, [r["p95_us"] for r in rows], marker="s", ms=2.5, lw=1, ls="--", label="p95")
        if key == "trace":
            ax.plot(xs, [r["median_us_noprune"] for r in rows], marker="^", ms=2.5, lw=1, ls=":",
                    color="gray", label="median, no pruning")
        ax.set_xlabel(label, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel(r"decision time ($\mu$s)", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "latency.pdf")
    fig.savefig(OUT / "latency.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    q = args.quick
    t0 = time.time()
    eq = equivalence(n_traces=50 if q else 1000, n_events=40 if q else 60)
    eq["dual_workload"] = dual_equivalence(50 if q else 1000)
    res = {
        "trace": latency_vs_trace(2000 if q else 20000),
        "context": latency_vs_param("context", [1, 4, 16, 64, 256], 1000 if q else 4000),
        "depth": latency_vs_param("depth", [0, 1, 2, 4, 8, 16], 1000 if q else 4000),
        "hot": latency_vs_param("hot", [50, 100, 200, 400, 800] if q else [100, 200, 400, 800, 1600], 0),
    }
    for k, rows in res.items():
        write_csv(OUT / f"latency_{k}.csv", rows)
    plot(res)
    summary = {"equivalence": eq, "latency": res, "python": sys.version.split()[0],
               "wall_s": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"equivalence": eq, "wall_s": summary["wall_s"]}, indent=2))
    for k, rows in res.items():
        print(k, [tuple(round(v, 1) for v in r.values()) for r in rows])


if __name__ == "__main__":
    main()
