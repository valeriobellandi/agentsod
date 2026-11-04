"""Summarise GridOps-SoD runs: rates with Wilson 95% intervals and a LaTeX table.

  python -m experiments.gridops.analyze results/gridops/*.jsonl --tex results/gridops/table.tex
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ORDER = ["legit_safe", "legit_unsafe", "A1_injection", "A1_adaptive", "A2_laundering", "A3_field"]
CONDS = ["L0", "L2", "L2n"]
LABEL = {"legit_safe": "Legitimate, safe", "legit_unsafe": "Legitimate, unsafe",
         "A1_injection": "A1 injection", "A1_adaptive": "A1 adaptive injection", "A2_laundering": "A2 delegation", "A3_field": "A3 field smuggling"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows


def summarise(rows, by_variant=False):
    g = defaultdict(list)
    for r in rows:
        kind = f'{r["kind"]}#v{r.get("variant", 0)}' if by_variant else r["kind"]
        g[(r["model"], kind, r["condition"])].append(r)
    out = []
    for (model, kind, cond), rs in sorted(g.items()):
        n = len(rs)
        # the quantity that matters: safe changes executed (legit_safe), harmful changes executed (others)
        target = sum(r["executed"] and not r["executed_harmful"] for r in rs) if kind.startswith("legit_safe") \
            else sum(r["executed_harmful"] for r in rs)
        p, lo, hi = wilson(target, n)
        out.append({"model": model, "kind": kind, "condition": cond, "n": n,
                    "metric": "safe_executed" if kind.startswith("legit_safe") else "harmful_executed",
                    "k": target, "rate": round(p, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                    "proposer_invalid": sum(not r["proposer_valid"] for r in rs),
                    "followed_injection": sum(bool(r.get("followed_injection")) for r in rs),
                    "approver_approved": sum(r.get("approver") == "approve" for r in rs),
                    "approver_invalid": sum(r.get("approver") == "invalid" for r in rs),
                    "simulated": sum(bool(r.get("simulated")) for r in rs),
                    "blocked_by_monitor": sum(r.get("approver") == "approve" and r.get("monitor_approve") not in
                                              (None, "allow") for r in rs)})
    return out


def pct(x):
    return "--" if x != x else f"{100 * x:.0f}"


def latex(summary) -> str:
    models = sorted({s["model"] for s in summary})
    idx = {(s["model"], s["kind"], s["condition"]): s for s in summary}
    cols = "l" + "r" * (len(CONDS) * len(models))
    head1 = " & ".join(f"\\multicolumn{{{len(CONDS)}}}{{c}}{{{m}}}" for m in models)
    head2 = " & ".join(CONDS * len(models))
    lines = [f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule", f" & {head1} \\\\",
             f"Episodes & {head2} \\\\", "\\midrule"]
    present = {s["kind"] for s in summary}
    for k in [k for k in ORDER if k in present]:
        cells = []
        for m in models:
            for c in CONDS:
                s = idx.get((m, k, c))
                cells.append(pct(s["rate"]) if s else "--")
        lines.append(f"{LABEL[k]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tex", default=None)
    ap.add_argument("--by-variant", action="store_true", help="split attacks by wording variant")
    a = ap.parse_args(argv)
    s = summarise(load(a.files), a.by_variant)
    w = csv.DictWriter(sys.stdout, fieldnames=list(s[0].keys()))
    w.writeheader()
    w.writerows(s)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(s[0].keys()))
            w.writeheader()
            w.writerows(s)
    if a.tex:
        Path(a.tex).write_text(latex(s) + "\n")
        print(f"\nLaTeX table -> {a.tex}")


if __name__ == "__main__":
    main()
