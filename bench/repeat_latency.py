"""Repeated latency measurements: the latency runs of run.py repeated R times on the same
traces, reporting for each configuration the median over repetitions of the per-run median
and p95, with their range (min-max).

  python bench/repeat_latency.py --reps 5
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import OUT, latency_vs_param, latency_vs_trace  # noqa: E402

SPECS = {
    "trace": lambda: latency_vs_trace(20000),
    "context": lambda: latency_vs_param("context", [1, 4, 16, 64, 256], 4000),
    "depth": lambda: latency_vs_param("depth", [0, 1, 2, 4, 8, 16], 4000),
    "hot": lambda: latency_vs_param("hot", [100, 200, 400, 800, 1600], 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    for name, fn in SPECS.items():
        runs = [fn() for _ in range(a.reps)]
        key = list(runs[0][0])[0]
        cols = [c for c in runs[0][0] if c != key]
        out = []
        for i, row in enumerate(runs[0]):
            r = {key: row[key], "reps": a.reps}
            for c in cols:
                vals = [run[i][c] for run in runs]
                r[c] = statistics.median(vals)
                r[c + "_min"], r[c + "_max"] = min(vals), max(vals)
            out.append(r)
        with open(OUT / f"latency_{name}_rep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
