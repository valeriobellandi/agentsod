"""Re-plot Stage 1 latency results for single-column (LNCS, 12.2 cm) layout."""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "latency_lncs.pdf"
def load(k): return list(csv.DictReader(open(R / f"latency_{k}.csv")))
specs = [("trace", "trace_len", "trace length (events)"), ("context", "context", r"context size $|C_j|$"),
         ("depth", "depth", "delegation depth"), ("hot", "hot", r"events on one object ($k$)")]
fig, axes = plt.subplots(2, 2, figsize=(4.8, 3.0))
for ax, (k, col, label) in zip(axes.flat, specs):
    rows = load(k); xs = [float(r[col]) for r in rows]
    ax.plot(xs, [float(r["median_us"]) / 1000 for r in rows], marker="o", ms=2.5, lw=1, label="median")
    ax.plot(xs, [float(r["p95_us"]) / 1000 for r in rows], marker="s", ms=2.5, lw=1, ls="--", label="p95")
    if k == "trace":
        ax.plot(xs, [float(r["median_us_noprune"]) / 1000 for r in rows], marker="^", ms=2.5, lw=1,
                ls=":", color="gray", label="median, no pruning")
        ax.legend(fontsize=5.5, frameon=False)
    ax.set_xlabel(label, fontsize=7); ax.set_ylabel("ms / decision", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.8)
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
