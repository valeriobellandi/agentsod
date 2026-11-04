"""Fig. 2 of the paper: latency results for single-column (LNCS, 12.2 cm) layout."""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

def eu(x, pos=None):
    """European notation: decimal comma, thin-space thousands for 5+ digits."""
    t = "%g" % round(x, 6)
    if "e" in t: t = "%.0f" % x
    if "." in t: return t.replace(".", ",")
    neg, d = t.startswith("-"), t.lstrip("-")
    if len(d) > 4:
        d = "{:,}".format(int(d)).replace(",", "\u2009")
    return ("-" if neg else "") + d
EU = FuncFormatter(eu)

R = Path(__file__).resolve().parent.parent / "results"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PANELS = 4 if "--panels=4" in sys.argv else 2
out = Path(args[0]) if args else R / "latency_lncs.pdf"
def load(k): return list(csv.DictReader(open(R / f"latency_{k}.csv")))
specs = [("trace", "trace_len", "trace length (events)"), ("context", "context", r"context size $|C_j|$"),
         ("depth", "depth", "delegation depth"), ("hot", "hot", r"events on one object ($k$)")][:PANELS]
fig, axes = plt.subplots(*((2, 2) if PANELS == 4 else (1, 2)), figsize=(4.8, 3.1) if PANELS == 4 else (4.0, 1.42))
for idx, (ax, (k, col, label)) in enumerate(zip(axes.flat, specs)):
    rows = load(k); xs = [float(r[col]) for r in rows]
    ax.plot(xs, [float(r["median_us"]) / 1000 for r in rows], marker="o", ms=3.2, lw=1, label="median")
    ax.plot(xs, [float(r["p95_us"]) / 1000 for r in rows], marker="s", ms=3.2, lw=1, ls="--", label="p95")
    if k == "trace":
        ax.plot(xs, [float(r["median_us_noprune"]) / 1000 for r in rows], marker="^", ms=3.2, lw=1,
                ls=":", color="gray", label="median, no pruning")
        ax.set_ylim(0, 0.75); ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.0)
    ax.set_xlabel(f"({'abcd'[idx]}) " + label, fontsize=8.5); ax.set_ylabel("ms / decision", fontsize=8)
    ax.tick_params(labelsize=7.5); ax.grid(alpha=0.3, lw=0.5)
    ax.xaxis.set_major_formatter(EU); ax.yaxis.set_major_formatter(EU)
    if k == "trace": ax.set_xticks([5000, 10000, 15000, 20000])
fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.8)
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
