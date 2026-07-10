"""A5 ablation figure: metric vs k_NN, one line per method, two panels
(uniform | non-uniform swiss roll). Reads the level-2 sweep JSONs.

IMPORTANT: point --geodesic at the SAME ground truth your main swiss-roll
tables use. The paper's primary convention is the analytic arc-length geodesic,
so regenerate the sweep with GEODESIC_TYPE="analytic" and run this with
--geodesic analytic. The shortest_path ground truth is circular for
Isomap/Shortest Path (they score ~1.0 by construction) and should not be the
reported version.

Usage (from project root):
    python scripts/plot_knn_ablation.py --geodesic analytic --metric spearman_row
    python scripts/plot_knn_ablation.py --geodesic shortest_path --metric demap
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Diffusion methods only (the comparison set for this ablation). EntroPath last.
METHODS = ["Diffusion Maps", "PHATE", "HeatGeo", "DTNE", "EntroPath"]
KNNS = [5, 10, 15, 20]

# Two panels: (file-stem, display title). The stems match your saved JSONs.
PANELS = [
    ("swiss_roll_uniform", "Swiss roll (uniform)"),
    ("non_uniform_swiss_roll_nonuniform", "Swiss roll (non-uniform)"),
]

# Distinct line styles; EntroPath emphasized (thicker, on top).
STYLE = {
    "Diffusion Maps": dict(color="#9467bd", marker="o"),
    "PHATE":          dict(color="#2ca02c", marker="s"),
    "HeatGeo":        dict(color="#ff7f0e", marker="^"),
    "DTNE":           dict(color="#1f77b4", marker="D"),
    "EntroPath":      dict(color="#d62728", marker="*", linewidth=2.6, markersize=11, zorder=5),
}


def load_metric(results_dir, stem, geodesic, metric):
    """Return {method: (means[], stds[])} across KNNS for one dataset."""
    out = {m: ([], []) for m in METHODS}
    for k in KNNS:
        path = results_dir / f"{stem}_{geodesic}_level2_knn{k}.json"
        d = json.load(open(path))
        for m in METHODS:
            out[m][0].append(d[m][metric]["mean"])
            out[m][1].append(d[m][metric]["std"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geodesic", default="analytic",
                    choices=["analytic", "shortest_path"],
                    help="Ground-truth geodesic; use 'analytic' for the paper.")
    ap.add_argument("--metric", default="spearman_row",
                    help="level-2 metric key (spearman_row, pearson_row, demap, trust, ...)")
    ap.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "knn_ablation"))
    ap.add_argument("--output", default=str(PROJECT_ROOT / "figures" / "ablation_knn"))
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    label = {"spearman_row": "Spearman (row)", "pearson_row": "Pearson (row)",
             "demap": "DEMaP", "trust": "Trustworthiness"}.get(args.metric, args.metric)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(9.2, 3.8), sharey=True)
    for ax, (stem, title) in zip(axes, PANELS):
        data = load_metric(results_dir, stem, args.geodesic, args.metric)
        for m in METHODS:
            means, stds = data[m]
            st = {"linewidth": 1.6, "markersize": 6, **STYLE[m]}
            ax.plot(KNNS, means, label=m, **st)
            lo = [a - b for a, b in zip(means, stds)]
            hi = [a + b for a, b in zip(means, stds)]
            ax.fill_between(KNNS, lo, hi, color=STYLE[m]["color"], alpha=0.12, linewidth=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$k_{\mathrm{NN}}$")
        ax.set_xticks(KNNS)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.axvline(15, color="0.5", linestyle=":", linewidth=1, zorder=0)  # chosen value
    axes[0].set_ylabel(label)
    axes[-1].legend(fontsize=8, frameon=True, framealpha=0.95, loc="lower right")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.pdf')} and .png  [{args.geodesic}, {args.metric}]")


if __name__ == "__main__":
    main()
