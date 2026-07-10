"""Build the multi-dataset bio composite figure (paper Figure).

Loads per-dataset embedding artifacts produced by save_bio_artifacts() in
each bio notebook and lays them out as a rows=datasets x columns=methods
grid, matching the DTNE Fig. 3 layout convention.

Layout (DTNE-style):
  - One row per dataset, one column per method
  - Method titles only on the top row
  - Dataset name as a ROTATED label in the left figure margin (with n_cells)
  - Per-row color legend placed just right of the label, anchored to that
    row's vertical midpoint in figure coordinates
  - No per-panel borders, no axis ticks; points always rasterized
  - Saves PDF (vector text/axes, rasterized points) + PNG under figures/

Palette is per-dataset (PALETTE_MODE), default "spectral" (DTNE parity:
category code i -> Spectral(i/(n-1))); Paul15 uses "saved" (19 clusters are
too muddy in Spectral). Embryoid-body stages are ordinal, so Spectral is also
the natural choice there.

Usage (from project root):
    python scripts/build_paper_figure.py
    python scripts/build_paper_figure.py --palette saved        # readable categorical everywhere
    python scripts/build_paper_figure.py --datasets pancreas nestorowa
    python scripts/build_paper_figure.py --skip-methods HeatGeo PCA
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.bio_utils import BIO_PLOT_ORDER   # noqa: E402


# -- Defaults ----------------------------------------------------------------

DEFAULT_DATASETS = [
    ("paul15",         "Paul15"),
    ("nestorowa",      "Nestorowa"),
    ("pancreas",       "Pancreas"),
    ("lymphoid",       "Lymphoid"),
    ("embryoid_body",  "Embryoid Body"),
    ("root_atlas",     "Root Atlas"),
]

SKIP_METHODS = {"HeatGeo"}        # infeasible at root-atlas scale; omitted everywhere

# Per-dataset palette overrides. Default is DEFAULT_PALETTE; list only exceptions.
PALETTE_MODE = {
    "paul15": "saved",            # 19 clusters: scanpy categorical; Spectral too muddy
}
DEFAULT_PALETTE = "spectral"

# Display-time relabeling for artifacts that stored integer codes.
LABEL_REMAP = {
    "embryoid_body": {1: "00-03", 2: "06-09", 3: "12-15", 4: "18-21", 5: "24-27"},
}

POINT_SIZE = {
    "nestorowa": 2.0, "pancreas": 2.0, "lymphoid": 2.0,
    "embryoid_body": 1.0, "paul15": 2.0, "root_atlas": 0.6, "pbmc": 3.0,
}

PANEL_SIZE = 2.4
DPI = 400
LEGEND_COL_RATIO = 0.15           # narrow: legends are placed in figure coords, not here
FIG_LEFT = 0.1                   # where method panels start (room for label + legend)
LABEL_X = 0.018                   # figure-x of the rotated dataset name
LEGEND_X = 0.032                  # figure-x of the legend's left edge (just past label)


# -- Loading -----------------------------------------------------------------

def load_artifact(dataset_key, results_dir):
    path = results_dir / dataset_key / f"{dataset_key}_embeddings.pkl"
    if not path.exists():
        print(f"  [skip] {dataset_key}: no artifact at {path}")
        return None
    with open(path, "rb") as f:
        art = pickle.load(f)
    print(f"  [load] {dataset_key}: n={art['X_shape'][0]}, methods={len(art['embeddings'])}")
    return art


# -- Palette -----------------------------------------------------------------

def _remap(remap, v):
    """Look up v in remap, tolerating int/str/np-int key mismatches."""
    if remap is None:
        return v
    if v in remap:
        return remap[v]
    try:
        iv = int(v)
        if iv in remap:
            return remap[iv]
    except (ValueError, TypeError):
        pass
    return v


def build_palette(art, key, mode):
    """Return (zip_types, cell_colors), applying any LABEL_REMAP and palette mode."""
    remap = LABEL_REMAP.get(key)
    labels = [_remap(remap, l) for l in art["labels"]]
    #categories = [_remap(remap, c) for c in art["zip_types"].keys()]
    categories = list(dict.fromkeys(labels))      # unique, in first-appearance order
    saved = {_remap(remap, c): col for c, col in art["zip_types"].items()}

    if mode == "saved":
        zip_types = saved
    elif mode == "spectral":
        cmap = plt.get_cmap("Spectral")
        cats_sorted = sorted(categories, key=str)          # the ONE canonical order
        n = len(cats_sorted)
        zip_types = {c: matplotlib.colors.to_hex(cmap(i/(n-1) if n > 1 else 0.5))
                     for i, c in enumerate(cats_sorted)}
    else:
        raise ValueError(f"Unknown palette mode: {mode}")

    assert set(labels) <= set(zip_types), \
        f"label/zip_types key mismatch: {set(labels) - set(zip_types)}"
    assert all(l in zip_types for l in labels), \
        f"{key}: labels not in palette: {set(labels) - set(zip_types)}"
    
    cell_colors = [zip_types[l] for l in labels]
    assert all(l in zip_types for l in labels)

    return zip_types, cell_colors


def make_handles(zip_types):
    return [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
                       markeredgewidth=0, markersize=6, label=str(label))
            for label, color in zip_types.items()]


# -- Plotting primitives -----------------------------------------------------

def blank_axis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_method_panel(ax, embedding, cell_colors, s=2.0):
    if embedding is None:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                fontsize=11, color="0.5", transform=ax.transAxes)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=cell_colors, s=s,
                   alpha=0.85, linewidths=0, edgecolors="white", rasterized=True)
        #ax.set_aspect("equal") # , adjustable="box", datalim
    blank_axis(ax)


# -- Main figure -------------------------------------------------------------

def build_composite(artifacts, method_order, output_path, palette_default,
                    panel_size=PANEL_SIZE, dpi=DPI):
    n_rows = len(artifacts)
    n_methods = len(method_order)
    n_cols = n_methods + 1                       # col 0 is a (blank) legend spacer

    width_ratios = [LEGEND_COL_RATIO] + [1.0] * n_methods
    PANEL_ASPECT = 4 / 3          # method cell width : height (DTNE-like landscape panels)
    panel_h = panel_size / PANEL_ASPECT     # height per row = width / 1.333

    fig_w = panel_size * (LEGEND_COL_RATIO + n_methods) + 0.3
    fig_h = panel_h * n_rows + 0.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             gridspec_kw={"width_ratios": width_ratios}, squeeze=False)

    # Pass 1: draw panels; stash each row's palette for the legend pass
    row_zip_types = []
    for row, (key, display_name, art) in enumerate(artifacts):
        mode = PALETTE_MODE.get(key, palette_default)
        zip_types, cell_colors = build_palette(art, key, mode)
        row_zip_types.append(zip_types)

        blank_axis(axes[row, 0])                 # legend column is just a spacer
        s = POINT_SIZE.get(key, 2.0)
        for col_idx, method_name in enumerate(method_order):
            ax = axes[row, col_idx + 1]
            plot_method_panel(ax, art["embeddings"].get(method_name), cell_colors, s=s)
            if row == 0:
                ax.set_title(method_name, fontsize=12, pad=12)

    plt.tight_layout()
    fig.subplots_adjust(left=FIG_LEFT, top=0.95, wspace=0.12, hspace=0.10)

    # Pass 2: per-row rotated label + legend, anchored to each row's figure-y.
    # MUST run after layout, since get_position() reflects the final geometry.
    for row, (key, display_name, art) in enumerate(artifacts):
        pos = axes[row, 0].get_position()
        y_mid = (pos.y0 + pos.y1) / 2

        fig.text(LABEL_X, y_mid, f"{display_name}\n(n = {art['X_shape'][0]:,})",
                 rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")

        zip_types = row_zip_types[row]
        handles = make_handles(zip_types)
        n_labels = len(handles)
        max_name = max((len(str(l)) for l in zip_types), default=0)
        ncol = 1 if max_name > 15 else max(1, (n_labels + 9) // 10)
        fontsize = 8 if n_labels <= 10 else (5 if max_name > 20 else 6)

        fig.legend(
            handles=handles, loc="center left",
            bbox_to_anchor=(LEGEND_X, y_mid),    # per-row y -> each legend on its own row
            fontsize=fontsize, ncol=ncol, frameon=True, framealpha=1.0,
            facecolor="white", edgecolor="#9aa0a6", handletextpad=0.4,
            labelspacing=0.3, borderpad=0.5, columnspacing=0.8,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": it re-crops and would shift the figure-coord labels/legends.
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi)
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi)
    plt.close(fig)
    print(f"\nSaved {output_path.with_suffix('.pdf')}")
    print(f"Saved {output_path.with_suffix('.png')}")


# -- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        help=f"Dataset keys in row order (default: {[d[0] for d in DEFAULT_DATASETS]})")
    parser.add_argument("--output", default="figures/composite_bio",
                        help="Output path relative to project root (no extension).")
    parser.add_argument("--palette", choices=["saved", "spectral"], default=DEFAULT_PALETTE,
                        help="Default palette for datasets not pinned in PALETTE_MODE.")
    parser.add_argument("--skip-methods", nargs="*", default=sorted(SKIP_METHODS),
                        help=f"Methods to omit as columns (default: {sorted(SKIP_METHODS)}).")
    parser.add_argument("--panel-size", type=float, default=PANEL_SIZE)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    name_map = dict(DEFAULT_DATASETS)
    dataset_keys = args.datasets if args.datasets else [k for k, _ in DEFAULT_DATASETS]

    print("Loading per-dataset artifacts:")
    results_dir = PROJECT_ROOT / "results"
    artifacts = []
    for key in dataset_keys:
        display = name_map.get(key, key.replace("_", " ").title())
        art = load_artifact(key, results_dir)
        if art is not None:
            artifacts.append((key, display, art))

    if not artifacts:
        print("\nNo artifacts found. Run the bio notebooks first.")
        sys.exit(1)

    method_order = [m for m in BIO_PLOT_ORDER if m not in set(args.skip_methods)]
    print(f"\nColumns: {method_order}")
    print(f"Building composite ({len(artifacts)} rows x {len(method_order)} cols), "
          f"default palette={args.palette}, dpi={args.dpi}...")

    build_composite(
        artifacts, method_order, PROJECT_ROOT / args.output,
        palette_default=args.palette, panel_size=args.panel_size, dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
