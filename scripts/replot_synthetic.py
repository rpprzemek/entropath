import argparse, pickle
from pathlib import Path
import matplotlib.pyplot as plt
from utils.plot_config import PLOT_METHODS, POINT_KWARGS, PANEL_SIZE

def replot(dataset, project_root, panels=("input_3d", "row"),
           point_size=15, show_runtimes=False):
    results_dir = project_root / "results" / dataset
    figures_dir = project_root / "figures" / dataset
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{dataset}_artifacts.pkl", "rb") as f:
        art = pickle.load(f)

    if "input_3d" in panels:
        # 3D scatter of the ground-truth manifold
        fig = plt.figure(figsize=(PANEL_SIZE, PANEL_SIZE))
        ax = fig.add_subplot(111, projection="3d")
        X_3d = art["X_3d"]
        ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                   c=art["labels"], cmap="Spectral", s=point_size)
        ax.set_axis_off()
        fig.savefig(figures_dir / f"{dataset}_input_3d.pdf",
                    bbox_inches="tight")

    if "row" in panels:
        n_panels = len(PLOT_METHODS)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(PANEL_SIZE * n_panels, PANEL_SIZE + 0.8),
        )
        kwargs = {**POINT_KWARGS, "s": point_size}
        for ax, name in zip(axes, PLOT_METHODS):
            if name in art["embeddings"]:
                emb = art["embeddings"][name]
                ax.scatter(emb[:, 0], emb[:, 1],
                           c=art["labels"], cmap="Spectral", **kwargs)
            title = name
            if show_runtimes and art["runtimes"].get(name):
                title += f"\n({art['runtimes'][name]:.1f}s)"
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
        fig.savefig(figures_dir / f"{dataset}_row.pdf", bbox_inches="tight")
        fig.savefig(figures_dir / f"{dataset}_row.png",
                    dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--panels", nargs="+",
                   default=["input_3d", "row"],
                   choices=["input_3d", "row"])
    p.add_argument("--point-size", type=float, default=15)
    p.add_argument("--show-runtimes", action="store_true")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    replot(args.dataset, project_root,
           panels=tuple(args.panels),
           point_size=args.point_size,
           show_runtimes=args.show_runtimes)