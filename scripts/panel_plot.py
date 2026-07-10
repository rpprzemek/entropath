"""
Panel plot: Swiss Roll (3D) + 9 dimensionality reduction embeddings.
Saves panel_plot.png for use in a GitHub README.

Requirements:
    pip install numpy matplotlib scikit-learn umap-learn phate heatgeo dtne entropath pydiffmap
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import umap

# ── Specialised libraries (skip gracefully if missing) ─────────────────────
try:
    import phate
    HAS_PHATE = True
except ImportError:
    HAS_PHATE = False
    print("phate not found – PHATE panel will be blank")

try:
    from heatgeo.embedding import HeatGeo
    HAS_HEATGEO = True
except ImportError:
    HAS_HEATGEO = False
    print("heatgeo not found – HeatgEo panel will be blank")

try:
    from dtne import DTNE
    HAS_DTNE = True
except ImportError:
    HAS_DTNE = False
    print("dtne not found – DTNE panel will be blank")

try:
    from entropath import EntroPath
    HAS_ENTROPATH = True
except ImportError:
    HAS_ENTROPATH = False
    print("entropath not found – EntroPath panel will be blank")

try:
    from pydiffmap import diffusion_map as dm
    HAS_PYDIFFMAP = True
except ImportError:
    HAS_PYDIFFMAP = False
    print("pydiffmap not found – falling back to sklearn SpectralEmbedding for Diffusion Map")

# ── Data ───────────────────────────────────────────────────────────────────
np.random.seed(42)
N = 2000
X, color = make_swiss_roll(n_samples=N, noise=0.05, hole=False, random_state=42)

print("Computing embeddings …")

# PCA
Z_pca = PCA(n_components=2).fit_transform(X)

# t-SNE
Z_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

# UMAP
Z_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

# Isomap
Z_isomap = Isomap(n_components=2, n_neighbors=10).fit_transform(X)

# Diffusion Map
if HAS_PYDIFFMAP:
    dmap = dm.DiffusionMap.from_sklearn(n_evecs=2, epsilon="bgh", alpha=0.5, k=64)
    dmap.fit(X)
    Z_diffmap = dmap.dmap[:, :2]
else:
    from sklearn.manifold import SpectralEmbedding
    Z_diffmap = SpectralEmbedding(n_components=2, n_neighbors=10,
                                  random_state=42).fit_transform(X)

# PHATE
if HAS_PHATE:
    Z_phate = phate.PHATE(n_components=2, random_state=42, verbose=0).fit_transform(X)
else:
    Z_phate = None

# HeatgEo
if HAS_HEATGEO:
    Z_heatgeo = HeatGeo(knn=5).fit_transform(X)
else:
    Z_heatgeo = None

# DTNE
if HAS_DTNE:
    Z_dtne = DTNE(n_components=2).fit_transform(X)
else:
    Z_dtne = None

# EntroPath
if HAS_ENTROPATH:
    Z_entropath = EntroPath(n_components=2).fit_transform(X)
else:
    Z_entropath = None

print("Plotting …")

# ── Figure layout ──────────────────────────────────────────────────────────
CMAP = "viridis"
S = 4          # marker size
TITLE_FS = 13  # font size for subplot titles

fig = plt.figure(figsize=(22, 9))

# Slot 1: 3-D Swiss roll
ax3d = fig.add_subplot(2, 5, 1, projection="3d")
ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=CMAP, s=S, alpha=0.85)
ax3d.set_title("Swiss Roll", fontsize=TITLE_FS, fontweight="bold", pad=6)
ax3d.view_init(azim=-66, elev=12)
ax3d.set_axis_off()

# Slots 2-10: 2-D embeddings
panels = [
    ("PCA",          Z_pca),
    ("t-SNE",        Z_tsne),
    ("UMAP",         Z_umap),
    ("Isomap",       Z_isomap),
    ("Diffusion Map",Z_diffmap),
    ("PHATE",        Z_phate),
    ("HeatgEo",      Z_heatgeo),
    ("DTNE",         Z_dtne),
    ("EntroPath",    Z_entropath),
]

for slot, (title, Z) in enumerate(panels, start=2):
    ax = fig.add_subplot(2, 5, slot)
    if Z is not None:
        ax.scatter(Z[:, 0], Z[:, 1], c=color, cmap=CMAP, s=S)
    else:
        ax.text(0.5, 0.5, "library not installed",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=6)
    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")

#fig.suptitle("Dimensionality Reduction Comparison – Swiss Roll 2000 points",
#             fontsize=16, fontweight="bold")
plt.tight_layout()

out = Path(__file__).resolve().parent.parent / "figures" / "panel_plot.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
