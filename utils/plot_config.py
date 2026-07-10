"""
Plot configuration for the paper.
==================

"""

PANEL_SIZE = 2.4

POINT_KWARGS = dict(
    s=15, alpha=0.9, linewidths=0.4,
    edgecolors="white", rasterized=False,
)

INPUT_KWARGS = dict(
    s=4, alpha=0.8, linewidths=0, rasterized=False,
)

CMAP = "viridis"
TITLE_FONTSIZE = 11
TITLE_PAD = 6

PLOT_METHODS = [
    't-SNE', 'UMAP', 'MDS', 'Diffusion Maps',
    'PHATE', 'HeatGeo', 'DTNE', 'EntroPath',
]