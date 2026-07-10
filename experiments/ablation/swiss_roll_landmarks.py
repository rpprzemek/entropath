"""
Swiss Roll (50 000 points) + EntroPath embedding with FPS landmarks.
Saves swiss_roll_fps.png for the GitHub README.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from entropath import EntroPath

# ── Data ───────────────────────────────────────────────────────────────────
np.random.seed(42)
X, color = make_swiss_roll(n_samples=50_000, noise=0.05, random_state=42)

# ── Embedding ──────────────────────────────────────────────────────────────
model = EntroPath(
    k_neighbors=15,
    t_power="auto",
    landmarks_method="fps",
    use_landmarks=True,
    kernel="gaussian",
    smooth_entropy=False,
    mds_solver="smacof",
    random_state=42,
    plot=True,
    verbose=1,
)
Y = model.fit_transform(X)

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))

# Left: 3-D Swiss roll
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=0.2, alpha=0.6,
             cmap=plt.cm.Spectral)
ax3d.set_title("Swiss Roll  (50 000 pts, noise=0.05)", fontsize=13, fontweight="bold")
ax3d.view_init(azim=-70, elev=20)
ax3d.set_axis_off()

# Right: EntroPath embedding
ax2d = fig.add_subplot(1, 2, 2)
ax2d.scatter(Y[:, 0], Y[:, 1], c=color, s=0.2, alpha=0.6, cmap=plt.cm.Spectral)
ax2d.set_title("EntroPath  (FPS landmarks)", fontsize=13, fontweight="bold")
ax2d.set_axis_off()

plt.tight_layout()
fig.savefig("swiss_roll_fps.png", dpi=200, bbox_inches="tight")
print("Saved → swiss_roll_fps.png")
plt.show()
