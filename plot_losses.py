import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Key data points extracted from training log
iters = [50000, 50200, 51000, 52000, 54000, 56000, 58000, 60000, 64000, 68000, 72000, 76000, 80000, 84000, 88000, 92000, 96000, 100000, 104000, 108000, 112000, 116000, 120000, 124000, 128000, 132000, 136000, 140000, 144000, 150000]
reconB = [0.0014, 0.1220, 0.1218, 0.1232, 0.1241, 0.1244, 0.1247, 0.1244, 0.1241, 0.1240, 0.1233, 0.1223, 0.1214, 0.1214, 0.1200, 0.1193, 0.1188, 0.1182, 0.1179, 0.1173, 0.1162, 0.1159, 0.1155, 0.1149, 0.1149, 0.1144, 0.1138, 0.1128, 0.1119, 0.1113]
fk_sup = [0.004, 0.696, 0.690, 0.687, 0.685, 0.684, 0.684, 0.683, 0.682, 0.683, 0.681, 0.682, 0.679, 0.680, 0.678, 0.677, 0.677, 0.676, 0.676, 0.675, 0.675, 0.674, 0.673, 0.675, 0.675, 0.673, 0.674, 0.672, 0.672, 0.672]
fk_ee = [0.011, 1.543, 1.521, 1.515, 1.511, 1.506, 1.505, 1.504, 1.498, 1.497, 1.492, 1.495, 1.488, 1.488, 1.484, 1.471, 1.471, 1.474, 1.472, 1.470, 1.473, 1.468, 1.463, 1.470, 1.472, 1.461, 1.468, 1.462, 1.459, 1.458]
contact = [0.0006, 0.190, 0.189, 0.189, 0.189, 0.189, 0.190, 0.190, 0.190, 0.189, 0.190, 0.190, 0.190, 0.190, 0.190, 0.190, 0.191, 0.190, 0.190, 0.191, 0.190, 0.190, 0.191, 0.190, 0.190, 0.190, 0.191, 0.190, 0.190, 0.190]
pplA = [2.2, 389, 384, 381, 380, 378, 379, 378, 378, 376, 376, 377, 377, 375, 377, 377, 378, 378, 377, 377, 377, 376, 375, 375, 375, 374, 374, 374, 376, 374]
pplB = [1.4, 269, 284, 288, 292, 292, 292, 296, 296, 297, 298, 300, 299, 298, 302, 301, 301, 304, 302, 301, 303, 302, 302, 303, 301, 302, 303, 302, 303, 302]
reconA = [0.0007, 0.060, 0.054, 0.053, 0.053, 0.053, 0.052, 0.051, 0.051, 0.050, 0.049, 0.048, 0.047, 0.047, 0.046, 0.045, 0.045, 0.045, 0.044, 0.044, 0.043, 0.043, 0.043, 0.043, 0.043, 0.042, 0.042, 0.042, 0.042, 0.041]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Cross-Embodiment VQ-VAE Training Losses (50k-150k)", fontsize=14, fontweight="bold")

plots = [
    (axes[0,0], "Reconstruction L1 Loss", [("ReconA (SMPL)", reconA), ("ReconB (G1)", reconB)]),
    (axes[0,1], "FK Physics Losses", [("FK Supervised", fk_sup), ("FK End-Effector", fk_ee), ("Contact", contact)]),
    (axes[0,2], "Perplexity (Codebook Usage)", [("PPL A (SMPL)", pplA), ("PPL B (G1)", pplB)]),
    (axes[1,0], "FK_sup Detail (log scale)", [("FK_sup", fk_sup)]),
    (axes[1,1], "FK_ee Detail (log scale)", [("FK_ee", fk_ee)]),
    (axes[1,2], "ReconB Detail", [("ReconB", reconB)]),
]

for ax, title, lines in plots:
    for label, vals in lines:
        ax.plot(iters, vals, marker=".", markersize=3, label=label, alpha=0.8)
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    if "log" in title: ax.set_yscale("log")

axes[0,2].axhline(512, color="red", ls="--", alpha=0.5, label="CB=512")

# Mark FK activation point
for ax in [axes[1,0], axes[1,1]]:
    ax.axvline(50200, color="orange", ls="--", alpha=0.6, label="FK activated")
    ax.legend()

plt.tight_layout()
plt.savefig("F:/VQVAE/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved loss_curves.png")
