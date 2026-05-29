import re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("F:/VQVAE/training_log.txt", "r", encoding="utf-8") as f:
    text = f.read()

pat = r"Iter\s+(\d+)\s*\|\s*ReconA=([\d.]+)\s+ReconB=([\d.]+)\s*\|\s*VelA=([\d.]+)\s+VelB=([\d.]+)\s*\|\s*Cyc=([\d.]+)/([\d.]+)\s*\|\s*Cont=([\d.]+)\s+UnpC=([\d.]+)\s*\|\s*FK=([\d.]+)\s+EE=([\d.]+)\s+Ct=([\d.]+)\s*\|\s*PPL=([\d.]+)/([\d.]+)"
data = {k: [] for k in ["iter","reconA","reconB","velA","velB","cycA","cycB","cont","unpC","fk_sup","fk_ee","contact","pplA","pplB"]}
for m in re.findall(pat, text):
    data["iter"].append(int(m[0]))
    for i, k in enumerate(["reconA","reconB","velA","velB","cycA","cycB","cont","unpC","fk_sup","fk_ee","contact","pplA","pplB"], 1):
        data[k].append(float(m[i]))
for k in data: data[k] = np.array(data[k])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("VQ-VAE Training Losses (50k-150k iter)", fontsize=14, fontweight="bold")
for ax, (ys, title, labels) in zip(axes.flat, [
    (["reconA","reconB"], "Reconstruction Loss", ["SMPL", "G1"]),
    (["velA","velB"], "Velocity Loss", ["SMPL", "G1"]),
    (["cycA","cycB"], "Cycle Consistency", ["A->B->A", "B->A->B"]),
    (["cont","unpC"], "Contrastive / CUT", ["Paired", "Unpaired"]),
    (["fk_sup","fk_ee","contact"], "FK Physics Losses", ["FK Sup", "FK EE", "Contact"]),
    (["pplA","pplB"], "Perplexity", ["SMPL", "G1"]),
]):
    for y, lbl in zip(ys, labels): ax.plot(data["iter"], data[y], label=lbl, alpha=0.8)
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    if title == "Perplexity": ax.axhline(512, color="red", ls="--", alpha=0.5, label="CB=512")
plt.tight_layout(); plt.savefig("F:/VQVAE/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 5))
for y, lbl in [("reconB","ReconB"), ("fk_sup","FK_sup")]:
    v = data[y]; ax2.plot(data["iter"], (v-v.min())/(v.max()-v.min()), label=lbl, alpha=0.8)
ax2.set_title("Convergence Speed: ReconB vs FK_sup (normalized)"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("F:/VQVAE/convergence_speed.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Parsed {len(data['iter'])} entries, iter {data['iter'][0]}-{data['iter'][-1]}")
for k in ["reconB","fk_sup","fk_ee","contact","pplA","pplB"]:
    print(f"  {k}: {data[k][0]:.4f} -> {data[k][-1]:.4f}")
print("Done: loss_curves.png, convergence_speed.png")
