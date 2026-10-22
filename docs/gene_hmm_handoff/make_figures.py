"""Figures for the gene-HMM handoff note (Dmel chrX+ / Athal chr5+).

Figures 0-2 need only data/ and the repo. Figures 3-4 also read the training work directory
(WORK_DIR: run_plateau.log and score arrays), the raw Dmel data (DMEL_RAW: dmel_reference.gtf,
dmel_genome.fa) and the plus-strand evaluation reference (DMEL_REF_PLUS_GFF3).
"""
import csv
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
S = OUT / "data"
REPO = OUT.parents[1]
W = Path(os.environ.get("WORK_DIR", "."))
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

ARMS = [("B: no HMM", "raw", "B (no HMM)", "#888888", "o"),
        ("B + post-hoc HMM", "hmm", "B + post-hoc HMM", "#d62728", "x"),
        ("A: HMM loss", "raw", "A raw heads", "#1f77b4", "s"),
        ("A: HMM loss", "hmm", "A HMM posterior", "#ff7f0e", "D"),
        ("C: HMM, all isoforms", "hmm", "C HMM posterior (all isoforms)", "#2ca02c", "^")]
SETTINGS = [("Dmel", "Scratch, 5 ep"), ("Dmel", "Human init, 5 ep"), ("Dmel", "Scratch, early stop"), ("Athal", "Scratch, 5 ep")]
rows = list(csv.DictReader(open(S / "compare.csv")))


def get(species, setting, arm, decode):
    for r in rows:
        if (r["species"], r["setting"], r["arm"], r["decode"]) == (species, setting, arm, decode):
            return r


# Fig 0: headline -- Bowen's CNN-Mamba vs + gene-HMM (Dmel chrX+, Athal chr5+)
wide = {r["key"]: r for r in csv.DictReader(
    open(REPO / "results/joint_frame_grid_v1/summary/all_metrics_wide.tsv"), delimiter="\t")}["b010"]
SITES4 = ("donor", "acceptor", "start", "stop")
bowen_held = np.mean([float(wide[f"heldout_{k}_AUPRC"]) for k in SITES4])
bowen_xtrain = np.mean([float(wide[f"chrxtrain_{k}_AUPRC"]) for k in SITES4])
b = get("Dmel", "Scratch, 5 ep", "B: no HMM", "raw")
a = get("Dmel", "Scratch, 5 ep", "A: HMM loss", "hmm")
ab = get("Athal", "Scratch, 5 ep", "B: no HMM", "raw")
aa = get("Athal", "Scratch, 5 ep", "A: HMM loss", "hmm")
xb = get("Dmel", "chrX in training, 5 ep", "B: no HMM", "raw")
xa = get("Dmel", "chrX in training, 5 ep", "A: HMM loss", "hmm")
xl = get("Dmel", "chrX in training, 30 ep", "A: HMM loss", "hmm")
bars = [("Bowen\nCNN-Mamba\n26 ep", bowen_held, None, "#bbbbbb", ""),
        ("CNN-Mamba\n(our run)\n5 ep", float(b["mean_ap"]), float(b["locus_f1"]), "#888888", ""),
        ("+ gene-\nHMM\n5 ep", float(a["mean_ap"]), float(a["locus_f1"]), "#ff7f0e", ""),
        ("Bowen\nCNN-Mamba\n26 ep", bowen_xtrain, float(wide["uniann_f1"]), "#bbbbbb", "//"),
        ("CNN-Mamba\n(our run)\n5 ep", float(xb["mean_ap"]), float(xb["locus_f1"]), "#888888", "//"),
        ("+ gene-\nHMM\n5 ep", float(xa["mean_ap"]), float(xa["locus_f1"]), "#ff7f0e", "//"),
        ("+ gene-\nHMM\n30 ep", float(xl["mean_ap"]), float(xl["locus_f1"]), "#c45a00", "//")]
fig = plt.figure(figsize=(9, 7.6))
gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.4], hspace=0.55, wspace=0.3)
axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
for metric, ax in ((1, axes[0]), (2, axes[1])):
    for i, bar in enumerate(bars):
        v = bar[metric]
        if v is None:
            ax.text(i, 62, "not run", ha="center", va="bottom", fontsize=7.5, color="#666666", rotation=90)
            continue
        ax.bar(i, v, color=bar[3], hatch=bar[4], edgecolor="#555555", linewidth=0.5)
        ax.text(i, v, f"{v:.4f}" if metric == 1 else f"{v + 1e-9:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(bars)), [x[0].replace("gene-\nHMM", "gene-HMM") if metric == 1 else x[0] for x in bars], fontsize=7.5)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.axvline(2.5, color="#999999", lw=0.8, ls="--")
    for x0, lab in ((1, "chrX held out"), (4.5, "chrX in training")):
        ax.text(x0, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5,
                fontweight="bold")
axes[0].set_ylim(0.85, 0.94)
axes[0].set_ylabel("Mean site AUPRC")
axes[0].set_title("(a) Dmel chrX+: site AUPRC (donor/acceptor/start/stop)", pad=18)
axes[1].set_ylim(60, 84)
axes[1].set_ylabel("UniAnn strict locus F1")
axes[1].set_title("(b) Dmel chrX+: UniAnn locus F1", pad=18)
for i, (lab, r) in enumerate((("CNN-Mamba", ab), ("+ gene-HMM", aa))):
    axes[2].bar(i, float(r["locus_f1"]), color=("#888888", "#ff7f0e")[i])
    axes[2].text(i, float(r["locus_f1"]), f"{float(r['locus_f1']):.1f}\nAP {float(r['mean_ap']):.3f}",
                 ha="center", va="bottom", fontsize=8)
axes[2].set_xticks([0, 1], ["CNN-\nMamba\n5 ep", "+ gene-\nHMM\n5 ep"], fontsize=7.5)
axes[2].set_xlim(-0.6, 1.6)
axes[2].set_ylim(60, 84)
axes[2].set_title("(c) Athal chr5+\n(held out)", pad=8)
for ax in axes:
    ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "fig0_headline.png", dpi=150, bbox_inches="tight", pad_inches=0.1)

# Fig 1: locus F1 per setting and arm
fig, ax = plt.subplots(figsize=(8.5, 3.2))
for j, (arm, dec, label, color, marker) in enumerate(ARMS):
    xs, ys = [], []
    for i, (sp, st) in enumerate(SETTINGS):
        r = get(sp, st, arm, dec)
        if r:
            xs.append(i + (j - 2) * 0.12)
            ys.append(float(r["locus_f1"]))
    ax.scatter(xs, ys, color=color, marker=marker, s=40, label=label, zorder=3)
ax.set_xticks(range(len(SETTINGS)), [f"{sp}\n{st}" for sp, st in SETTINGS])
ax.set_ylabel("UniAnn locus F1 (strict)")
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
ax.set_title("Annotation-level result: HMM-posterior decoding vs no HMM")
fig.tight_layout()
fig.savefig(OUT / "fig1_locus_f1.png", dpi=150, bbox_inches="tight", pad_inches=0.1)

# Fig 2: site AP vs locus F1, and locus Sn vs Pr
fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
for arm, dec, label, color, marker in ARMS:
    pts = [get(sp, st, arm, dec) for sp, st in SETTINGS]
    pts = [p for p in pts if p and p["locus_f1"]]
    axes[0].scatter([float(p["mean_ap"]) for p in pts], [float(p["locus_f1"]) for p in pts],
                    color=color, marker=marker, s=35, label=label)
    axes[1].scatter([float(p["locus_sn"]) for p in pts], [float(p["locus_pr"]) for p in pts],
                    color=color, marker=marker, s=35, label=label)
axes[0].set_xlabel("Mean site AUPRC (donor/acceptor/start/stop)")
axes[0].set_ylabel("UniAnn locus F1")
axes[0].set_title("(a) Site AP hides most of the gain")
axes[1].set_xlabel("Locus sensitivity")
axes[1].set_ylabel("Locus precision")
axes[1].set_title("(b) Gain = precision (fewer false loci)")
fig.legend(*axes[1].get_legend_handles_labels(), fontsize=7, frameon=False, loc="lower center", ncol=5)
for a in axes:
    a.grid(alpha=0.3)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "fig2_ap_vs_locus.png", dpi=150, bbox_inches="tight", pad_inches=0.1)

# Fig 3: LR schedule and validation AP (cosine 5 ep vs cosine 30 ep + ES vs plateau)
steps = list(csv.DictReader(open(S / "training_steps.csv")))
epochs = list(csv.DictReader(open(S / "epoch_metrics.csv")))
pl_lr, pl_ap, best = [], {}, None
for line in open(W / "run_plateau.log"):
    m = re.match(r"epoch=(\d+) batch=([\d,]+)/([\d,]+) .* lr=([\d.e+-]+)", line)
    if m:
        e, b, n = int(m[1]), int(m[2].replace(",", "")), int(m[3].replace(",", ""))
        pl_lr.append((e - 1 + b / n, float(m[4])))
    m = re.match(r"Epoch (\d+):.*selection hmm=([\d.]+)", line)
    if m:
        pl_ap[int(m[1])] = float(m[2])
    if line.startswith("Training finished") or "baseline_pl" in line:
        break
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
for run, label, color in (("hmm_w1", "cosine, 5 ep", "#1f77b4"), ("hmm_w1_es", "cosine, 30 ep + ES (patience 3)", "#d62728")):
    s = [r for r in steps if r["run"] == run]
    axes[0].plot([float(r["x"]) for r in s], [float(r["lr"]) for r in s], color=color, label=label)
    e = [r for r in epochs if r["run"] == run]
    axes[1].plot([int(r["epoch"]) for r in e], [float(r["ap_hmm"]) for r in e], "o-", ms=3, color=color, label=label)
    b = [r for r in e if r["best"] == "1"][0]
    axes[1].scatter([int(b["epoch"])], [float(b["ap_hmm"])], s=90, facecolors="none", edgecolors=color)
if pl_lr:
    x, y = zip(*pl_lr)
    axes[0].plot(x, y, color="#2ca02c", label="plateau (halve after 1 bad ep, floor 0.1x)")
    ks = sorted(pl_ap)
    axes[1].plot(ks, [pl_ap[k] for k in ks], "o-", ms=3, color="#2ca02c", label="plateau (A, held out)")
    kb = max(ks, key=pl_ap.get)
    axes[1].scatter([kb], [pl_ap[kb]], s=90, facecolors="none", edgecolors="#2ca02c")
xs = np.linspace(0, 30, 300)
axes[0].plot(xs, 3e-4 * (0.1 + 0.45 * (1 + np.cos(np.pi * xs / 30))), ":", color="#d62728", alpha=0.6,
             label="30-ep cosine if it had run to the end")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Learning rate")
axes[0].set_title("(a) ES stops a 30-ep cosine at ~0.6x LR")
axes[0].legend(fontsize=6.5, frameon=False)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val mean AP (HMM posterior, chr3L)")
axes[1].set_title("(b) Validation AP (circle = selected epoch)")
axes[1].set_ylim(0.82, 0.93)
axes[1].legend(fontsize=6.5, frameon=False, loc="lower right")
for a in axes:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig3_schedule.png", dpi=150, bbox_inches="tight", pad_inches=0.1)

# Fig 4: calibration (Dmel chrX+, scratch 5 ep model A)
sys.path.insert(0, str(REPO / "src"))
from cnn_mamba.evaluate_candidates import reference_sites  # noqa: E402
RAW = os.environ["DMEL_RAW"]
truth, _, _ = reference_sites("drosophila", f"{RAW}/dmel_reference.gtf", f"{RAW}/dmel_genome.fa", "NC_004354.4")
edges = np.array([1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001])
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
miss = {}
for name, d, color in (("raw heads", "scores/hmm_w1_raw", "#1f77b4"), ("HMM posterior", "scores_st/hmm_w1", "#ff7f0e")):
    ps, ys = [], []
    for site in ("donor", "acceptor", "start", "stop"):
        pos = np.load(W / d / f"{site}_positions0.npy")
        p = np.load(W / d / f"{site}_scores.npy").astype(float)
        y = np.isin(pos, truth[site])
        ps.append(p)
        ys.append(y)
        miss.setdefault(site, {})[name] = (p[y] < 1e-4).mean() * 100
    p, y = np.concatenate(ps), np.concatenate(ys)
    mx, my = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() > 50 and y[m].any():
            mx.append(p[m].mean())
            my.append(y[m].mean())
    axes[0].plot(mx, my, "o-", ms=3, color=color, label=name)
axes[0].plot([1e-7, 1], [1e-7, 1], "k:", lw=0.8)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Predicted site probability (bin mean)")
axes[0].set_ylabel("Observed fraction true (FlyBase)")
axes[0].set_title("(a) Site reliability, 4 site types pooled")
axes[0].legend(fontsize=7, frameon=False)
sites = list(miss)
xs = np.arange(len(sites))
for k, (name, color) in enumerate((("raw heads", "#1f77b4"), ("HMM posterior", "#ff7f0e"))):
    axes[1].bar(xs + (k - 0.5) * 0.38, [miss[s][name] for s in sites], 0.38, color=color, label=name)
axes[1].set_xticks(xs, sites)
axes[1].set_ylabel("% of true sites with p < 1e-4")
axes[1].set_title("(b) True sites the model vetoes")
axes[1].legend(fontsize=7, frameon=False)
ref = os.environ["DMEL_REF_PLUS_GFF3"]
jf = np.load(W / "scores_jf/hmm_w1/joint_frame_probabilities.npy", mmap_mode="r")
st = np.load(W / "scores_st/hmm_w1/state_posteriors.npy", mmap_mode="r")
cds = np.zeros(jf.shape[0], bool)
for line in open(ref):
    f = line.split("\t")
    if len(f) > 4 and f[2] == "CDS" and f[6] == "+":
        cds[int(f[3]) - 1:int(f[4])] = True
pedges = np.array([0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1.0001])
for name, p, color in (("raw joint-frame head", 1 - np.asarray(jf[:, 0], float), "#1f77b4"),
                       ("HMM posterior", np.asarray(st[:, 1:4], float).sum(1), "#ff7f0e")):
    mx, my = [], []
    for a, b in zip(pedges[:-1], pedges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() > 1000:
            mx.append(p[m].mean())
            my.append(cds[m].mean())
    ll = -np.log(np.clip(np.where(cds, p, 1 - p), 1e-10, 1)).mean()
    axes[2].plot(mx, my, "o-", ms=3, color=color, label=f"{name} (log loss {ll:.3f})")
axes[2].plot([0, 1], [0, 1], "k:", lw=0.8)
axes[2].set_xlabel("Predicted P(coding) (bin mean)")
axes[2].set_ylabel("Observed fraction reference CDS")
axes[2].set_title("(c) Per-base coding reliability")
axes[2].legend(fontsize=6.5, frameon=False, loc="upper left")
for a in axes:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig4_calibration.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
print({s: {k: round(v, 2) for k, v in d.items()} for s, d in miss.items()})
