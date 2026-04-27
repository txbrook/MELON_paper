#!/usr/bin/env python3
"""Regenerate simulation figures for the NeurIPS paper:

  - sim_method_comparison.pdf : main-content figure (all methods on BalAcc(Y))
  - sim_ablation.pdf          : appendix ablation bar chart (grouped colors)
  - sim_spatial_heatmaps.pdf  : appendix 2x2 spatial heatmaps (R, U, C, Y)

Numbers come straight from the tables already present in neurips_2024.tex
(tab:sim-phase2 and tab:sim-ablation), and the spatial heatmaps use the
simulation meta.csv ground-truth labels.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).resolve().parent

# ---- Method comparison data (from tab:sim-phase2) ----------------------------
# (display_name, balacc_mean, balacc_std, group)
METHOD_COMPARISON = [
    ("MELON full",      0.818, 0.045, "melon"),
    ("MELON no-space",  0.554, 0.018, "melon-ablation"),
    ("M1+M2 (raw)",     0.580, 0.058, "simple"),
    ("M1 only (raw)",   0.504, 0.006, "simple"),
    ("M2 only (raw)",   0.500, 0.000, "simple"),
    ("Space-only",      0.502, 0.020, "simple"),
    ("PCA concat",      0.502, 0.006, "simple"),
    ("MOFA+",           0.540, 0.022, "external"),
    ("MEFISTO",         0.540, 0.027, "external"),
    ("SpatialGLUE",     0.532, 0.027, "external"),
    ("MISO",            0.525, 0.048, "external"),
    ("Seurat WNN",      0.520, 0.051, "external"),
    ("totalVI",         0.504, 0.047, "external"),
]

GROUP_COLOR = {
    "melon":           "#B77F70",  # accent (matches main panels' MELON color)
    "melon-ablation":  "#D9B6AB",  # lighter accent
    "simple":          "#8A95A9",  # cool blue-grey
    "external":        "#7A8A71",  # green-grey
}

GROUP_LABEL = {
    "melon":          "MELON (full)",
    "melon-ablation": "MELON (no-space)",
    "simple":         "Simple baselines",
    "external":       "External multi-omics",
}

# ---- Ablation data (from tab:sim-ablation) -----------------------------------
ABLATION_DATA = [
    # (display_name, balacc, std, axis_group)
    ("MELON full",          0.818, 0.045, "full"),
    ("− augmentation",      0.711, 0.069, "aug"),
    ("− fusion",            0.547, 0.030, "fusion"),
    ("− spatial",           0.554, 0.018, "spatial"),
    ("− spatial − aug",     0.516, 0.030, "spatial"),
    ("M1-only",             0.532, 0.038, "modality"),
    ("M2-only",             0.509, 0.031, "modality"),
]

ABLATION_COLOR = {
    "full":     "#B77F70",
    "aug":      "#E5C2B5",
    "fusion":   "#7A8A71",
    "spatial":  "#8A95A9",
    "modality": "#BEB1A8",
}

ABLATION_GROUP_LABEL = {
    "full":     "Full model",
    "aug":      "− augmentation",
    "fusion":   "− fusion block",
    "spatial":  "− spatial bias",
    "modality": "Single modality",
}


# =============================================================================
# Figure 1: Main-content method comparison
# =============================================================================
def make_method_comparison():
    # Sort within group: MELON first (descending), then simple (descending), then external (descending).
    order = (
        sorted([m for m in METHOD_COMPARISON if m[3] == "melon"],          key=lambda r: -r[1])
        + sorted([m for m in METHOD_COMPARISON if m[3] == "melon-ablation"], key=lambda r: -r[1])
        + sorted([m for m in METHOD_COMPARISON if m[3] == "simple"],       key=lambda r: -r[1])
        + sorted([m for m in METHOD_COMPARISON if m[3] == "external"],     key=lambda r: -r[1])
    )
    names  = [r[0] for r in order]
    means  = [r[1] for r in order]
    stds   = [r[2] for r in order]
    colors = [GROUP_COLOR[r[3]] for r in order]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y_pos = np.arange(len(names))[::-1]  # MELON on top
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="none",
                   height=0.68, capsize=2.5,
                   error_kw=dict(elinewidth=0.8, ecolor="#444444"))

    # Chance line
    ax.axvline(0.5, linestyle="--", color="#888888", linewidth=0.8, zorder=0)
    ax.text(0.502, len(names) - 0.2, "chance", fontsize=7, color="#666666",
            ha="left", va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Balanced accuracy on synergy label $Y$", fontsize=10)
    ax.set_xlim(0.45, 0.92)
    ax.tick_params(axis="x", labelsize=9)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(axis="x", linestyle=":", color="#CCCCCC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_width() + s + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{m:.3f}",
            va="center", ha="left", fontsize=8, color="#333333",
        )

    # Bold MELON full
    yticklabels = ax.get_yticklabels()
    for lbl in yticklabels:
        if lbl.get_text() == "MELON full":
            lbl.set_fontweight("bold")

    # Legend (color → group)
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="None",
               markersize=9, markerfacecolor=GROUP_COLOR[g],
               markeredgecolor="none", label=GROUP_LABEL[g])
        for g in ["melon", "melon-ablation", "simple", "external"]
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              frameon=False, handlelength=1.0, handletextpad=0.5,
              labelspacing=0.4, borderpad=0.2)

    fig.tight_layout()
    out = OUT_DIR / "sim_method_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Figure 2: Appendix ablation
# =============================================================================
def make_ablation():
    names  = [r[0] for r in ABLATION_DATA]
    means  = [r[1] for r in ABLATION_DATA]
    stds   = [r[2] for r in ABLATION_DATA]
    colors = [ABLATION_COLOR[r[3]] for r in ABLATION_DATA]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y_pos = np.arange(len(names))[::-1]
    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="none",
                   height=0.68, capsize=2.5,
                   error_kw=dict(elinewidth=0.8, ecolor="#444444"))

    ax.axvline(0.5, linestyle="--", color="#888888", linewidth=0.8, zorder=0)
    ax.text(0.502, len(names) - 0.2, "chance", fontsize=7, color="#666666",
            ha="left", va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Balanced accuracy on synergy label $Y$", fontsize=10)
    ax.set_xlim(0.45, 0.92)
    ax.tick_params(axis="x", labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(axis="x", linestyle=":", color="#CCCCCC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_width() + s + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}",
                va="center", ha="left", fontsize=8, color="#333333")

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="None",
               markersize=9, markerfacecolor=ABLATION_COLOR[g],
               markeredgecolor="none", label=ABLATION_GROUP_LABEL[g])
        for g in ["full", "aug", "fusion", "spatial", "modality"]
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              frameon=False, handlelength=1.0, handletextpad=0.5,
              labelspacing=0.4, borderpad=0.2)

    fig.tight_layout()
    out = OUT_DIR / "sim_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# =============================================================================
# Figure 3: Appendix spatial heatmaps
# =============================================================================
def make_spatial_heatmaps():
    # Use the v2-tuned dataset that matches the paper's reported numbers
    # (600-dim features, 5/10/10 R/U/C levels). The smoke-test directory has
    # different parameters and produces visually different heatmaps.
    meta_path = Path("/Users/morita/Desktop/MELON/simulation/output/sim_v2_tuned_other_model_bundle/data/sim_v2_tuned_dataset/meta.csv")
    if not meta_path.exists():
        print(f"[warn] meta file not found, skipping spatial heatmaps: {meta_path}")
        return

    df = pd.read_csv(meta_path)
    x = df["x"].astype(float).to_numpy()
    y = df["y"].astype(float).to_numpy()

    panels = [
        ("Region $R$",       df["R"].to_numpy(), "tab10",  5),
        ("M1 attribute $U$", df["U"].to_numpy(), "tab20", 10),
        ("M2 attribute $C$", df["C"].to_numpy(), "tab20", 10),
        ("Synergy label $Y$", df["Y"].to_numpy(), None,    2),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.4),
                             gridspec_kw={"wspace": 0.18, "hspace": 0.28})
    axes_flat = axes.flatten()

    for ax, (title, vals, cmap_name, ncats) in zip(axes_flat, panels):
        if title.startswith("Synergy"):
            cmap = ListedColormap(["#DDDDDD", "#B77F70"])
            ax.scatter(x, y, c=vals, cmap=cmap, s=10, linewidths=0)
            handles = [
                Line2D([0], [0], marker="o", linestyle="None", markersize=6,
                       markerfacecolor="#DDDDDD", markeredgecolor="none", label="$Y=0$"),
                Line2D([0], [0], marker="o", linestyle="None", markersize=6,
                       markerfacecolor="#B77F70", markeredgecolor="none", label="$Y=1$"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=7,
                      frameon=False, handletextpad=0.3, borderpad=0.2)
        else:
            cmap = plt.get_cmap(cmap_name, ncats)
            sc = ax.scatter(x, y, c=vals, cmap=cmap, s=10, linewidths=0,
                            vmin=0.5, vmax=ncats + 0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_edgecolor("#BBBBBB")
            spine.set_linewidth(0.6)

    fig.tight_layout()
    out = OUT_DIR / "sim_spatial_heatmaps.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    make_method_comparison()
    make_ablation()
    make_spatial_heatmaps()


if __name__ == "__main__":
    main()
