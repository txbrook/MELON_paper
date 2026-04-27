#!/usr/bin/env python3
"""Regenerate clean spatial method-comparison figures for the two
exploratory datasets that lack per-spot ground-truth labels:

  - Human hippocampus (RNA + ATAC, 2500 spots)
  - ccRCC Y7_T tumor   (RNA + methylation, 2926 spots)

Layout: 2x4 grid of method spatial maps (MELON + 7 baselines), no
ground-truth panel, no ARI titles. Each panel is colored by its own
cluster ids since no shared label space exists.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

OUT_DIR = Path(__file__).resolve().parent

# Display order matches main 3-dataset figures
METHOD_ORDER = ["melon", "miso", "seurat", "spatialglue",
                "multigate", "totalvi", "mofaplus", "mefisto"]

DISPLAY_NAME = {
    "melon": "MELON",
    "seurat": "Seurat",
    "spatialglue": "SpatialGlue",
    "mofaplus": "MOFA+",
    "mefisto": "MEFISTO",
    "totalvi": "TotalVI",
    "multigate": "MultiGate",
    "miso": "MISO",
}

DATASETS = [
    {
        "key": "human_hippocampus",
        "title": "Human hippocampus (RNA + ATAC)",
        "input_csv": "/Users/morita/Desktop/MELON/results/Human_hippocampus_compare_methods_with_melon.csv",
        "flip_y": False,
    },
    {
        "key": "ccRCC_Y7_T",
        "title": "ccRCC Y7_T tumor (RNA + methylation)",
        "input_csv": "/Users/morita/Desktop/MELON/results/ccRCC_Y7_T_compare_methods_with_melon.csv",
        "flip_y": False,
    },
]


def load_dataset(cfg: dict):
    df = pd.read_csv(cfg["input_csv"], index_col=0)
    method_cols = [c for c in df.columns if c not in ("x", "y", "ground_truth")]
    renamed = {}
    for c in method_cols:
        lc = c.lower()
        renamed[c] = "melon" if "melon" in lc else lc
    df = df.rename(columns=renamed)
    return df, list(renamed.values())


def scale_unit(x: pd.Series, y: pd.Series):
    x = x.astype(float).to_numpy()
    y = y.astype(float).to_numpy()
    x_s = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    y_s = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    return x_s, y_s


def make_panel(cfg: dict):
    df, available = load_dataset(cfg)
    methods = [m for m in METHOD_ORDER if m in available]

    x_plot, y_plot = scale_unit(df["x"], df["y"])
    if cfg["flip_y"]:
        y_plot = 1.0 - y_plot

    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.1 * n_cols, 2.2 * n_rows + 0.4),
        gridspec_kw={"wspace": 0.06, "hspace": 0.32},
    )
    axes = np.atleast_2d(axes)

    dot_size = 5.5

    for i, m in enumerate(methods):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        labels = df[m].astype(str).to_numpy()
        unique_labels = np.array(sorted(pd.unique(labels), key=lambda v: str(v)))
        idx = np.array([np.where(unique_labels == l)[0][0] for l in labels])
        n_lab = len(unique_labels)
        # Use a discrete tab20 cycle so methods with many clusters stay readable.
        base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Set3").colors)
        cmap = ListedColormap(base[:max(n_lab, 3)])
        ax.scatter(x_plot, y_plot, c=idx, cmap=cmap, s=dot_size, linewidths=0,
                   vmin=-0.5, vmax=n_lab - 0.5)
        ax.set_title(f"{DISPLAY_NAME.get(m, m)}\n($k={n_lab}$)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        for spine in ax.spines.values():
            spine.set_edgecolor("#BBBBBB")
            spine.set_linewidth(0.6)

    # Hide unused cells if fewer than 8 methods
    for j in range(len(methods), n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
    out = OUT_DIR / f"{cfg['key']}_exploratory_panel.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    for cfg in DATASETS:
        make_panel(cfg)


if __name__ == "__main__":
    main()
