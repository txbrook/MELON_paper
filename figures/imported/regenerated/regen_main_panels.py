#!/usr/bin/env python3
"""Regenerate clean Panel C (spatial method comparison) and Panel D (ARI bar)
for the three main-paper datasets (Mouse E15.5, Human Lymph, Mouse E13 Embryo).

Panel C: GT + MELON + 4 strongest baselines, fixed order, legend below, no suptitle.
Panel D: Horizontal ARI bar chart (full-width, single panel).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

OUT_DIR = Path(__file__).resolve().parent

# Fixed method order shown in Panel C (all 8 methods, consistent across datasets).
# GT is NOT included here - the manual annotation subfigure already shows it.
MAIN_METHODS = ["melon", "miso", "seurat", "spatialglue",
                "multigate", "totalvi", "mofaplus", "mefisto"]
# Panel D bar chart order (same as Panel C).
BAR_ORDER = MAIN_METHODS
# Metrics to show on Panel D. ARI is also shown on each Panel C subplot title,
# but we keep it here so the bar chart is self-contained.
PANEL_D_METRICS = ["ARI", "NMI", "Homogeneity", "V-Measure", "AMI"]

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

# Highlight MELON with a distinctive color; baselines share muted tones.
BAR_COLORS = {
    "melon": "#B77F70",
    "seurat": "#8A95A9",
    "spatialglue": "#7A8A71",
    "totalvi": "#9FABB9",
    "multigate": "#BEB1A8",
    "mofaplus": "#A79A89",
    "mefisto": "#B0B1B6",
    "miso": "#99857E",
}

PROJECT_ROOT = Path("/Users/morita/Library/CloudStorage/OneDrive-UTHealthHouston/MELON_master")
REPO_ROOT = PROJECT_ROOT / "NeurIPS_MELON"

DATASETS = [
    {
        "key": "mouse_E15_brain",
        "title": "Mouse E15.5 brain",
        "input_csv": str(PROJECT_ROOT / "results/mouse_E15_brain_compare_methods_with_melon.csv"),
        "truth_csv": str(PROJECT_ROOT / "Mouse_E15.5/barcode_truth_k12.csv"),
        "truth_barcode_col": "barcode",
        "truth_label_col": "region_truth",
        "he_image": str(REPO_ROOT / "dataset_info/pic/he/E15_5-HE.jpg"),
    },
    {
        "key": "human_lymph_node",
        "title": "Human lymph node",
        "input_csv": str(PROJECT_ROOT / "results/human_lymph_compare_methods_with_melon_raw_seed42.csv"),
        "truth_csv": str(PROJECT_ROOT / "human_lymph_node/truth.csv"),
        "truth_barcode_col": "Barcode",
        "truth_label_col": "manual-anno",
        "he_image": str(REPO_ROOT / "dataset_info/pic/he/Human_pymph.png"),
    },
    {
        "key": "mouse_embryo",
        "title": "Mouse E13 embryo",
        "input_csv": str(PROJECT_ROOT / "results/mouse_embryo_brain_compare_methods_with_melon.csv"),
        "truth_csv": str(PROJECT_ROOT / "Mouse_Embryo/embryo_truth.csv"),
        "truth_barcode_col": "barcode",
        "truth_label_col": "cluster",
        "he_image": str(REPO_ROOT / "dataset_info/pic/he/Mouse_e13_embryo.png"),
    },
]


def load_dataset(cfg: dict):
    df = pd.read_csv(cfg["input_csv"], index_col=0)
    truth_df = pd.read_csv(cfg["truth_csv"])
    truth_series = truth_df.set_index(cfg["truth_barcode_col"])[cfg["truth_label_col"]]
    common = df.index.intersection(truth_series.index)
    if len(common) == 0:
        raise ValueError(f"No barcode overlap for {cfg['key']}")
    df = df.loc[common]
    truth = truth_series.loc[common]

    method_cols = [c for c in df.columns if c not in ("x", "y", "ground_truth")]
    # Normalize method names: the MELON column may have a long suffix.
    renamed = {}
    for c in method_cols:
        lc = c.lower()
        if "melon" in lc:
            renamed[c] = "melon"
        else:
            renamed[c] = lc
    df = df.rename(columns=renamed)
    method_cols = [renamed[c] for c in method_cols]
    return df, truth, method_cols


def compute_metrics(truth: pd.Series, pred: pd.Series) -> dict:
    t = truth.to_numpy()
    p = pred.to_numpy()
    return {
        "ARI": float(adjusted_rand_score(t, p)),
        "NMI": float(normalized_mutual_info_score(t, p)),
        "Homogeneity": float(homogeneity_score(t, p)),
        "V-Measure": float(v_measure_score(t, p)),
        "AMI": float(adjusted_mutual_info_score(t, p)),
    }


def align_pred(truth: pd.Series, pred: pd.Series, truth_labels: np.ndarray, truth_to_idx: dict):
    pred_labels = np.array(sorted(pd.unique(pred)))
    p_to_i = {v: j for j, v in enumerate(pred_labels)}
    counts = np.zeros((len(truth_labels), len(pred_labels)), dtype=int)
    for t, p in zip(truth.to_numpy(), pred.to_numpy()):
        counts[truth_to_idx[t], p_to_i[p]] += 1
    cost = counts.max() - counts
    r_idx, c_idx = linear_sum_assignment(cost)
    pred_to_truth = {pred_labels[c]: truth_labels[r] for r, c in zip(r_idx, c_idx)}
    pred_aligned = pred.map(lambda v: pred_to_truth.get(v, v))
    aligned_idx = pred_aligned.map(lambda v: truth_to_idx.get(v, 0))
    metrics = compute_metrics(truth, pred)
    return aligned_idx.to_numpy(), metrics


def scale_unit(x: pd.Series, y: pd.Series):
    x = x.astype(float).to_numpy()
    y = y.astype(float).to_numpy()
    x_s = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    y_s = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    return x_s, y_s


def make_panel_c(cfg: dict):
    df, truth, method_cols = load_dataset(cfg)

    truth_labels = np.array(sorted(pd.unique(truth), key=lambda v: str(v)))
    truth_to_idx = {v: i for i, v in enumerate(truth_labels)}

    base = list(plt.get_cmap("Set3").colors)
    if len(truth_labels) > len(base):
        base += list(plt.get_cmap("tab20").colors)
    colors = base[: len(truth_labels)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(truth_labels) + 1) - 0.5, cmap.N)

    x_plot, y_plot = scale_unit(df["x"], df["y"])
    # For mouse embryo, y increases downward in the image; flip to match tissue orientation.
    if cfg["key"] == "mouse_embryo":
        y_plot = 1.0 - y_plot

    # Compute aligned predictions and metrics for every available method.
    per_method = {}
    for m in method_cols:
        aligned, metrics = align_pred(truth, df[m], truth_labels, truth_to_idx)
        per_method[m] = {"aligned": aligned, "metrics": metrics}

    # Layout: 2 rows x 5 cols. Col 0 = H&E (top) + Ground Truth (bottom).
    # Cols 1..4 = 8 method panels.
    methods_to_show = [m for m in MAIN_METHODS if m in per_method]
    n_rows, n_cols = 2, 5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.1 * n_cols, 2.2 * n_rows + 0.9),
        gridspec_kw={"wspace": 0.06, "hspace": 0.32},
    )

    dot_size = 5.5

    # --- Col 0, row 0: H&E image ---
    ax_he = axes[0, 0]
    try:
        he_img = mpimg.imread(cfg["he_image"])
        ax_he.imshow(he_img)
        ax_he.set_title("H&E", fontsize=10)
    except Exception as e:
        print(f"[warn] could not load H&E for {cfg['key']}: {e}")
        ax_he.text(0.5, 0.5, "H&E", ha="center", va="center")
        ax_he.set_title("H&E", fontsize=10)
    ax_he.set_xticks([]); ax_he.set_yticks([])
    ax_he.set_aspect("equal", adjustable="box")
    for spine in ax_he.spines.values():
        spine.set_edgecolor("#BBBBBB")
        spine.set_linewidth(0.6)

    # --- Col 0, row 1: Ground Truth scatter (same coords/colors as methods) ---
    ax_gt = axes[1, 0]
    truth_idx = truth.map(truth_to_idx).to_numpy()
    ax_gt.scatter(x_plot, y_plot, c=truth_idx,
                  cmap=cmap, norm=norm, s=dot_size, linewidths=0)
    ax_gt.set_title("Ground Truth", fontsize=10)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_gt.set_aspect("equal", adjustable="box")
    ax_gt.set_xlim(-0.02, 1.02); ax_gt.set_ylim(-0.02, 1.02)
    for spine in ax_gt.spines.values():
        spine.set_edgecolor("#BBBBBB")
        spine.set_linewidth(0.6)

    # --- Cols 1..4: eight method panels in fixed order ---
    for i, m in enumerate(methods_to_show):
        r, c = divmod(i, n_cols - 1)
        ax = axes[r, c + 1]
        info = per_method[m]
        ax.scatter(x_plot, y_plot, c=info["aligned"],
                   cmap=cmap, norm=norm, s=dot_size, linewidths=0)
        ari = info["metrics"]["ARI"]
        ax.set_title(f"{DISPLAY_NAME.get(m, m)}\nARI = {ari:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        for spine in ax.spines.values():
            spine.set_edgecolor("#BBBBBB")
            spine.set_linewidth(0.6)

    # Hide any unused method cells (there shouldn't be any with 8 methods).
    for j in range(len(methods_to_show), (n_cols - 1) * n_rows):
        r, c = divmod(j, n_cols - 1)
        axes[r, c + 1].axis("off")

    # Horizontal legend below the grid.
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markersize=6, markerfacecolor=cmap(norm(i)),
               markeredgecolor="none", label=str(lbl))
        for i, lbl in enumerate(truth_labels)
    ]
    ncol = min(len(handles), 6)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=ncol,
        fontsize=7,
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.9,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.14)
    out = OUT_DIR / f"{cfg['key']}_panelC.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    return {m: per_method[m]["metrics"] for m in per_method}


def make_panel_d(cfg: dict, all_metrics: dict):
    """Grouped bar chart: x = metric, groups = methods. ARI is NOT shown here
    because it is already displayed in every Panel C subplot title."""
    methods = [m for m in BAR_ORDER if m in all_metrics]
    metric_names = PANEL_D_METRICS

    n_methods = len(methods)
    n_metrics = len(metric_names)
    group_width = 0.82
    bar_w = group_width / n_methods
    x_centers = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(7.2, 2.3))

    for i, m in enumerate(methods):
        vals = [all_metrics[m][metric] for metric in metric_names]
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(
            x_centers + offset, vals,
            width=bar_w * 0.94,
            color=BAR_COLORS.get(m, "#888888"),
            edgecolor="none",
            label=DISPLAY_NAME.get(m, m),
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylabel("Score", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, None)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(axis="y", linestyle=":", color="#CCCCCC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(n_methods, 8),
        fontsize=8,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.1,
    )

    fig.tight_layout()
    out = OUT_DIR / f"{cfg['key']}_panelD.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    for cfg in DATASETS:
        print(f"\n=== {cfg['key']} ===")
        all_metrics = make_panel_c(cfg)
        make_panel_d(cfg, all_metrics)
        for m, mm in sorted(all_metrics.items(), key=lambda kv: -kv[1]["ARI"]):
            print(f"  {m:<14s} ARI={mm['ARI']:.3f}  NMI={mm['NMI']:.3f}  "
                  f"Homo={mm['Homogeneity']:.3f}  AMI={mm['AMI']:.3f}")


if __name__ == "__main__":
    main()
