#!/usr/bin/env python3
"""Generate the LaTeX body for tab:multimethod_metrics.

Reads MELON_master/test_score/multik_metrics_long_combined.csv (one row per
{method, k, dataset}), aggregates each metric across the available k values
per (dataset, method), and emits a table with mean +/- SE cells, bolding the
best mean per (dataset, metric). Intended to be pasted into
manuscript/neurips_2026.tex (and the arxiv copy) to replace the older
single-k tables tab:dataset2_metrics / tab:dataset3_metrics.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(
    "/Users/morita/Library/CloudStorage/OneDrive-UTHealthHouston/"
    "MELON_master/test_score/multik_metrics_long_combined.csv"
)

DISPLAY = {
    "melon": "MELON", "miso": "MISO", "seurat": "Seurat",
    "spatialglue": "SpatialGlue", "multigate": "MultiGate",
    "totalvi": "TotalVI", "mofaplus": "MOFA+", "mefisto": "MEFISTO",
}
METHOD_ORDER = ["melon", "miso", "seurat", "spatialglue",
                "multigate", "totalvi", "mofaplus", "mefisto"]
DATASET_ORDER = ["Mouse E15.5 Brain", "Human Lymph Node", "Mouse E13 Embryo"]
# (csv_col, header, lower_is_better, decimals).
METRICS = [
    ("ari",         "ARI",      False, 3),
    ("nmi",         "NMI",      False, 3),
    ("ami",         "AMI",      False, 3),
    ("mi",          "MI",       False, 3),
    ("homogeneity", "Homog.",   False, 3),
    ("v_measure",   "V-Meas.",  False, 3),
    ("spri",        "SpRI",     False, 3),
    ("spari",       "SpARI",    False, 3),
    ("fmi",         "FMI",      False, 3),
    ("sc",          "SC",       False, 3),
    ("jaccard",     "Jaccard",  False, 3),
    ("chi",         "CHI",      False, 0),
    ("purity",      "Purity",   False, 3),
    ("dbi",         r"DBI$\downarrow$", True, 3),
]


def fmt(mean: float, sem: float, bold: bool, decimals: int) -> str:
    body = f"{mean:.{decimals}f} \\pm {sem:.{decimals}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def main() -> None:
    d = pd.read_csv(CSV)
    agg = (d.groupby(["dataset", "method"])
             .agg({m: ["mean", "sem"] for m, _, _, _ in METRICS}))

    n_metrics = len(METRICS)
    col_spec = "ll" + "c" * n_metrics
    header_row = "Dataset & Method & " + " & ".join(h for _, h, _, _ in METRICS) + r" \\"

    out = [
        r"\begin{table}[!htbp]",
        r"\centering",
        (r"\caption{Clustering metrics across the three benchmarked datasets, "
         r"aggregated across cluster numbers $k$ (mean $\pm$ SE; "
         r"$k\in\{6,8,10,12,14\}$ for Mouse E15.5 Brain and Human Lymph Node, "
         r"$k\in\{10,12,14,16,18\}$ for Mouse E13 Embryo). Best mean per "
         r"(dataset, metric) in \textbf{bold}; DBI is lower-is-better. "
         r"MI: mutual information; SpRI/SpARI: spatial Rand / adjusted Rand "
         r"index; FMI: Fowlkes--Mallows; SC: silhouette; CHI: "
         r"Calinski--Harabasz; DBI: Davies--Bouldin.}"),
        r"\label{tab:multimethod_metrics}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header_row,
    ]

    n_methods = len(METHOD_ORDER)
    for ds in DATASET_ORDER:
        out.append(r"\midrule")
        # For each metric, find the best method (max for higher-is-better, min for DBI).
        best = {}
        for col, _, lower_better, _ in METRICS:
            means = {m: agg.loc[(ds, m), (col, "mean")] for m in METHOD_ORDER}
            best[col] = (min if lower_better else max)(means, key=means.get)
        for i, m in enumerate(METHOD_ORDER):
            cells = []
            for col, _, _, dec in METRICS:
                cells.append(fmt(agg.loc[(ds, m), (col, "mean")],
                                 agg.loc[(ds, m), (col, "sem")],
                                 best[col] == m, dec))
            ds_label = (f"\\multirow{{{n_methods}}}{{*}}{{\\textit{{{ds}}}}}"
                        if i == 0 else "")
            method_label = (f"\\textbf{{{DISPLAY[m]}}}"
                            if m == "melon" else DISPLAY[m])
            out.append(f"{ds_label} & {method_label} & "
                       + " & ".join(cells) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
