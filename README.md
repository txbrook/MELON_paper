# NeurIPS_MELON

This folder has been cleaned so the active paper source is easy to find.

## Active files

- `neurips_2024.tex`
- `neurips_2024.sty`
- `ref.bib`
- `dataset_info/` with copied dataset descriptions, references, and annotation/tissue images
- `figures/` with only the six figures referenced by the paper
- `figures/imported/compare_methods/` with copied comparison plots from `/Users/morita/Desktop/MELON/plots/plot`
- `figures/imported/test_score/` with copied benchmarking and learning-curve plots from `/Users/morita/Desktop/MELON/test_score`
- `output/neurips_2024.pdf` as the latest compiled PDF

## Archived files

- `archive/root_build/` contains old top-level LaTeX build artifacts
- `archive/unused_bib/` contains bibliography files not referenced by the paper
- `archive/unused_figures/` contains figure/image files not referenced by the paper
- `archive/sim_summary_delivery/` contains old packaged delivery output from the removed simulation summary folder

## Build

Use:

```bash
latexmk -pdf -interaction=nonstopmode -outdir=output/build neurips_2024.tex
```

The compiled PDF is produced at `output/build/neurips_2024.pdf`. A copy is also kept at `output/neurips_2024.pdf`.

## Notes

- The main paper currently reads only `neurips_2024.tex`, `neurips_2024.sty`, `ref.bib`, and the six top-level files in `figures/`.
- `dataset_info/` and `figures/imported/` are support material and do not affect the paper build.
