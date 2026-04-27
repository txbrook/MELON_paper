# Cross-Modal Prediction Experiment — Protocol

## Goal
Show that MELON's embedding preserves **cross-modal synergistic information** better than baselines, by measuring how well each method's frozen embedding can predict one modality from the *other* modality alone. Ground truth = measured data (not human labels).

## Hypothesis
If a method's embedding only captures shared (redundant) structure, then masking one modality at inference time should still let it reconstruct the common signal — but it cannot recover modality-specific or synergistic content. Methods that preserve synergy (MELON, by PID design) should predict held-out modality values more accurately than shared-only baselines (MOFA+, MEFISTO, SpatialGlue, MISO).

---

## Datasets
Use the three main-paper datasets:
- Mouse E15.5 Brain (RNA + ATAC, 1,949 spots)
- Human Lymph Node (RNA + ADT, 3,484 spots)
- Mouse E13 Embryo (RNA + ATAC, 2,187 spots)

---

## Two prediction directions per dataset
For modality pair $(A, B)$:
1. **A → B:** predict modality B from a spot embedding computed with modality B masked.
2. **B → A:** predict modality A from a spot embedding computed with modality A masked.

Report both directions; they test symmetric synergy preservation.

---

## Protocol (per method, per dataset, per direction)

### Step 1 — Train the representation model (standard)
Train MELON / SpatialGlue / MISO / MultiGate / TotalVI / MOFA+ / MEFISTO / Seurat-WNN on the full paired dataset using their **published default hyperparameters** (same as main benchmark).

### Step 2 — Produce masked embeddings at inference
For each spot $i$, compute an embedding $z_i^{[A]}$ using **only modality A** (modality B replaced with zeros / mask token / mean imputation — method-specific, documented per method).

- **MELON:** use the built-in `[MSK]` token single-modality embedding $z^{[A]}$ from §3.2 — no retraining needed.
- **Baselines without masking support:** replace modality B with its per-feature mean (standard zero-knowledge baseline), pass through the trained model, take the latent.
- Document the masking strategy for each method in an appendix table; reviewers will check this.

### Step 3 — Train linear probe to predict held-out modality
For each held-out feature in modality B, fit a **ridge regression** (closed form, no tuning required):

$$\hat{x}_i^{(B,j)} = w_j^\top z_i^{[A]} + b_j$$

where $j$ indexes features in modality B.

- Use 5-fold cross-validation across **spots** (not features).
- Ridge penalty $\lambda = 1.0$ fixed (no tuning, to avoid overfitting comparison advantage).
- Separate probe per target feature $j$, or a single multi-output ridge — either works, report which.

### Step 4 — Metrics
For each test fold, compute per-feature Pearson correlation between predicted and measured modality B, then average over features and folds.

Report:
- **Mean Pearson $r$** across features (primary metric).
- **Mean $R^2$** (secondary, since $R^2$ can be negative and penalizes bias).
- **Top-$K$ feature recovery:** fraction of highly variable features (top-500 HVGs / top-500 peaks / all proteins) with $r > 0.3$.

---

## Expected table layout

| Method      | E15 RNA→ATAC $r$ | E15 ATAC→RNA $r$ | Lymph RNA→ADT $r$ | Lymph ADT→RNA $r$ | E13 RNA→ATAC $r$ | E13 ATAC→RNA $r$ |
|-------------|------------------|------------------|-------------------|-------------------|------------------|------------------|
| **MELON**   | **bold**         | **bold**         | **bold**          | **bold**          | **bold**         | **bold**         |
| MISO        |                  |                  |                   |                   |                  |                  |
| SpatialGlue |                  |                  |                   |                   |                  |                  |
| MultiGate   |                  |                  |                   |                   |                  |                  |
| TotalVI     |                  |                  |                   |                   |                  |                  |
| MOFA+       |                  |                  |                   |                   |                  |                  |
| MEFISTO     |                  |                  |                   |                   |                  |                  |
| Seurat-WNN  |                  |                  |                   |                   |                  |                  |

Bold the best per column. Report $r \pm \sigma$ over 5 folds.

---

## Controls (add to appendix)

1. **Random-embedding floor:** replace $z_i$ with $\mathcal{N}(0, I)$ of same dim. Sets the "no useful signal" baseline.
2. **Raw-A floor:** use raw modality-A features as "embedding" (no learned model). Shows the gain from representation learning beyond raw input.
3. **Joint-embedding ceiling:** for MELON, also report the joint embedding $z^{[\star]}$ that sees both modalities. Upper bound — MELON single-modality embeddings should approach this, baselines should not.

---

## Statistical test
For each column in the main table, run a paired Wilcoxon signed-rank test between MELON's per-feature $r$ values and each baseline's. Report $p$-values with Bonferroni correction across the 6 columns × 7 baselines. Flag $p < 0.01$ in the table footnote.

---

## Implementation checklist
- [ ] Freeze each method's trained model; save checkpoints.
- [ ] Write single masking routine per method (documented table).
- [ ] Write one shared probing script: `probe.py --method X --dataset Y --direction A2B`.
- [ ] Run 5-fold CV, 3 random seeds for MELON and deep baselines (linear methods are deterministic).
- [ ] Export a single CSV; render Table from CSV.
- [ ] Add controls to appendix.

---

## What this buys you in the paper

- **Direct PID validation on real data.** Your Prop. 1 says MELON preserves unique + synergistic info; this test operationalizes that claim on measured data.
- **No new ground truth required.** The held-out modality *is* the ground truth.
- **Resistant to reviewer pushback.** "Did you validate on real biology, not just simulation?" — yes, on measured RNA/ATAC/ADT.
- **Tight narrative with your existing simulation.** Simulation shows synergy recovery on a controlled signal; this shows synergy recovery on real measurements. Same story, two scales.

---

## Writing placement
- One paragraph in §4 (main), after Table 3 or as §4.3, with a 6-column summary table.
- Full results + controls + per-feature histograms in a new Appendix subsection (e.g., `\subsection{Cross-Modal Prediction}\label{app:cross-modal}`).
- Reference from checklist Q7 (statistical significance) as additional evidence.
