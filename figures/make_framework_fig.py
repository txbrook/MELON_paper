"""
MELON Fig. 1 — framework schematic (matches hand-drawn sketch).

Layout:
  [1 Input]  [2 Augment A/B]  [3 Shared encoders]  -->  [5 Fusion Transformer]
      -->  [6 Masked views M1/M2/Mstar]  -->  [7 PID contrastive loss]  -->  [8 Output]
                                             ^
                                  [4 NeighborhoodGAT] (above, dashed B arrow down)

  Numbered stage ladder along the bottom: 1 -> 2 -> 3 -> 5 -> 6 -> 7 -> 8
  with section refs (§3.3, §3.4, §3.5).

Run:
    python make_framework_fig.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Rectangle, Polygon, Ellipse)
from matplotlib.colors import ListedColormap
import numpy as np

# ---------- palette ----------
C_MOD1   = "#4c78a8"   # modality 1 (blue)
C_MOD2   = "#dd8452"   # modality 2 (orange)
C_SPAT   = "#5f8f71"   # spatial / GAT (green)
C_JOINT  = "#8172b2"   # joint view (violet)
C_GRAY   = "#f1f1f1"
C_EDGE   = "#1f1f1f"
C_MUTED  = "#666666"
C_VIEWA  = "#f6f8fc"
C_VIEWB  = "#fcf7f1"
C_PID    = "#fbf9fd"
C_GATBG  = "#f7fbf8"
C_FUSBG  = "#fafafa"

FS_T  = 11
FS_L  = 10
FS_S  = 8.4
FS_XS = 7.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": FS_L,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def export_input_panel(data, cmap, label, stem):
    panel_fig = plt.figure(figsize=(1.55, 1.55))
    panel_ax = panel_fig.add_axes([0, 0, 1, 1])
    panel_ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest")
    panel_ax.set_xticks([])
    panel_ax.set_yticks([])
    for spine in panel_ax.spines.values():
        spine.set_color(C_EDGE)
        spine.set_linewidth(0.8)
    panel_ax.text(
        1.05, 0.98, label,
        transform=panel_ax.transAxes,
        color=C_EDGE,
        fontsize=FS_L,
        fontweight="bold",
        ha="left",
        va="top",
    )
    panel_fig.savefig(f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    panel_fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(panel_fig)


def export_spatial_panel(stem):
    panel_fig = plt.figure(figsize=(1.55, 1.55))
    panel_ax = panel_fig.add_axes([0, 0, 1, 1])
    panel_ax.set_xlim(0, 12)
    panel_ax.set_ylim(0, 12)
    panel_ax.set_aspect("equal")
    panel_ax.set_xticks([])
    panel_ax.set_yticks([])
    panel_ax.set_facecolor("white")
    for spine in panel_ax.spines.values():
        spine.set_color(C_EDGE)
        spine.set_linewidth(0.8)

    n_grid = 6
    margin = 1.2
    coords = np.linspace(margin, 12 - margin, n_grid)
    for i in coords:
        for j in coords:
            panel_ax.add_patch(Circle((i, j), 0.36, fc="white", ec=C_SPAT, lw=0.8))

    panel_ax.text(
        1.05, 0.98, r"$\mathbf{S}$",
        transform=panel_ax.transAxes,
        color=C_EDGE,
        fontsize=FS_L,
        fontweight="bold",
        ha="left",
        va="top",
    )
    panel_fig.savefig(f"{stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    panel_fig.savefig(f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(panel_fig)


# ---------- canvas ----------
fig = plt.figure(figsize=(17.6, 6.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 178)
ax.set_ylim(21, 88)
ax.axis("off")


# ---------- helpers ----------
def rbox(x, y, w, h, fc="white", ec=C_EDGE, lw=0.9, rounding=0.55, ls="-"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.2,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(x1, y1, x2, y2, color=C_EDGE, lw=1.0, style="-|>", ls="-",
          mscale=12):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=mscale,
                        color=color, lw=lw, linestyle=ls,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def txt(x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", FS_L)
    kw.setdefault("color", C_EDGE)
    ax.text(x, y, s, **kw)


# =================================================================
# (1) INPUT — two square heatmaps + a square coordinate grid
# =================================================================
input_cx = 13

# Heatmap X^(1) (blue)
hm1_x, hm1_y, hm1_s = input_cx - 6, 66, 12
mod1_data = np.random.default_rng(1).random((10, 10))
ax.imshow(mod1_data, extent=(hm1_x, hm1_x + hm1_s, hm1_y, hm1_y + hm1_s),
          cmap="Blues", aspect="auto", zorder=1, alpha=0.88)
ax.add_patch(Rectangle((hm1_x, hm1_y), hm1_s, hm1_s,
                       fc="none", ec=C_EDGE, lw=0.8))
ax.text(hm1_x + hm1_s + 0.8, hm1_y + hm1_s - 1.5,
        r"$\mathbf{X}^{(1)}$", color=C_MOD1, fontsize=FS_L,
        weight="bold", va="top", ha="left")

# Heatmap X^(2) (orange)
hm2_x, hm2_y, hm2_s = input_cx - 6, 52, 12
mod2_data = np.random.default_rng(7).random((10, 10))
ax.imshow(mod2_data, extent=(hm2_x, hm2_x + hm2_s, hm2_y, hm2_y + hm2_s),
          cmap="Oranges", aspect="auto", zorder=1, alpha=0.88)
ax.add_patch(Rectangle((hm2_x, hm2_y), hm2_s, hm2_s,
                       fc="none", ec=C_EDGE, lw=0.8))
ax.text(hm2_x + hm2_s + 0.8, hm2_y + hm2_s - 1.5,
        r"$\mathbf{X}^{(2)}$", color=C_MOD2, fontsize=FS_L,
        weight="bold", va="top", ha="left")

# Coordinate grid S — uniformly distributed hollow green circles
sg_x, sg_y, sg_s = input_cx - 6, 38, 12
ax.add_patch(Rectangle((sg_x, sg_y), sg_s, sg_s,
                       fc="white", ec=C_EDGE, lw=0.8))
n_grid = 6
margin = 1.2
coords = np.linspace(margin, sg_s - margin, n_grid)
for i in coords:
    for j in coords:
        ax.add_patch(Circle((sg_x + i, sg_y + j), 0.55,
                            fc="white", ec=C_SPAT, lw=0.9, zorder=5))
ax.text(sg_x + sg_s + 0.8, sg_y + sg_s - 1.5,
        r"$\mathbf{S}$", color=C_SPAT, fontsize=FS_L,
        weight="bold", va="top", ha="left")

# =================================================================
# (2) AUGMENTATION — 4 boxes: (Augment A, Augment B) x (X^(1), X^(2))
# =================================================================
# Two pairs of boxes: the top pair augments X^(1), the bottom pair augments X^(2)
aug_w, aug_h = 14, 6.5
# top pair (X^(1))
augA1_y, augB1_y = 69.0, 61.5
# bottom pair (X^(2))
augA2_y, augB2_y = 54.5, 47.0
aug_x = 28

def aug_box(y, title, formula, face, edge):
    rbox(aug_x, y, aug_w, aug_h, fc=face, ec=edge, lw=0.8)
    ax.text(aug_x + aug_w / 2, y + aug_h * 0.68, title,
            fontsize=FS_S, weight="bold", ha="center", va="center")
    ax.text(aug_x + aug_w / 2, y + aug_h * 0.28, formula,
            fontsize=FS_XS, color=C_MUTED, ha="center", va="center",
            style="italic")

aug_box(augA1_y, "Augment A", r"$t_1(\mathbf{X}^{(1)})$", C_VIEWA, "#c9dff5")
aug_box(augB1_y, "Augment B", r"$t_2(\mathbf{X}^{(1)})$", C_VIEWB, "#f3d8bd")
aug_box(augA2_y, "Augment A", r"$t_1(\mathbf{X}^{(2)})$", C_VIEWA, "#c9dff5")
aug_box(augB2_y, "Augment B", r"$t_2(\mathbf{X}^{(2)})$", C_VIEWB, "#f3d8bd")

# arrows: X^(1) center (~13, 72) -> Augment A (top) and Augment B (second)
x1_src_x, x1_src_y = 19, 72
x2_src_x, x2_src_y = 19, 58

arrow(x1_src_x, x1_src_y, aug_x, augA1_y + aug_h/2)
arrow(x1_src_x, x1_src_y, aug_x, augB1_y + aug_h/2)
arrow(x2_src_x, x2_src_y, aug_x, augA2_y + aug_h/2)
arrow(x2_src_x, x2_src_y, aug_x, augB2_y + aug_h/2)


# =================================================================
# (3) MODALITY ENCODERS — 4 trapezoids, narrow left -> wide right,
#     one per augmentation block (reflecting shared weights per modality)
# =================================================================
enc_x_left = 47
enc_x_right = 62
enc_h_left = 2.4   # narrow side (input)
enc_h_right = 6.2  # wide side (output)

def trapezoid(cy, face, label_T=None):
    pts = np.array([
        [enc_x_left,  cy + enc_h_left / 2],
        [enc_x_right, cy + enc_h_right / 2],
        [enc_x_right, cy - enc_h_right / 2],
        [enc_x_left,  cy - enc_h_left / 2],
    ])
    ax.add_patch(Polygon(pts, closed=True, fc=face, ec=C_EDGE, lw=0.75,
                         zorder=2))
    if label_T:
        ax.text(enc_x_right + 1.0, cy, label_T, fontsize=FS_S,
                va="center", ha="left", weight="bold")

# Trapezoid centers aligned with each augmentation block's mid-y
t_y_A1 = augA1_y + aug_h / 2
t_y_B1 = augB1_y + aug_h / 2
t_y_A2 = augA2_y + aug_h / 2
t_y_B2 = augB2_y + aug_h / 2

trapezoid(t_y_A1, C_MOD1, label_T=r"$\mathbf{T}^{(1)}_A$")
trapezoid(t_y_B1, C_MOD1, label_T=r"$\mathbf{T}^{(1)}_B$")
trapezoid(t_y_A2, C_MOD2, label_T=r"$\mathbf{T}^{(2)}_A$")
trapezoid(t_y_B2, C_MOD2, label_T=r"$\mathbf{T}^{(2)}_B$")

# shared-weights dotted frames: group the two X^(1) trapezoids, and the two X^(2)
def shared_frame(y_top, y_bot, label):
    pad = 1.2
    rect = Rectangle((enc_x_left - pad, y_bot - enc_h_right/2 - pad),
                     (enc_x_right - enc_x_left) + 2 * pad + 6,
                     (y_top - y_bot) + enc_h_right + 2 * pad,
                     fc="none", ec="#8c8c8c", lw=0.7, linestyle=(0, (2.5, 2)))
    ax.add_patch(rect)

shared_frame(t_y_A1, t_y_B1, "shared $f_{m_1}$")
shared_frame(t_y_A2, t_y_B2, "shared $f_{m_2}$")


# arrows from each augmentation box -> its trapezoid (narrow left edge)
arrow(aug_x + aug_w, t_y_A1, enc_x_left, t_y_A1)
arrow(aug_x + aug_w, t_y_B1, enc_x_left, t_y_B1)
arrow(aug_x + aug_w, t_y_A2, enc_x_left, t_y_A2)
arrow(aug_x + aug_w, t_y_B2, enc_x_left, t_y_B2)


# =================================================================
# (4) NeighborhoodGAT — placed inline between S and Fusion (below the
#     augmentation + encoder row)
# =================================================================
gat_x, gat_y, gat_w, gat_h = 24, 28, 46, 13
rbox(gat_x, gat_y, gat_w, gat_h, fc=C_GATBG, ec=C_SPAT, lw=1.0, rounding=0.8)
txt(gat_x + gat_w/2, gat_y + gat_h + 1.2, "NeighborhoodGAT",
    fontsize=FS_T, weight="bold", color=C_EDGE)

# "Spatial neighbor graph" panel (left inside GAT box)
sng_cx = gat_x + 8
# title inside the top of GAT box
ax.text(sng_cx, gat_y + gat_h - 1.3, "Spatial neighbor\ngraph",
        ha="center", va="top", fontsize=FS_XS)
# white inner panel (square-ish)
sng_w, sng_h = 9, 7.0
sng_left = sng_cx - sng_w / 2
sng_bottom = gat_y + 0.8
# graph region (upper 70% of the panel)
g_cy = sng_bottom + sng_h * 0.63
node_pos = [
    (sng_cx - 2.3, g_cy + 1.3),
    (sng_cx - 0.1, g_cy + 1.8),
    (sng_cx + 2.3, g_cy + 1.2),
    (sng_cx - 1.3, g_cy - 0.3),
    (sng_cx + 1.3, g_cy - 0.3),
    (sng_cx - 1.9, g_cy - 1.9),
    (sng_cx + 0.6, g_cy - 1.9),
]
edges = [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4),
         (3, 4), (3, 5), (4, 6), (5, 6)]
for (a, b) in edges:
    ax.plot([node_pos[a][0], node_pos[b][0]],
            [node_pos[a][1], node_pos[b][1]],
            color=C_EDGE, lw=0.7, zorder=2)
for (nx, ny) in node_pos:
    ax.add_patch(Circle((nx, ny), 0.48, fc="none", ec=C_EDGE,
                        lw=0.7, zorder=3))
# "○ Spots" legend at bottom of the panel
leg_y = sng_bottom + 0.9
ax.add_patch(Circle((sng_cx - 1.7, leg_y), 0.48,
                    fc="none", ec=C_EDGE, lw=0.7))
ax.text(sng_cx - 0.7, leg_y, "Spots",
        fontsize=FS_XS, va="center", ha="left")

# attention heads — stacked small cards
hx, hy = gat_x + 24, gat_y + 6
for i in range(3):
    ax.add_patch(Rectangle((hx - 3 + i * 0.8, hy - 3 - i * 0.8),
                           6, 6, fc="white", ec=C_SPAT, lw=0.5))
# dots inside top card indicating heads
for d in range(3):
    ax.add_patch(Circle((hx - 1 + d * 1, hy), 0.35,
                        fc=C_EDGE, ec="none"))
txt(hx, gat_y + 1.3, "attention\nheads", fontsize=FS_XS, style="italic")
ax.text(hx + 5.3, hy - 4.5, "$\\cdots$", fontsize=FS_S, color=C_EDGE)

# arrow between spatial neighbor graph panel -> attention heads
arrow(sng_left + sng_w, g_cy, hx - 3.2, hy, lw=1.0, mscale=10)

# B heatmap (viridis)
bx, by, bw, bh = gat_x + 40, gat_y + 2.3, 10, 8
Bmat = np.add.outer(np.linspace(0, 1, 12), np.linspace(0, 1, 12))
Bmat += 0.4 * np.eye(12)
ax.imshow(Bmat, extent=(bx, bx + bw, by, by + bh),
          cmap="Greens", aspect="auto", zorder=2, alpha=0.92)
# frame
ax.add_patch(Rectangle((bx, by), bw, bh, fc="none", ec=C_EDGE, lw=0.6))
ax.text(bx + bw + 1, by + bh - 1,
        r"$\mathbf{B} \in \mathbb{R}^{N\times N}$",
        fontsize=FS_S, va="top", ha="left")

# arrow heads -> B
arrow(hx + 4, hy, bx, by + bh/2, lw=1.0, mscale=10)

# arrow from the S coordinate grid into the GAT box
arrow(20, 44, gat_x, gat_y + gat_h/2, lw=1.2)

# dashed green arrow from GAT down into fusion (drawn later to land on fusion top)


# =================================================================
# (5) FUSION TRANSFORMER — big hero box
# =================================================================
fx, fy, fw, fh = 76, 40, 24, 31
rbox(fx, fy, fw, fh, fc=C_FUSBG, ec=C_EDGE, lw=1.0, rounding=0.7)


# attention equation
ax.text(fx + fw/2, fy + fh * 0.62,
        r"$\mathrm{softmax}\!\left(\dfrac{\mathbf{QK}^{T}}{\sqrt{d}} + \mathbf{B}\right) V$",
        ha="center", va="center", fontsize=FS_T + 1.5, color=C_EDGE)

txt(fx + fw/2, fy + fh * 0.30,
    "pooled spot\nembedding $\\mathbf{z}$",
    fontsize=FS_S, style="italic", color=C_MUTED)

# encoder outputs (wide right side of trapezoids) -> fusion transformer
arrow(enc_x_right + 5, t_y_A1, fx, fy + fh * 0.82, lw=1.2)
arrow(enc_x_right + 5, t_y_B1, fx, fy + fh * 0.62, lw=1.2)
arrow(enc_x_right + 5, t_y_A2, fx, fy + fh * 0.38, lw=1.2)
arrow(enc_x_right + 5, t_y_B2, fx, fy + fh * 0.18, lw=1.2)

# GAT B arrow -> fusion bottom-left (dashed green), since GAT now sits
# below/beside the main flow
arrow(gat_x + gat_w, gat_y + gat_h/2, fx, fy + 6,
      color=C_SPAT, lw=1.2, ls=(0, (3, 2)), mscale=12)
ax.text(fx - 3.2, fy + 7.5, r"$\mathbf{B}$",
        color=C_SPAT, fontsize=FS_T, weight="bold",
        ha="right", va="center")


# =================================================================
# (6) MASKED VIEWS  — 3 stacked boxes to the right of fusion
# =================================================================
mx, my_top = 106, 63.5
mw, mh, gap = 16, 9, 2

def msk_token(x, y, w=2.0, h=None):
    """Hatched/dashed black box representing a [MSK] token."""
    if h is None:
        h = mh - 2
    ax.add_patch(Rectangle((x, y), w, h, fc="#d8d8d8", ec=C_EDGE, lw=0.4,
                           hatch="///"))

def mod_bar(x, y, color, w=2.0, h=None):
    if h is None:
        h = mh - 2
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=C_EDGE, lw=0.4))

def mask_box(x, y, label, border, content):
    """content: sequence of token kinds — 'mod1', 'mod2', 'msk'."""
    rbox(x, y, mw, mh, fc="white", ec=border, lw=1.0, rounding=0.55)
    txt(x + 4.2, y + mh/2 + 0.3, label, fontsize=FS_L, weight="bold")
    # right-aligned token slot pair
    slot_x = x + mw - 5.0
    for i, kind in enumerate(content):
        bx_ = slot_x + i * 2.3
        if kind == "mod1":
            mod_bar(bx_, y + 1, C_MOD1)
        elif kind == "mod2":
            mod_bar(bx_, y + 1, C_MOD2)
        elif kind == "msk":
            msk_token(bx_, y + 1)

# M^(1): blue + [MSK]
mask_box(mx, my_top, r"$\mathbf{M}^{(1)}$", C_MOD1,
         content=("mod1", "msk"))
# M^(2): [MSK] + orange
mask_box(mx, my_top - (mh + gap), r"$\mathbf{M}^{(2)}$", C_MOD2,
         content=("msk", "mod2"))
# M^(*): blue + orange (joint)
mask_box(mx, my_top - 2 * (mh + gap), r"$\mathbf{M}^{(\star)}$", C_JOINT,
         content=("mod1", "mod2"))

# fusion -> masked views (3 arrows, spread vertically on fusion's right edge)
fusion_right_x = fx + fw
fusion_anchor_ys = [fy + fh * 0.80, fy + fh * 0.50, fy + fh * 0.20]
for i in range(3):
    y_mid = my_top - i * (mh + gap) + mh/2
    arrow(fusion_right_x, fusion_anchor_ys[i], mx, y_mid,
          lw=1.2, mscale=11)

# six embedding dots — 2 per mask (augment A above, augment B below)
dot_colors = [C_MOD1, C_MOD2, C_JOINT]
dot_x = mx + mw + 2.2
for i in range(3):
    ybase = my_top - i * (mh + gap) + mh/2
    col = dot_colors[i]
    ax.add_patch(Circle((dot_x, ybase + 1.3), 0.7,
                        fc=col, ec=C_EDGE, lw=0.4, zorder=4))
    ax.add_patch(Circle((dot_x, ybase - 1.3), 0.7,
                        fc=col, ec=C_EDGE, lw=0.4, zorder=4))
# A/B annotation on the topmost dot pair
ax.text(dot_x + 1.2, my_top + mh/2 + 1.3, "A",
        fontsize=FS_XS, color=C_MUTED, va="center", ha="left")
ax.text(dot_x + 1.2, my_top + mh/2 - 1.3, "B",
        fontsize=FS_XS, color=C_MUTED, va="center", ha="left")

# caption
txt(mx + mw/2, my_top - 2 * (mh + gap) - 4, "2($n$+1) = 6\nembeddings",
    fontsize=FS_S, style="italic", color=C_MUTED)


# =================================================================
# (7) PID CONTRASTIVE LOSS — faithful 2×3 grid of 6 embeddings with
#     cross-augmentation InfoNCE pairings (see paper §3.5, Eq. 3–4)
# =================================================================
px, py, pw, ph = 130, 39, 30, 40
rbox(px, py, pw, ph, fc=C_PID, ec=C_JOINT, lw=0.95, rounding=0.8)

# Column x's (Augment A on the left, Augment B on the right)
colA_x = px + 6.0
colB_x = px + pw - 6.0
# Row y's (m1 on top, m2 middle, * on bottom)
row1_y = py + ph - 8.5    # modality-1 mask
row2_y = py + ph/2        # modality-2 mask
row3_y = py + 8.5         # joint mask *

# column headers
ax.text(colA_x, py + ph - 2.5, "Augment A", fontsize=FS_XS,
        color=C_MUTED, ha="center", va="center", style="italic")
ax.text(colB_x, py + ph - 2.5, "Augment B", fontsize=FS_XS,
        color=C_MUTED, ha="center", va="center", style="italic")

# 6 embedding dots (single point per spot, not a cloud)
def emb_dot(x, y, color, label, label_side="left"):
    ax.add_patch(Circle((x, y), 1.2, fc=color, ec=C_EDGE,
                        lw=0.5, zorder=4))
    dx_ = -2.2 if label_side == "left" else 2.2
    ha_ = "right" if label_side == "left" else "left"
    ax.text(x + dx_, y, label, color=color, fontsize=FS_XS,
            weight="bold", ha=ha_, va="center")

emb_dot(colA_x, row1_y, C_MOD1,  r"$z'^{(1)}$",    "left")
emb_dot(colA_x, row2_y, C_MOD2,  r"$z'^{(2)}$",    "left")
emb_dot(colA_x, row3_y, C_JOINT, r"$z'^{(\star)}$","left")
emb_dot(colB_x, row1_y, C_MOD1,  r"$z''^{(1)}$",    "right")
emb_dot(colB_x, row2_y, C_MOD2,  r"$z''^{(2)}$",    "right")
emb_dot(colB_x, row3_y, C_JOINT, r"$z''^{(\star)}$","right")

# Cross-augmentation arrows (InfoNCE pairs)
# Double-headed (symmetric) connections. Each loss has TWO cross pairs.
def pair(p1, p2, color, ls="-"):
    a = FancyArrowPatch(p1, p2,
                        arrowstyle="<->", mutation_scale=10,
                        color=color, lw=1.0, linestyle=ls,
                        shrinkA=4, shrinkB=4)
    ax.add_patch(a)

# L_m1: z'^(1) <-> z''^(*) AND z''^(1) <-> z'^(*)
pair((colA_x, row1_y), (colB_x, row3_y), C_MOD1)
pair((colB_x, row1_y), (colA_x, row3_y), C_MOD1, ls=(0, (3, 2)))
# L_m2: z'^(2) <-> z''^(*) AND z''^(2) <-> z'^(*)
pair((colA_x, row2_y), (colB_x, row3_y), C_MOD2)
pair((colB_x, row2_y), (colA_x, row3_y), C_MOD2, ls=(0, (3, 2)))
# L_*: z'^(*) <-> z''^(*)   (self-consistency between joint views)
pair((colA_x, row3_y), (colB_x, row3_y), C_JOINT)

# loss labels near each set of connections
ax.text(px + pw/2, row1_y + 1.4, r"$\mathcal{L}_{m_1}$",
        color=C_MOD1, fontsize=FS_S, weight="bold",
        ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
ax.text(px + pw/2, row2_y + 1.2, r"$\mathcal{L}_{m_2}$",
        color=C_MOD2, fontsize=FS_S, weight="bold",
        ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
ax.text(px + pw/2, row3_y + 2.4, r"$\mathcal{L}_{\star}$",
        color=C_JOINT, fontsize=FS_S, weight="bold",
        ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))

# total loss equation under the PID box
ax.text(px + pw/2, py - 3.5,
        r"$\mathcal{L} = \frac{1}{3}(\mathcal{L}_{m_1}+\mathcal{L}_{m_2}+\mathcal{L}_{\star})$",
        ha="center", fontsize=FS_L, color=C_EDGE)

# masks -> PID box: each mask supplies BOTH its A and B embeddings,
# so route to the left column (A) row of that mask
mask_targets = [
    (colA_x - 1.5, row1_y),   # M^(1) -> z'^(1)
    (colA_x - 1.5, row2_y),   # M^(2) -> z'^(2)
    (colA_x - 1.5, row3_y),   # M^(*) -> z'^(*)
]
mask_colors = [C_MOD1, C_MOD2, C_JOINT]
for i in range(3):
    y_src = my_top - i * (mh + gap) + mh/2
    arrow(mx + mw, y_src, *mask_targets[i],
          color=mask_colors[i], lw=1.1, mscale=10)


# =================================================================
# (8) OUTPUT — spot embedding on real E15 brain (placeholder thumbnail)
# =================================================================
ox, oy, ow, oh = 166, 48, 10, 18
rbox(ox, oy, ow, oh, fc="white", ec=C_EDGE, lw=0.9)
txt(ox + ow/2, oy + oh + 1.5, "spot\nembedding",
    fontsize=FS_S, style="italic")
# mock brain clustering — swap this with a real thumbnail later
rng4 = np.random.default_rng(9)
cluster_map = rng4.integers(0, 6, size=(14, 8))
cluster_cmap = ListedColormap([
    "#4c78a8",
    "#dd8452",
    "#5f8f71",
    "#8172b2",
    "#c44e52",
    "#937860",
])
ax.imshow(cluster_map, extent=(ox + 0.4, ox + ow - 0.4,
                               oy + 0.8, oy + oh - 0.8),
          cmap=cluster_cmap, aspect="auto", alpha=0.82)

# arrow from PID box (right edge, middle) -> output
arrow(px + pw, py + ph/2, ox, oy + oh/2, lw=1.2)


# =================================================================
# STAGE TITLES — one label above each main box
# =================================================================
# Columns where the top of the main content is around y~74-80 → titles at y=84
# The Fusion Transformer and NeighborhoodGAT already carry their own titles
# (GAT sits above the flow; Fusion title is inside the box), so we skip those.
def stage_title(x, y, text):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=FS_L, weight="bold", color=C_EDGE)

stage_title(13,         83, "Input")
stage_title(35,         83, "Augmentation")
stage_title(56,         83, "Modality Encoders")
stage_title(fx + fw/2,  83, "Fusion Transformer")
stage_title(mx + mw/2,  83, "Masked Embeddings")
stage_title(px + pw/2,  83, "PID-Contrastive Loss")
stage_title(ox + ow/2,  83, "Output")

# ---------- export ----------
out_pdf = "fig_melon_framework.pdf"
out_png = "fig_melon_framework.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1, dpi=220)
export_input_panel(mod1_data, "Blues", r"$\mathbf{X}^{(1)}$", "fig_melon_framework_input_x1")
export_input_panel(mod2_data, "Oranges", r"$\mathbf{X}^{(2)}$", "fig_melon_framework_input_x2")
export_spatial_panel("fig_melon_framework_input_s")
print(
    "wrote "
    f"{out_pdf}, {out_png}, "
    "fig_melon_framework_input_x1.(png/svg), "
    "fig_melon_framework_input_x2.(png/svg), "
    "fig_melon_framework_input_s.(png/svg)"
)
