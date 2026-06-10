#!/usr/bin/env python3
"""
SBA conceptual figure (2 panels) -- pure numpy/matplotlib, no torch.

Surface:  f(x1,x2) = sigmoid( A*x1 + B*(x2 - KAP*x1^2) + C )
  - A >> B          : x1 dominates, x2 supports
  - x2 = KAP*x1^2   : curved data manifold (ridge of high output)
  - C != 0          : bias (origin is not the zero-output point)

The argument is single-baseline vs distribution-of-baselines:

Panel 1 (single baseline)  IG  -> SBA
    one baseline x', one straight IG line (cuts off-manifold) vs
    one SBA bridge that bows onto the ridge.

Panel 2 (baseline cloud)   EG  -> SBA-D
    a cloud of baselines ~ N(0,0); EG = fan of independent straight chords
    (several leave the manifold) vs SBA-D = same cloud transported along
    discretized bridge paths that converge onto the ridge to x.

Outputs: sba_concept_fig.pdf (vector) and sba_concept_fig.png (300 dpi)

NOTE: illustrative 2-feature construction, not the ImageNet/ResNet setting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ----- model -----
A, B, KAP, C = 3.0, 0.6, 0.9, -0.4

def sig(z):
    return 1.0 / (1.0 + np.exp(-z))

def f(x1, x2):
    return sig(A * x1 + B * (x2 - KAP * x1**2) + C)

# ----- grid -----
LO, HI, N = -2.2, 2.2, 80
g = np.linspace(LO, HI, N)
X1, X2 = np.meshgrid(g, g)
Z = f(X1, X2)

# ----- key points -----
ix1, ix2 = 1.5, KAP * 1.5**2      # input (high output, on ridge)
sbx1 = -C / A                     # single baseline on ridge (near-zero output)
sbx2 = KAP * sbx1**2

# baseline cloud scattered around (0,0)
rng = np.random.default_rng(0)
NB = 9
cloud_x1 = rng.normal(0.0, 0.45, NB)
cloud_x2 = rng.normal(0.0, 0.45, NB)

# ----- styling -----
BLUE_CMAP = cm.get_cmap("Blues")
RED    = "#A32D2D"   # IG / EG straight
PURPLE = "#534AB7"   # SBA / SBA-D bridge
GREEN  = "#0F6E56"   # manifold ridge
GRAY   = "#444441"
CAMERA = dict(elev=30, azim=-150)


def base_surface(ax, alpha=0.32):
    ax.plot_surface(X1, X2, Z, cmap=BLUE_CMAP, alpha=alpha,
                    linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax.set_xlabel(r"$x_1$ (dominant)")
    ax.set_ylabel(r"$x_2$ (support)")
    ax.set_zlabel(r"$y = f$")
    ax.set_zlim(0, 1.05)
    ax.view_init(**CAMERA)
    # data manifold ridge
    t = np.linspace(LO, HI, 200)
    keep = KAP * t**2 <= HI
    ax.plot(t[keep], KAP * t[keep]**2, f(t[keep], KAP * t[keep]**2) + 0.012,
            color=GREEN, lw=2.6, linestyle=(0, (1, 1)), zorder=9,
            label="data manifold (ridge)")


def straight_path(x1a, x2a, n=80, dz=0.015):
    s = np.linspace(0, 1, n)
    px1 = x1a + s * (ix1 - x1a)
    px2 = x2a + s * (ix2 - x2a)
    return px1, px2, f(px1, px2) + dz


def bridge_path(x1a, x2a, n=80, dz=0.015):
    # bow onto the ridge: interpolate x1 linearly, pull x2 toward ridge KAP*x1^2
    s = np.linspace(0, 1, n)
    px1 = x1a + s * (ix1 - x1a)
    target = KAP * px1**2
    px2 = (1 - s) * x2a + s * target
    # blend so it leaves the cloud naturally then snaps to ridge
    px2 = (1 - s**1.5) * px2 + s**1.5 * target
    return px1, px2, f(px1, px2) + dz


def panel1(ax):
    base_surface(ax)
    # single baseline
    sx, sy, sz = straight_path(sbx1, sbx2)
    bx, by, bz = bridge_path(sbx1, sbx2)
    ax.plot(sx, sy, sz, color=RED, lw=4.5, label="IG straight path",
            solid_capstyle="round", zorder=10)
    ax.plot(bx, by, bz, color=PURPLE, lw=4.5, label="SBA bridge path",
            solid_capstyle="round", zorder=10)
    for (xx, yy, lab) in [(sbx1, sbx2, r"$x'$ baseline"), (ix1, ix2, r"$x$ input")]:
        ax.scatter([xx], [yy], [f(xx, yy) + 0.02], color=GRAY, s=40)
        ax.text(xx, yy, f(xx, yy) + 0.06, lab, fontsize=9, color=GRAY)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title("(a) single baseline:  IG  →  SBA", fontsize=11)


def panel2(ax):
    base_surface(ax)
    # cloud of baselines around (0,0)
    for k in range(NB):
        x1a, x2a = cloud_x1[k], cloud_x2[k]
        sx, sy, sz = straight_path(x1a, x2a)
        bx, by, bz = bridge_path(x1a, x2a)
        ax.plot(sx, sy, sz, color=RED, lw=2.0, alpha=0.85, zorder=10,
                solid_capstyle="round",
                label="EG straight fan" if k == 0 else None)
        ax.plot(bx, by, bz, color=PURPLE, lw=2.0, alpha=0.9, zorder=10,
                solid_capstyle="round",
                label="SBA-D bridge fan" if k == 0 else None)
        ax.scatter([x1a], [x2a], [f(x1a, x2a) + 0.02], color=GRAY, s=16,
                   alpha=0.85, zorder=11)
    ax.scatter([ix1], [ix2], [f(ix1, ix2) + 0.02], color=GRAY, s=40)
    ax.text(ix1, ix2, f(ix1, ix2) + 0.06, r"$x$ input", fontsize=9, color=GRAY)
    ax.text(0, 0, f(0, 0) + 0.10, r"$\mu_0$ baselines", fontsize=9, color=GRAY)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.set_title("(b) baseline distribution:  EG  →  SBA-D", fontsize=11)


def main():
    fig = plt.figure(figsize=(13, 5.5))
    panel1(fig.add_subplot(1, 2, 1, projection="3d"))
    panel2(fig.add_subplot(1, 2, 2, projection="3d"))
    fig.tight_layout()
    fig.savefig("sba_concept_fig.pdf", bbox_inches="tight")
    fig.savefig("sba_concept_fig.png", dpi=300, bbox_inches="tight")
    print("wrote sba_concept_fig.pdf and sba_concept_fig.png")


if __name__ == "__main__":
    main()