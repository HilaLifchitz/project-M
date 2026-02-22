"""
ExAdaptiveLandscape2Loci.py — Adaptive landscape for 2 loci only, with LD (linkage disequilibrium)

Same fitness model and plotting as the 2-locus case in ExAdaptiveLandscape, but adds
an LD term: genotype frequencies are computed from (f₁, f₂, D) instead of linkage equilibrium.

Genotype frequencies with LD:
  D = p₁₁ − f₁f₂  (linkage disequilibrium)
  p₁₁ = f₁f₂ + D
  p₁₀ = f₁(1−f₂) − D
  p₀₁ = (1−f₁)f₂ − D
  p₀₀ = (1−f₁)(1−f₂) + D

WHY D≠0 DOES NOT COVER THE FULL (f₁,f₂) DOMAIN:
  For genotype frequencies to be valid (0 ≤ p ≤ 1), D must satisfy Lewontin bounds:
  D_min ≤ D ≤ D_max, where
    D_max = min(f₁(1−f₂), (1−f₁)f₂)
    D_min = max(−f₁f₂, −(1−f₁)(1−f₂))
  When we fix D globally (e.g. D=0.1), only (f₁,f₂) where D lies in [D_min, D_max] are valid.
  The rest would give negative genotype frequencies. Those points are masked (white).
  So the colored region is the "feasible" (f₁,f₂) for that fixed D — a diamond-shaped subset.

Fitness: W([0,0])=1, W([0,1])=1+s₂, W([1,0])=1+s₁, W([1,1])=(1+s₁)(1+s₂)+eps₁₂
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def _f(x) -> float:
    """Convert to Python float for display."""
    return float(x)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def fitness_n2(s1: float, s2: float, eps_12: float = 0.0) -> np.ndarray:
    """
    Fitness of each genotype for n=2.
    Order: [0,0], [0,1], [1,0], [1,1]
    """
    w = np.array([
        1.0,
        1.0 + s2,
        1.0 + s1,
        (1.0 + s1) * (1.0 + s2) + eps_12,
    ])
    return w


# ---------------------------------------------------------------------------
# Genotype frequencies: LE vs with LD
# ---------------------------------------------------------------------------

def genotype_freqs_LE(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Linkage equilibrium: (p00, p01, p10, p11) from (f1, f2)."""
    p00 = (1 - f1) * (1 - f2)
    p01 = (1 - f1) * f2
    p10 = f1 * (1 - f2)
    p11 = f1 * f2
    return np.stack([p00, p01, p10, p11], axis=-1)


def genotype_freqs_with_LD(f1: np.ndarray, f2: np.ndarray, D: float) -> np.ndarray:
    """
    Genotype frequencies with linkage disequilibrium D.
    p11 = f1*f2 + D, p10 = f1(1-f2)-D, p01 = (1-f1)f2-D, p00 = (1-f1)(1-f2)+D
    Returns (p00, p01, p10, p11). Invalid points (negative freq) set to nan.
    """
    p11 = f1 * f2 + D
    p10 = f1 * (1 - f2) - D
    p01 = (1 - f1) * f2 - D
    p00 = (1 - f1) * (1 - f2) + D
    p = np.stack([p00, p01, p10, p11], axis=-1)
    valid = np.all(p >= -1e-12, axis=-1) & np.all(p <= 1 + 1e-12, axis=-1)
    p = np.where(valid[..., np.newaxis], p, np.nan)
    return p


def mean_fitness(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mean fitness = E[W] = sum_g p_g * W_g."""
    return np.dot(p, w)


def make_allele_freq_grid(n_grid: int):
    """Regular grid on [0,1]²."""
    f = np.linspace(0.0, 1.0, n_grid)
    f1, f2 = np.meshgrid(f, f, indexing="ij")
    return f1, f2


# ---------------------------------------------------------------------------
# Plotting (same as ExAdaptiveLandscape n=2)
# ---------------------------------------------------------------------------

def plot_adaptive_landscape_allele_freq(
    f1: np.ndarray,
    f2: np.ndarray,
    mean_fit: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = "viridis",
    D: float | None = None,
):
    """2D contour of mean fitness over (f1, f2). When D≠0, white = invalid (Lewontin bounds)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    valid = np.isfinite(mean_fit)
    vmin, vmax = np.nanmin(mean_fit), np.nanmax(mean_fit)
    levels = np.linspace(vmin, vmax, 60)
    cf = ax.contourf(f1, f2, np.where(valid, mean_fit, np.nan), levels=levels, cmap=cmap, extend="both")
    ax.contour(f1, f2, mean_fit, levels=levels[::6], colors="k", linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="mean fitness W̄")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("f₁ (allele 1 freq. locus 1)")
    ax.set_ylabel("f₂ (allele 1 freq. locus 2)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if D is not None and D != 0:
        ax.text(0.02, 0.02, "White = invalid (f₁,f₂) for this D\n(Lewontin bounds)", transform=ax.transAxes, fontsize=9, va="bottom")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_allele_freq(
    f1: np.ndarray,
    f2: np.ndarray,
    mean_fit_left: np.ndarray,
    mean_fit_right: np.ndarray,
    save_path: str,
    s1: float,
    s2: float,
    left_title: str,
    right_title: str,
    right_has_LD: bool = False,
):
    """Side-by-side 2D contours. If right_has_LD, add note about white = invalid region."""
    vmin = min(np.nanmin(mean_fit_left), np.nanmin(mean_fit_right))
    vmax = max(np.nanmax(mean_fit_left), np.nanmax(mean_fit_right))
    levels = np.linspace(vmin, vmax, 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cf1 = ax1.contourf(f1, f2, mean_fit_left, levels=levels, cmap="viridis", extend="both")
    ax1.contour(f1, f2, mean_fit_left, levels=levels[::6], colors="k", linewidths=0.3, alpha=0.5)
    ax1.set_aspect("equal")
    ax1.set_title(left_title)
    ax1.set_xlabel("f₁ (allele 1 freq. locus 1)")
    ax1.set_ylabel("f₂ (allele 1 freq. locus 2)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    cf2 = ax2.contourf(f1, f2, mean_fit_right, levels=levels, cmap="viridis", extend="both")
    ax2.contour(f1, f2, mean_fit_right, levels=levels[::6], colors="k", linewidths=0.3, alpha=0.5)
    ax2.set_aspect("equal")
    ax2.set_title(right_title)
    ax2.set_xlabel("f₁ (allele 1 freq. locus 1)")
    ax2.set_ylabel("f₂ (allele 1 freq. locus 2)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    if right_has_LD:
        ax2.text(0.02, 0.02, "White = invalid (f₁,f₂)\nfor this D", transform=ax2.transAxes, fontsize=8, va="bottom")

    fig.suptitle(f"s₁={_f(s1)}, s₂={_f(s2)}", fontsize=12, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(cf2, cax=cbar_ax, label="mean fitness W̄")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_adaptive_landscape_3d(
    f1: np.ndarray,
    f2: np.ndarray,
    mean_fit: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = "viridis",
):
    """3D surface: f1, f2 on XY, mean fitness as Z."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    mean_fit_plot = np.where(np.isfinite(mean_fit), mean_fit, np.nan)
    surf = ax.plot_surface(f1, f2, mean_fit_plot, cmap=cmap, alpha=0.9, antialiased=True)
    ax.set_xlabel("f₁ (allele 1 freq. locus 1)")
    ax.set_ylabel("f₂ (allele 1 freq. locus 2)")
    ax.set_zlabel("mean fitness W̄")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label="mean fitness W̄")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_3d(
    f1: np.ndarray,
    f2: np.ndarray,
    mean_fit_left: np.ndarray,
    mean_fit_right: np.ndarray,
    save_path: str,
    s1: float,
    s2: float,
    left_title: str,
    right_title: str,
):
    """Side-by-side 3D surfaces."""
    vmin = min(np.nanmin(mean_fit_left), np.nanmin(mean_fit_right))
    vmax = max(np.nanmax(mean_fit_left), np.nanmax(mean_fit_right))

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    m1 = np.where(np.isfinite(mean_fit_left), mean_fit_left, np.nan)
    m2 = np.where(np.isfinite(mean_fit_right), mean_fit_right, np.nan)
    surf1 = ax1.plot_surface(f1, f2, m1, cmap="viridis", alpha=0.9, vmin=vmin, vmax=vmax)
    ax1.set_xlabel("f₁")
    ax1.set_ylabel("f₂")
    ax1.set_zlabel("W̄")
    ax1.set_title(left_title)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    surf2 = ax2.plot_surface(f1, f2, m2, cmap="viridis", alpha=0.9, vmin=vmin, vmax=vmax)
    ax2.set_xlabel("f₁")
    ax2.set_ylabel("f₂")
    ax2.set_zlabel("W̄")
    ax2.set_title(right_title)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle(f"s₁={_f(s1)}, s₂={_f(s2)}", fontsize=12, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(surf2, cax=cbar_ax, label="mean fitness W̄")
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    s1 = args.s1
    s2 = args.s2
    eps_12 = args.eps_12
    D = args.D
    n_grid = args.n_grid
    script_dir = os.path.dirname(os.path.abspath(__file__))

    f1, f2 = make_allele_freq_grid(n_grid)

    if D == 0:
        p = genotype_freqs_LE(f1, f2)
    else:
        p = genotype_freqs_with_LD(f1, f2, D)

    w_no_epi = fitness_n2(s1, s2, eps_12=0.0)
    w_epi = fitness_n2(s1, s2, eps_12=eps_12)

    mean_fit_no_epi = np.einsum("...g,g->...", p, w_no_epi)
    mean_fit_epi = np.einsum("...g,g->...", p, w_epi)

    ld_str = f"_D{_f(D)}" if D != 0 else ""
    epi_str = f"_eps{_f(eps_12)}" if eps_12 != 0 else ""

    # Single 2D plots
    plot_adaptive_landscape_allele_freq(
        f1, f2, mean_fit_no_epi,
        title=f"Adaptive landscape (no epistasis, D={_f(D)})\ns₁={_f(s1)}, s₂={_f(s2)}",
        save_path=os.path.join(script_dir, f"adaptive_landscape_2loci_no_epistasis{ld_str}.png"),
        D=D if D != 0 else None,
    )
    plot_adaptive_landscape_allele_freq(
        f1, f2, mean_fit_epi,
        title=f"Adaptive landscape (epistasis eps={_f(eps_12)}, D={_f(D)})\ns₁={_f(s1)}, s₂={_f(s2)}",
        save_path=os.path.join(script_dir, f"adaptive_landscape_2loci_epistasis{epi_str}{ld_str}.png"),
        D=D if D != 0 else None,
    )

    # Comparison: no epistasis vs epistasis
    plot_comparison_allele_freq(
        f1, f2, mean_fit_no_epi, mean_fit_epi,
        save_path=os.path.join(script_dir, f"adaptive_landscape_2loci_comparison{epi_str}{ld_str}.png"),
        s1=s1, s2=s2,
        left_title=f"No epistasis (eps₁₂=0)\n→ single peak",
        right_title=f"With epistasis (eps₁₂={_f(eps_12)})\n→ multiple peaks",
        right_has_LD=(D != 0),
    )

    # 3D plots
    plot_adaptive_landscape_3d(
        f1, f2, mean_fit_no_epi,
        title=f"3D (no epistasis, D={_f(D)}) s₁={_f(s1)}, s₂={_f(s2)}",
        save_path=os.path.join(script_dir, f"adaptive_landscape_2loci_3d_no_epistasis{ld_str}.png"),
    )
    plot_adaptive_landscape_3d(
        f1, f2, mean_fit_epi,
        title=f"3D (epistasis eps={_f(eps_12)}, D={_f(D)}) s₁={_f(s1)}, s₂={_f(s2)}",
        save_path=os.path.join(script_dir, f"adaptive_landscape_2loci_3d_epistasis{epi_str}{ld_str}.png"),
    )
    plot_comparison_3d(
        f1, f2, mean_fit_no_epi, mean_fit_epi,
        save_path=os.path.join(script_dir, f"adaptive_landscape_2loci_3d_comparison{epi_str}{ld_str}.png"),
        s1=s1, s2=s2,
        left_title=f"No epistasis, eps₁₂=0\n→ single peak",
        right_title=f"With epistasis, eps₁₂={_f(eps_12)}\n→ multiple peaks",
    )

    # If D != 0, also produce D=0 vs D=X comparison
    if D != 0:
        p_LE = genotype_freqs_LE(f1, f2)
        mean_fit_LE_no_epi = np.einsum("...g,g->...", p_LE, w_no_epi)
        mean_fit_LE_epi = np.einsum("...g,g->...", p_LE, w_epi)

        plot_comparison_allele_freq(
            f1, f2, mean_fit_LE_no_epi, mean_fit_no_epi,
            save_path=os.path.join(script_dir, f"adaptive_landscape_2loci_D0_vs_D{_f(D)}_no_epistasis.png"),
            s1=s1, s2=s2,
            left_title=f"D=0 (linkage equilibrium)\nno epistasis",
            right_title=f"D={_f(D)} (with LD)\nno epistasis",
            right_has_LD=True,
        )
        plot_comparison_allele_freq(
            f1, f2, mean_fit_LE_epi, mean_fit_epi,
            save_path=os.path.join(script_dir, f"adaptive_landscape_2loci_D0_vs_D{_f(D)}_epistasis.png"),
            s1=s1, s2=s2,
            left_title=f"D=0 (linkage equilibrium)\neps₁₂={_f(eps_12)}",
            right_title=f"D={_f(D)} (with LD)\neps₁₂={_f(eps_12)}",
            right_has_LD=True,
        )

    print(f"Plots saved to {script_dir}/")
    print(f"Parameters: s₁={s1}, s₂={s2}, eps₁₂={eps_12}, D={D}")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive landscape for 2 loci with epistasis and LD."
    )
    parser.add_argument("--s1", type=float, default=0.15, help="Selection coefficient locus 1.")
    parser.add_argument("--s2", type=float, default=0.10, help="Selection coefficient locus 2.")
    parser.add_argument(
        "--eps-12",
        type=float,
        default=-0.2,
        help="Epistasis for genotype [1,1]. Negative for multiple peaks.",
    )
    parser.add_argument(
        "--D",
        type=float,
        default=0.0,
        help="Linkage disequilibrium D = p11 - f1*f2. 0 = linkage equilibrium.",
    )
    parser.add_argument("--n-grid", type=int, default=80, help="Grid resolution.")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()


"""
EXAMPLE USAGE
=============

Run from the command line (from the script directory):

1) Linkage equilibrium (D=0), default parameters — full [0,1]² domain:
   python ExAdaptiveLandscape2Loci.py

2) With epistasis, single peak vs multiple peaks comparison:
   python ExAdaptiveLandscape2Loci.py --eps-12 -0.2 --s1 0.15 --s2 0.1

3) With linkage disequilibrium (D≠0) — note the restricted domain:
   python ExAdaptiveLandscape2Loci.py --D 0.1 --eps-12 -0.2
   The plot will show a diamond-shaped colored region; white = invalid (f₁,f₂) for that D
   because genotype frequencies would be negative there (Lewontin bounds).

4) Compare D=0 vs D=0.1 (generated automatically when --D is nonzero):
   python ExAdaptiveLandscape2Loci.py --D 0.1
   Produces adaptive_landscape_2loci_D0_vs_D0.1_*.png

5) Finer grid:
   python ExAdaptiveLandscape2Loci.py --n-grid 120

WHY THE DOMAIN SHRINKS WHEN D≠0
-------------------------------
For fixed D, only (f₁,f₂) where D is within Lewontin bounds yield valid genotype freqs.
Valid region: D ∈ [max(-f₁f₂,-(1-f₁)(1-f₂)), min(f₁(1-f₂),(1-f₁)f₂)].
So the colored region is the feasible (f₁,f₂) for that D — not the full square.
"""
"""
WHY FIXING D (LINKAGE DISEQUILIBRIUM) RESTRICTS WHICH MARGINALS ARE POSSIBLE
----------------------------------------------------------------------------

Consider two biallelic loci:

    p_A = P(A)
    p_B = P(B)

Let the joint haplotype frequency be:

    p_AB = P(A and B)

Linkage disequilibrium is defined as:

    D = p_AB - p_A p_B

If we fix D, then automatically:

    p_AB = p_A p_B + D

However, p_AB must be a valid probability. This means all haplotype
frequencies must be nonnegative:

    p_AB >= 0
    p_Ab = p_A - p_AB >= 0
    p_aB = p_B - p_AB >= 0
    p_ab = 1 - p_A - p_B + p_AB >= 0

From these constraints we obtain the Fréchet bounds:

    max(0, p_A + p_B - 1) <= p_AB <= min(p_A, p_B)

Since p_AB = p_A p_B + D, we must have:

    max(0, p_A + p_B - 1)
        <= p_A p_B + D
        <= min(p_A, p_B)

This shows that for a fixed nonzero D, not all marginal pairs (p_A, p_B)
are feasible.

Intuition:
----------
Fixing D > 0 forces an "excess" of AB haplotypes relative to independence.
But if p_A or p_B is small, there are simply not enough A or B alleles
available to support that excess. Similarly, near the boundaries of the
unit square, some haplotype frequencies would become negative.

Therefore:

- If D is allowed to vary, every (p_A, p_B) in [0,1]^2 is possible.
- If D is fixed, only a restricted region of (p_A, p_B) values is feasible.
- The admissible range of D depends on the marginals.

General principle:
------------------
Fixing correlations (like D) slices the simplex of all joint distributions
by additional linear constraints. Near the boundaries of allele frequency
space, these constraints can become incompatible with nonnegativity of
genotype probabilities.

This phenomenon generalizes to multiple loci: fixing higher-order LD terms
restricts which marginal allele frequencies are compatible with a valid
joint distribution.
"""