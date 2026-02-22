"""
ExAdaptiveLandscape.py — Adaptive landscape over the genotype simplex

For n loci, we have 2^n genotypes. Under linkage equilibrium, allele frequencies
(f_1, ..., f_n) ∈ [0,1]^n determine genotype frequencies.

Fitness model (pairwise epistasis only):
  - s_i: selection coefficient at locus i
  - W(g) = prod_i (1 + s_i * g_i)  [multiplicative base]
          + sum_{i<j} eps_ij * g_i * g_j  [pairwise epistasis]

For n>2: slice-based plotting. Fix n-2 allele freqs, vary 2 → 2D contour/3D surface.

Usage:
  n=2: full (f1,f2) landscape (default)
  n>2: slice over (f_i, f_j) with others fixed at --fix-at (default 0.5)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def _f(x) -> float:
    """Convert to Python float for display (avoids np.float64 in titles)."""
    return float(x)


# ---------------------------------------------------------------------------
# Genotype ordering for n loci: [0,0], [0,1], [1,0], [1,1] for n=2
# ---------------------------------------------------------------------------

def genotype_index(genotype: tuple, n: int) -> int:
    """Map genotype tuple (e.g. (0,0), (1,1)) to index 0..2^n-1."""
    return sum(b << (n - 1 - i) for i, b in enumerate(genotype))


# ---------------------------------------------------------------------------
# Simplex grid: lattice points with denominator N
# For n=2: 4 genotypes → 3-simplex. Points (p0,p1,p2,p3) with sum=1.
# Number of lattice points = C(N+3, 3) < (N+1)^3
# ---------------------------------------------------------------------------

def make_simplex_grid_n2(N: int):
    """
    Build the 3-simplex grid for n=2 loci.
    Returns arrays: p00, p01, p10, p11 (frequencies), and flat arrays for plotting.
    """
    points = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            for k in range(N + 1 - i - j):
                l = N - i - j - k
                p00 = i / N
                p01 = j / N
                p10 = k / N
                p11 = l / N
                points.append((p00, p01, p10, p11))
    return np.array(points)


# ---------------------------------------------------------------------------
# Allele frequency view (linkage equilibrium): smooth 2D on [0,1]²
# f1 = P(allele 1 at locus 1) = p10 + p11,  f2 = P(allele 1 at locus 2) = p01 + p11
# Under LE: p00=(1-f1)(1-f2), p01=(1-f1)f2, p10=f1(1-f2), p11=f1*f2
# ---------------------------------------------------------------------------

def genotype_freqs_from_allele_freqs(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Under linkage equilibrium, return (p00, p01, p10, p11) from (f1, f2)."""
    p00 = (1 - f1) * (1 - f2)
    p01 = (1 - f1) * f2
    p10 = f1 * (1 - f2)
    p11 = f1 * f2
    return np.stack([p00, p01, p10, p11], axis=-1)


def make_allele_freq_grid(n_grid: int):
    """Regular grid on [0,1]² for allele frequencies. Returns f1, f2 as 2D arrays."""
    f = np.linspace(0.0, 1.0, n_grid)
    f1, f2 = np.meshgrid(f, f, indexing="ij")
    return f1, f2


def make_slice_grid(
    n: int,
    slice_axes: tuple[int, int],
    fix_at: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2D slice of [0,1]^n for allele frequencies.
    slice_axes: (i, j) — the two dimensions to vary.
    fix_at: value for all other dimensions.
    Returns: f_axis1, f_axis2 (2D arrays), and f_full (n_grid, n_grid, n).
    """
    f = np.linspace(0.0, 1.0, n_grid)
    fa, fb = np.meshgrid(f, f, indexing="ij")
    f_full = np.full((n_grid, n_grid, n), fix_at, dtype=float)
    f_full[..., slice_axes[0]] = fa
    f_full[..., slice_axes[1]] = fb
    return fa, fb, f_full


def make_multi_slice_grids(
    n: int,
    slice_axes: tuple[int, int],
    fixed_dims: list[int],
    slice_values: np.ndarray,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[tuple]]:
    """
    Build multiple 2D slices for multi-panel plotting.
    fixed_dims: indices of dimensions to vary over slice_values.
    For n=3: fixed_dims=[2], slice_values=[0,0.25,0.5,0.75,1] → 5 slices.
    For n=4: fixed_dims=[2,3], slice_values=[0,0.25,0.5,0.75,1] → 5×5 = 25 slices.
    Returns: fa, fb (same for all), list of f_full arrays, list of (fix_val1, fix_val2, ...) per slice.
    """
    f = np.linspace(0.0, 1.0, n_grid)
    fa, fb = np.meshgrid(f, f, indexing="ij")

    if len(fixed_dims) == 1:
        # n=3: 5 slices in a row
        f_full_list = []
        fix_tuples = []
        for v in slice_values:
            f_full = np.full((n_grid, n_grid, n), 0.5, dtype=float)  # default others at 0.5
            f_full[..., slice_axes[0]] = fa
            f_full[..., slice_axes[1]] = fb
            f_full[..., fixed_dims[0]] = v
            f_full_list.append(f_full)
            fix_tuples.append((v,))
        return fa, fb, f_full_list, fix_tuples

    # n=4: 5×5 matrix, fixed_dims has 2 elements
    f_full_list = []
    fix_tuples = []
    for v1 in slice_values:
        for v2 in slice_values:
            f_full = np.full((n_grid, n_grid, n), 0.5, dtype=float)
            f_full[..., slice_axes[0]] = fa
            f_full[..., slice_axes[1]] = fb
            f_full[..., fixed_dims[0]] = v1
            f_full[..., fixed_dims[1]] = v2
            f_full_list.append(f_full)
            fix_tuples.append((v1, v2))
    return fa, fb, f_full_list, fix_tuples


def tetrahedron_to_2d(p: np.ndarray) -> np.ndarray:
    """
    Project 4D point on 3-simplex to 2D (standard tetrahedron projection).
    p has shape (..., 4) with p[..., :].sum(axis=-1) = 1.
    Returns (X, Y) with shape (..., 2) or two arrays of shape (...).
    """
    p = np.asarray(p)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    # Vertices of regular tetrahedron in 2D:
    # (1,0,0,0)->(0,0), (0,1,0,0)->(1,0), (0,0,1,0)->(0.5,√3/2), (0,0,0,1)->(0.5,√3/6)
    sqrt3 = np.sqrt(3)
    X = p[:, 1] + 0.5 * p[:, 2] + 0.5 * p[:, 3]
    Y = (sqrt3 / 2) * p[:, 2] + (sqrt3 / 6) * p[:, 3]
    return np.column_stack([X, Y])


# ---------------------------------------------------------------------------
# Fitness and mean fitness
# ---------------------------------------------------------------------------

def fitness_n2(s1: float, s2: float, eps_12: float = 0.0) -> np.ndarray:
    """
    Fitness of each genotype for n=2.
    Order: [0,0], [0,1], [1,0], [1,1]
    W([0,0])=1, W([0,1])=1+s2, W([1,0])=1+s1,
    W([1,1]) = (1+s1)(1+s2) + eps_12
    """
    w = np.array([
        1.0,
        1.0 + s2,
        1.0 + s1,
        (1.0 + s1) * (1.0 + s2) + eps_12,
    ])
    return w


def fitness_ngeneral(s: np.ndarray, eps_matrix: np.ndarray) -> np.ndarray:
    """
    Fitness for n loci with pairwise epistasis.
    s: (n,) selection coefficients. eps_matrix: (n,n) symmetric, eps_ij for pair (i,j).
    Genotype index: g=0..2^n-1, bit i = allele at locus i.
    W(g) = prod_i (1 + s_i * g_i) + sum_{i<j} eps_ij * g_i * g_j
    """
    n = len(s)
    n_geno = 1 << n
    w = np.zeros(n_geno)
    for g in range(n_geno):
        geno = [(g >> (n - 1 - i)) & 1 for i in range(n)]  # genotype as 0/1 list
        mult = 1.0
        for i in range(n):
            mult *= 1.0 + s[i] * geno[i]
        epistasis = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                epistasis += eps_matrix[i, j] * geno[i] * geno[j]
        w[g] = mult + epistasis
    return w


def genotype_probs_from_allele_freqs(f: np.ndarray) -> np.ndarray:
    """
    Under linkage equilibrium: P(g) = prod_i [ f_i^g_i * (1-f_i)^(1-g_i) ].
    f: (..., n) allele freqs. Returns (..., 2^n) genotype probs.
    """
    n = f.shape[-1]
    n_geno = 1 << n
    shape = f.shape[:-1] + (n_geno,)
    p = np.ones(shape)
    for g in range(n_geno):
        for i in range(n):
            gi = (g >> (n - 1 - i)) & 1
            p[..., g] *= np.where(gi, f[..., i], 1.0 - f[..., i])
    return p


def genotype_freqs_from_allele_freqs(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Under linkage equilibrium, return (p00, p01, p10, p11) from (f1, f2). [n=2 only]"""
    p00 = (1 - f1) * (1 - f2)
    p01 = (1 - f1) * f2
    p10 = f1 * (1 - f2)
    p11 = f1 * f2
    return np.stack([p00, p01, p10, p11], axis=-1)


def mean_fitness(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mean fitness = E[W] = sum_g p_g * W_g."""
    return np.dot(p, w)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_adaptive_landscape_allele_freq(
    f1: np.ndarray,
    f2: np.ndarray,
    mean_fit: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = "viridis",
):
    """
    Plot mean fitness over allele frequencies (f1, f2) on a regular grid.
    Smooth, continuous gradients — ideal for seeing single vs multiple peaks.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    levels = np.linspace(mean_fit.min(), mean_fit.max(), 60)
    cf = ax.contourf(f1, f2, mean_fit, levels=levels, cmap=cmap)
    ax.contour(f1, f2, mean_fit, levels=levels[::6], colors="k", linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="mean fitness W̄")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("f₁ (allele 1 freq. at locus 1)")
    ax.set_ylabel("f₂ (allele 1 freq. at locus 2)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_adaptive_landscape_simplex(
    points: np.ndarray,
    mean_fit: np.ndarray,
    title: str,
    save_path: str,
    cmap: str = "viridis",
):
    """
    Plot mean fitness over the flattened 2D simplex using triangulation.
    points: (N_pts, 4) simplex coordinates
    mean_fit: (N_pts,) mean fitness at each point
    """
    xy = tetrahedron_to_2d(points)
    x, y = xy[:, 0], xy[:, 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    levels = np.linspace(mean_fit.min(), mean_fit.max(), 50)
    tcf = ax.tricontourf(x, y, mean_fit, levels=levels, cmap=cmap)
    ax.tricontour(x, y, mean_fit, levels=levels[::5], colors="k", linewidths=0.3, alpha=0.4)
    plt.colorbar(tcf, ax=ax, label="mean fitness W̄")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("(projected simplex)")
    ax.set_ylabel("(projected simplex)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_allele_freq(
    f1: np.ndarray,
    f2: np.ndarray,
    mean_fit_no_epi: np.ndarray,
    mean_fit_epi: np.ndarray,
    save_path: str,
    s1: float,
    s2: float,
    eps_12: float,
):
    """
    Side-by-side: no epistasis vs with epistasis (allele freq view).

    The contour lines (grey/black) are isolines: they connect points with the SAME
    mean fitness W̄. Like a topo map: each line = constant height.
    - Single peak: concentric rings around one maximum (one "mountain").
    - Multiple peaks: separate rings around each local maximum (several "mountains").
    Use negative eps_12 (e.g. -0.2) to see multiple peaks.
    """
    vmin = min(mean_fit_no_epi.min(), mean_fit_epi.min())
    vmax = max(mean_fit_no_epi.max(), mean_fit_epi.max())
    levels = np.linspace(vmin, vmax, 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cf1 = ax1.contourf(f1, f2, mean_fit_no_epi, levels=levels, cmap="viridis")
    ax1.contour(f1, f2, mean_fit_no_epi, levels=levels[::6], colors="k", linewidths=0.3, alpha=0.5)
    ax1.set_aspect("equal")
    ax1.set_title(f"No epistasis, eps₁₂=0\n→ single peak")
    ax1.set_xlabel("f₁ (allele 1 freq. locus 1)")
    ax1.set_ylabel("f₂ (allele 1 freq. locus 2)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    cf2 = ax2.contourf(f1, f2, mean_fit_epi, levels=levels, cmap="viridis")
    ax2.contour(f1, f2, mean_fit_epi, levels=levels[::6], colors="k", linewidths=0.3, alpha=0.5)
    ax2.set_aspect("equal")
    ax2.set_title(f"With epistasis, eps₁₂={_f(eps_12)}\n→ multiple peaks")
    ax2.set_xlabel("f₁ (allele 1 freq. locus 1)")
    ax2.set_ylabel("f₂ (allele 1 freq. locus 2)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle(f"s₁={_f(s1)}, s₂={_f(s2)}", fontsize=12, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # placed to the right of both plots
    fig.colorbar(cf2, cax=cbar_ax, label="mean fitness W̄")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(
    points: np.ndarray,
    mean_fit_no_epi: np.ndarray,
    mean_fit_epi: np.ndarray,
    save_path: str,
    s1: float,
    s2: float,
    eps_12: float,
):
    """Side-by-side: no epistasis vs with epistasis (simplex view)."""
    xy = tetrahedron_to_2d(points)
    x, y = xy[:, 0], xy[:, 1]

    vmin = min(mean_fit_no_epi.min(), mean_fit_epi.min())
    vmax = max(mean_fit_no_epi.max(), mean_fit_epi.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    levels = np.linspace(vmin, vmax, 50)

    tcf1 = ax1.tricontourf(x, y, mean_fit_no_epi, levels=levels, cmap="viridis")
    ax1.tricontour(x, y, mean_fit_no_epi, levels=levels[::5], colors="k", linewidths=0.2, alpha=0.3)
    ax1.set_aspect("equal")
    ax1.set_title(f"No epistasis, eps₁₂=0\n→ single peak")
    ax1.set_xlabel("(projected simplex)")

    tcf2 = ax2.tricontourf(x, y, mean_fit_epi, levels=levels, cmap="viridis")
    ax2.tricontour(x, y, mean_fit_epi, levels=levels[::5], colors="k", linewidths=0.2, alpha=0.3)
    ax2.set_aspect("equal")
    ax2.set_title(f"With epistasis, eps₁₂={_f(eps_12)}\n→ multiple peaks")
    ax2.set_xlabel("(projected simplex)")

    fig.suptitle(f"s₁={_f(s1)}, s₂={_f(s2)}", fontsize=12, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(tcf2, cax=cbar_ax, label="mean fitness W̄")
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
    """3D surface: f1, f2 on XY plane, mean fitness W̄ as Z height."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(f1, f2, mean_fit, cmap=cmap, alpha=0.9, antialiased=True)
    ax.set_xlabel("f₁ (allele 1 freq. locus 1)")
    ax.set_ylabel("f₂ (allele 1 freq. locus 2)")
    ax.set_zlabel("mean fitness W̄")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label="mean fitness W̄")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_slice_allele_freq(
    fa: np.ndarray,
    fb: np.ndarray,
    mean_fit: np.ndarray,
    title: str,
    save_path: str,
    axis_i: int,
    axis_j: int,
    fix_at: float,
    n: int,
    cmap: str = "viridis",
):
    """2D contour of mean fitness over slice (f_i, f_j), others fixed at fix_at."""
    fig, ax = plt.subplots(figsize=(8, 7))
    levels = np.linspace(mean_fit.min(), mean_fit.max(), 60)
    cf = ax.contourf(fa, fb, mean_fit, levels=levels, cmap=cmap)
    ax.contour(fa, fb, mean_fit, levels=levels[::6], colors="k", linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="mean fitness W̄")
    ax.set_aspect("equal")
    ax.set_title(title)
    fixed_dims = [k for k in range(n) if k not in (axis_i, axis_j)]
    fix_str = ", ".join(f"locus {k+1}={_f(fix_at)}" for k in fixed_dims) if fixed_dims else ""
    ax.set_xlabel(f"Locus {axis_i + 1} freq (f₁)")
    ax.set_ylabel(f"Locus {axis_j + 1} freq (f₂)")
    if fix_str:
        ax.text(0.02, 0.98, f"Fixed: {fix_str}", transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_slice_comparison(
    fa: np.ndarray,
    fb: np.ndarray,
    mean_fit_no_epi: np.ndarray,
    mean_fit_epi: np.ndarray,
    save_path: str,
    s: np.ndarray,
    eps_pairwise: float,
    axis_i: int,
    axis_j: int,
    fix_at: float,
    n: int,
):
    """Side-by-side slice: no epistasis vs with pairwise epistasis."""
    vmin = min(mean_fit_no_epi.min(), mean_fit_epi.min())
    vmax = max(mean_fit_no_epi.max(), mean_fit_epi.max())
    levels = np.linspace(vmin, vmax, 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cf1 = ax1.contourf(fa, fb, mean_fit_no_epi, levels=levels, cmap="viridis")
    ax1.contour(fa, fb, mean_fit_no_epi, levels=levels[::6], colors="k", linewidths=0.3, alpha=0.5)
    ax1.set_aspect("equal")
    ax1.set_title(f"No epistasis (all eps=0)\n→ single peak")
    ax1.set_xlabel(f"Locus {axis_i + 1} freq")
    ax1.set_ylabel(f"Locus {axis_j + 1} freq")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    cf2 = ax2.contourf(fa, fb, mean_fit_epi, levels=levels, cmap="viridis")
    ax2.contour(fa, fb, mean_fit_epi, levels=levels[::6], colors="k", linewidths=0.3, alpha=0.5)
    ax2.set_aspect("equal")
    ax2.set_title(f"Pairwise epistasis eps={_f(eps_pairwise)}\n→ multiple peaks")
    ax2.set_xlabel(f"Locus {axis_i + 1} freq")
    ax2.set_ylabel(f"Locus {axis_j + 1} freq")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fixed_dims = [k for k in range(n) if k not in (axis_i, axis_j)]
    fix_str = ", ".join(f"locus {k+1}={_f(fix_at)}" for k in fixed_dims) if fixed_dims else ""
    s_str = ",".join(f"{_f(x):.2g}" for x in s)
    fig.suptitle(f"n={n} loci, s=({s_str}), slice (f₁,f₂), {fix_str}", fontsize=11, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(cf2, cax=cbar_ax, label="mean fitness W̄")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_multi_slice_row(
    fa: np.ndarray,
    fb: np.ndarray,
    mean_fit_list: list[np.ndarray],
    fix_tuples: list[tuple],
    save_path: str,
    n: int,
    axis_i: int,
    axis_j: int,
    fixed_dim: int,
    with_epistasis: bool,
    eps_pairwise: float = 0.0,
    s: np.ndarray | None = None,
):
    """5 slices in a row (n=3): W̄(f₁,f₂) at f₃ = 0, 0.25, 0.5, 0.75, 1."""
    n_panels = len(mean_fit_list)
    vmin = min(m.min() for m in mean_fit_list)
    vmax = max(m.max() for m in mean_fit_list)
    levels = np.linspace(vmin, vmax, 50)

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    for ax, mean_fit, ft in zip(axes, mean_fit_list, fix_tuples):
        cf = ax.contourf(fa, fb, mean_fit, levels=levels, cmap="viridis")
        ax.contour(fa, fb, mean_fit, levels=levels[::5], colors="k", linewidths=0.3, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"Locus {fixed_dim + 1} freq = {_f(ft[0]):.2f}")
        ax.set_xlabel(f"Locus {axis_i + 1} freq (f₁)")
        ax.set_ylabel(f"Locus {axis_j + 1} freq (f₂)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    epi_str = f", eps={_f(eps_pairwise)}" if with_epistasis else ", eps=0"
    fig.suptitle(f"n={n} loci: W̄(f₁,f₂) with locus {fixed_dim+1} freq shown per tile{epi_str}", fontsize=11, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label="mean fitness W̄")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_multi_slice_matrix(
    fa: np.ndarray,
    fb: np.ndarray,
    mean_fit_list: list[np.ndarray],
    fix_tuples: list[tuple],
    save_path: str,
    n: int,
    axis_i: int,
    axis_j: int,
    fixed_dims: tuple[int, int],
    with_epistasis: bool,
    eps_pairwise: float = 0.0,
):
    """5×5 matrix of slices (n=4): W̄(f₁,f₂) at (f₃,f₄) ∈ {0,0.25,0.5,0.75,1}²."""
    n_slice = 5
    vmin = min(m.min() for m in mean_fit_list)
    vmax = max(m.max() for m in mean_fit_list)
    levels = np.linspace(vmin, vmax, 50)

    fig, axes = plt.subplots(n_slice, n_slice, figsize=(4 * n_slice, 4 * n_slice))
    for idx, (ax, mean_fit, ft) in enumerate(zip(axes.flat, mean_fit_list, fix_tuples)):
        cf = ax.contourf(fa, fb, mean_fit, levels=levels, cmap="viridis")
        ax.contour(fa, fb, mean_fit, levels=levels[::5], colors="k", linewidths=0.2, alpha=0.4)
        ax.set_aspect("equal")
        ax.set_title(f"Locus {fixed_dims[0]+1}={_f(ft[0]):.2f}, Locus {fixed_dims[1]+1}={_f(ft[1]):.2f}", fontsize=9)
        ax.set_xlabel(f"Locus {axis_i+1}")
        ax.set_ylabel(f"Locus {axis_j+1}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    epi_str = f", eps={_f(eps_pairwise)}" if with_epistasis else ", eps=0"
    fig.suptitle(f"n={n} loci: W̄(f₁,f₂), loci {fixed_dims[0]+1} and {fixed_dims[1]+1} shown per tile{epi_str}", fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label="mean fitness W̄")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_slice_3d(
    fa: np.ndarray,
    fb: np.ndarray,
    mean_fit: np.ndarray,
    title: str,
    save_path: str,
    axis_i: int,
    axis_j: int,
    cmap: str = "viridis",
):
    """3D surface of slice."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(fa, fb, mean_fit, cmap=cmap, alpha=0.9, antialiased=True)
    ax.set_xlabel(f"f_{axis_i + 1}")
    ax.set_ylabel(f"f_{axis_j + 1}")
    ax.set_zlabel("mean fitness W̄")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, label="mean fitness W̄")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_3d(
    f1: np.ndarray,
    f2: np.ndarray,
    mean_fit_no_epi: np.ndarray,
    mean_fit_epi: np.ndarray,
    save_path: str,
    s1: float,
    s2: float,
    eps_12: float,
):
    """Side-by-side 3D surfaces: no epistasis vs with epistasis."""
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    vmin = min(mean_fit_no_epi.min(), mean_fit_epi.min())
    vmax = max(mean_fit_no_epi.max(), mean_fit_epi.max())

    surf1 = ax1.plot_surface(f1, f2, mean_fit_no_epi, cmap="viridis", alpha=0.9, vmin=vmin, vmax=vmax)
    ax1.set_xlabel("f₁")
    ax1.set_ylabel("f₂")
    ax1.set_zlabel("W̄")
    ax1.set_title(f"No epistasis, eps₁₂=0\n→ single peak")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    surf2 = ax2.plot_surface(f1, f2, mean_fit_epi, cmap="viridis", alpha=0.9, vmin=vmin, vmax=vmax)
    ax2.set_xlabel("f₁")
    ax2.set_ylabel("f₂")
    ax2.set_zlabel("W̄")
    ax2.set_title(f"With epistasis, eps₁₂={_f(eps_12)}\n→ multiple peaks")
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
    n_loci = args.n_loci
    N = args.N
    n_grid = args.n_grid
    s = np.array(args.s, dtype=float)
    eps_pairwise = args.eps_pairwise
    projection = args.projection
    slice_axes = tuple(args.slice_axes)
    fix_at = args.fix_at

    if len(s) != n_loci:
        raise ValueError(f"--s must have {n_loci} values, got {len(s)}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build eps matrix (symmetric, pairwise only)
    eps_matrix = np.zeros((n_loci, n_loci))
    for i in range(n_loci):
        for j in range(i + 1, n_loci):
            eps_matrix[i, j] = eps_matrix[j, i] = eps_pairwise

    w_no_epi = fitness_ngeneral(s, np.zeros((n_loci, n_loci)))
    w_epi = fitness_ngeneral(s, eps_matrix)

    if n_loci == 2 and projection == "allele_freq":
        # Original n=2 full landscape
        f1, f2 = make_allele_freq_grid(n_grid)
        p = genotype_freqs_from_allele_freqs(f1, f2)
        mean_fit_no_epi = np.einsum("...g,g->...", p, w_no_epi)
        mean_fit_epi = np.einsum("...g,g->...", p, w_epi)
        print(f"n=2: allele freq view {n_grid}×{n_grid} on [0,1]²")

        plot_adaptive_landscape_allele_freq(
            f1, f2, mean_fit_no_epi,
            title=f"Adaptive landscape (no epistasis)\ns=({','.join(f'{_f(x):.2g}' for x in s)}), eps=0",
            save_path=os.path.join(script_dir, f"adaptive_landscape_no_epistasis_n2_s{'_'.join(map(str,s))}_eps0.png"),
        )
        plot_adaptive_landscape_allele_freq(
            f1, f2, mean_fit_epi,
            title=f"Adaptive landscape (pairwise epistasis)\ns=({','.join(f'{_f(x):.2g}' for x in s)}), eps={_f(eps_pairwise)}",
            save_path=os.path.join(script_dir, f"adaptive_landscape_epistasis_n2_s{'_'.join(map(str,s))}_eps{eps_pairwise}.png"),
        )
        plot_comparison_allele_freq(
            f1, f2, mean_fit_no_epi, mean_fit_epi,
            save_path=os.path.join(script_dir, f"adaptive_landscape_comparison_n2_s{'_'.join(map(str,s))}_eps{eps_pairwise}.png"),
            s1=s[0], s2=s[1], eps_12=eps_pairwise,
        )
        plot_adaptive_landscape_3d(
            f1, f2, mean_fit_no_epi,
            title=f"3D (no epistasis) s=({','.join(f'{_f(x):.2g}' for x in s)})",
            save_path=os.path.join(script_dir, f"adaptive_landscape_3d_no_epistasis_n2_s{'_'.join(map(str,s))}.png"),
        )
        plot_adaptive_landscape_3d(
            f1, f2, mean_fit_epi,
            title=f"3D (pairwise epistasis) s=({','.join(f'{_f(x):.2g}' for x in s)}), eps={_f(eps_pairwise)}",
            save_path=os.path.join(script_dir, f"adaptive_landscape_3d_epistasis_n2_s{'_'.join(map(str,s))}_eps{eps_pairwise}.png"),
        )
        plot_comparison_3d(
            f1, f2, mean_fit_no_epi, mean_fit_epi,
            save_path=os.path.join(script_dir, f"adaptive_landscape_3d_comparison_n2_s{'_'.join(map(str,s))}_eps{eps_pairwise}.png"),
            s1=s[0], s2=s[1], eps_12=eps_pairwise,
        )
    elif n_loci >= 2 and projection == "allele_freq":
        axis_i, axis_j = slice_axes[0], slice_axes[1]
        fixed_dims = [k for k in range(n_loci) if k not in (axis_i, axis_j)]
        slice_values = np.linspace(0.0, 1.0, args.n_slice_values)

        if args.multi_slice and n_loci == 3:
            # n=3: 5 slices in a row (f₃ = 0, 0.25, 0.5, 0.75, 1)
            fa, fb, f_full_list, fix_tuples = make_multi_slice_grids(
                n_loci, (axis_i, axis_j), fixed_dims, slice_values, n_grid
            )
            mean_fit_no_epi_list = [genotype_probs_from_allele_freqs(ff) @ w_no_epi for ff in f_full_list]
            mean_fit_epi_list = [genotype_probs_from_allele_freqs(ff) @ w_epi for ff in f_full_list]
            print(f"n=3: 5 slices in a row, f_3 ∈ {{0, 0.25, 0.5, 0.75, 1}}")

            plot_multi_slice_row(
                fa, fb, mean_fit_no_epi_list, fix_tuples,
                save_path=os.path.join(script_dir, f"adaptive_landscape_multi_slice_row_no_epistasis_n3.png"),
                n=n_loci, axis_i=axis_i, axis_j=axis_j, fixed_dim=fixed_dims[0],
                with_epistasis=False,
            )
            plot_multi_slice_row(
                fa, fb, mean_fit_epi_list, fix_tuples,
                save_path=os.path.join(script_dir, f"adaptive_landscape_multi_slice_row_epistasis_n3_eps{eps_pairwise}.png"),
                n=n_loci, axis_i=axis_i, axis_j=axis_j, fixed_dim=fixed_dims[0],
                with_epistasis=True, eps_pairwise=eps_pairwise,
            )
        elif args.multi_slice and n_loci == 4:
            # n=4: 5×5 matrix of slices
            fa, fb, f_full_list, fix_tuples = make_multi_slice_grids(
                n_loci, (axis_i, axis_j), fixed_dims, slice_values, n_grid
            )
            mean_fit_no_epi_list = [genotype_probs_from_allele_freqs(ff) @ w_no_epi for ff in f_full_list]
            mean_fit_epi_list = [genotype_probs_from_allele_freqs(ff) @ w_epi for ff in f_full_list]
            print(f"n=4: 5×5 matrix of slices, (f_3, f_4) ∈ {{0..1}}²")

            plot_multi_slice_matrix(
                fa, fb, mean_fit_no_epi_list, fix_tuples,
                save_path=os.path.join(script_dir, f"adaptive_landscape_multi_slice_matrix_no_epistasis_n4.png"),
                n=n_loci, axis_i=axis_i, axis_j=axis_j, fixed_dims=tuple(fixed_dims),
                with_epistasis=False,
            )
            plot_multi_slice_matrix(
                fa, fb, mean_fit_epi_list, fix_tuples,
                save_path=os.path.join(script_dir, f"adaptive_landscape_multi_slice_matrix_epistasis_n4_eps{eps_pairwise}.png"),
                n=n_loci, axis_i=axis_i, axis_j=axis_j, fixed_dims=tuple(fixed_dims),
                with_epistasis=True, eps_pairwise=eps_pairwise,
            )
        else:
            # Single slice (original behavior)
            fa, fb, f_full = make_slice_grid(n_loci, (axis_i, axis_j), fix_at, n_grid)
            p = genotype_probs_from_allele_freqs(f_full)
            mean_fit_no_epi = p @ w_no_epi
            mean_fit_epi = p @ w_epi
            fixed_str = "_".join(f"f{k}={fix_at}" for k in range(n_loci) if k not in (axis_i, axis_j))
            print(f"n={n_loci}: slice (f_{axis_i+1}, f_{axis_j+1}), others={fix_at}")

            plot_slice_allele_freq(
                fa, fb, mean_fit_no_epi,
                title=f"Slice (f₁,f₂) no epistasis, n={n_loci}, s=({','.join(f'{_f(x):.2g}' for x in s)}), eps=0",
                save_path=os.path.join(script_dir, f"adaptive_landscape_slice_no_epistasis_n{n_loci}_a{axis_i}{axis_j}_{fixed_str}.png"),
                axis_i=axis_i, axis_j=axis_j, fix_at=fix_at, n=n_loci,
            )
            plot_slice_allele_freq(
                fa, fb, mean_fit_epi,
                title=f"Slice (f₁,f₂) pairwise epistasis, n={n_loci}, s=({','.join(f'{_f(x):.2g}' for x in s)}), eps={_f(eps_pairwise)}",
                save_path=os.path.join(script_dir, f"adaptive_landscape_slice_epistasis_n{n_loci}_a{axis_i}{axis_j}_eps{eps_pairwise}.png"),
                axis_i=axis_i, axis_j=axis_j, fix_at=fix_at, n=n_loci,
            )
            plot_slice_comparison(
                fa, fb, mean_fit_no_epi, mean_fit_epi,
                save_path=os.path.join(script_dir, f"adaptive_landscape_slice_comparison_n{n_loci}_a{axis_i}{axis_j}_eps{eps_pairwise}.png"),
                s=s, eps_pairwise=eps_pairwise, axis_i=axis_i, axis_j=axis_j, fix_at=fix_at, n=n_loci,
            )
            plot_slice_3d(
                fa, fb, mean_fit_no_epi,
                title=f"3D slice (f₁,f₂) no epistasis, n={n_loci}",
                save_path=os.path.join(script_dir, f"adaptive_landscape_slice_3d_no_epistasis_n{n_loci}_a{axis_i}{axis_j}.png"),
                axis_i=axis_i, axis_j=axis_j,
            )
            plot_slice_3d(
                fa, fb, mean_fit_epi,
                title=f"3D slice (f₁,f₂) epistasis, n={n_loci}, eps={_f(eps_pairwise)}",
                save_path=os.path.join(script_dir, f"adaptive_landscape_slice_3d_epistasis_n{n_loci}_a{axis_i}{axis_j}.png"),
                axis_i=axis_i, axis_j=axis_j,
            )
    elif n_loci == 2 and projection == "simplex":
        w_no_epi = fitness_n2(s[0], s[1], eps_12=0.0)
        w_epi = fitness_n2(s[0], s[1], eps_12=eps_pairwise)
        points = make_simplex_grid_n2(N)
        mean_fit_no_epi = points @ w_no_epi
        mean_fit_epi = points @ w_epi
        print(f"n=2 simplex view: N={N}")

        plot_adaptive_landscape_simplex(
            points, mean_fit_no_epi,
            title=f"Simplex (no epistasis) s=({','.join(f'{_f(x):.2g}' for x in s)})",
            save_path=os.path.join(script_dir, f"adaptive_landscape_simplex_no_epistasis_n2.png"),
        )
        plot_adaptive_landscape_simplex(
            points, mean_fit_epi,
            title=f"Simplex (epistasis) s=({','.join(f'{_f(x):.2g}' for x in s)}), eps={_f(eps_pairwise)}",
            save_path=os.path.join(script_dir, f"adaptive_landscape_simplex_epistasis_n2.png"),
        )
        plot_comparison(
            points, mean_fit_no_epi, mean_fit_epi,
            save_path=os.path.join(script_dir, f"adaptive_landscape_simplex_comparison_n2.png"),
            s1=s[0], s2=s[1], eps_12=eps_pairwise,
        )
    else:
        raise ValueError("simplex projection only supported for n=2")

    print(f"Plots saved to {script_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive landscape over the genotype simplex (mean fitness). Pairwise epistasis only."
    )
    parser.add_argument("--n-loci", type=int, default=2, help="Number of loci.")
    parser.add_argument(
        "--s",
        type=float,
        nargs="+",
        default=None,
        help="Selection coefficients (one per locus). Default for n=2: 0.15 0.1",
    )
    parser.add_argument(
        "--eps-pairwise",
        type=float,
        default=-0.2,
        help="Pairwise epistasis (same for all pairs). Negative for multiple peaks.",
    )
    parser.add_argument(
        "--projection",
        choices=["allele_freq", "simplex"],
        default="allele_freq",
        help="allele_freq: allele space. simplex: full genotype lattice (n=2 only).",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=80,
        help="Grid resolution (n_grid×n_grid).",
    )
    parser.add_argument(
        "--slice-axes",
        type=int,
        nargs=2,
        default=[0, 1],
        metavar=("I", "J"),
        help="For n>2: which two allele freqs to vary (0-based). Default: 0 1",
    )
    parser.add_argument(
        "--fix-at",
        type=float,
        default=0.5,
        help="For n>2 slice: value for fixed allele freqs. Default: 0.5",
    )
    parser.add_argument(
        "--multi-slice",
        action="store_true",
        default=True,
        help="For n=3: 5 slices in a row. For n=4: 5×5 matrix. (default: True for n>=3)",
    )
    parser.add_argument(
        "--no-multi-slice",
        action="store_true",
        help="Disable multi-slice; use single slice at --fix-at.",
    )
    parser.add_argument(
        "--n-slice-values",
        type=int,
        default=5,
        help="Number of slice values per fixed dim (0..1). Default: 5 → 5 panels (n=3) or 5×5 (n=4).",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=39,
        help="Population size for simplex view (n=2 only).",
    )
    args = parser.parse_args()

    # Default --s for n=2
    if args.s is None:
        args.s = [0.15, 0.10] if args.n_loci == 2 else [0.15] * args.n_loci
    if len(args.s) != args.n_loci:
        parser.error(f"--s must have {args.n_loci} values for --n-loci {args.n_loci}")
    if args.no_multi_slice:
        args.multi_slice = False

    run(args)


if __name__ == "__main__":
    main()
