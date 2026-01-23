"""
1D heat equation on [0,1] with Neumann BCs.

Two solvers:
- Implicit Euler PDE step (sanity check)
- "True JKO" step for entropy in 1D quantile coordinates (Newton + tridiagonal)

This file is a cleaned reference copy of `1D.py` with:
- more careful positivity/mass handling in the Euler step (no unconditional clip/renorm)
- initialization uses `density * dx` (robust if grid changes)
- `jko_objective(..., edges=...)` naming fixed
- `np.trapezoid` instead of `np.trapz`
- a real `main()` + lightweight diagnostics
"""

from __future__ import annotations

import argparse
import numpy as np  # type: ignore

# Use a non-interactive backend to avoid Qt/Wayland issues when saving figures.
import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore


def make_grid(n_cells: int = 200):
    """Uniform cell grid on [0,1]. Returns (centers x, dx, edges)."""
    edges = np.linspace(0.0, 1.0, n_cells + 1)
    dx = edges[1] - edges[0]
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, dx, edges


def entropy(mass: np.ndarray, dx: float, eps: float = 1e-12) -> float:
    """
    Discrete entropy F(ρ)=∫ρ log ρ, with mass-vector m_i ≈ ∫_{cell i} ρ dx.
    On a uniform grid: ρ_i ≈ m_i/dx, so F ≈ Σ m_i log(m_i/dx).
    """
    m = np.asarray(mass, float)
    m = np.maximum(m, eps)
    return float(np.sum(m * (np.log(m) - np.log(dx))))


def quantile_from_mass(mass: np.ndarray, edges: np.ndarray, n_quantiles: int = 512) -> np.ndarray:
    """
    Exact inverse CDF Q(u)=F^{-1}(u) for a piecewise-constant density on cells.
    Evaluated on u-grid INCLUDING endpoints.
    """
    m = np.asarray(mass, float)
    m = m / m.sum()
    n_cells = m.size
    dx = edges[1] - edges[0]

    u = np.linspace(0.0, 1.0, n_quantiles)
    Q = np.empty_like(u)

    cum = np.cumsum(m)
    cum[-1] = 1.0
    cum_prev = np.concatenate([[0.0], cum[:-1]])
    idx = np.searchsorted(cum, u, side="left")
    idx = np.clip(idx, 0, n_cells - 1)

    for j, (uu, i) in enumerate(zip(u, idx)):
        mi = m[i]
        if mi <= 0.0:
            Q[j] = edges[i]
            continue
        frac = (uu - cum_prev[i]) / mi
        frac = float(np.clip(frac, 0.0, 1.0))
        Q[j] = edges[i] + frac * dx

    Q[0] = edges[0]
    Q[-1] = edges[-1]
    return Q


def w2_squared_1d(mass_a: np.ndarray, mass_b: np.ndarray, edges: np.ndarray, n_quantiles: int = 512) -> float:
    """W2^2 = ∫_0^1 |Q_a(u)-Q_b(u)|^2 du, computed on an endpoint u-grid with trapezoid."""
    q_a = quantile_from_mass(mass_a, edges, n_quantiles)
    q_b = quantile_from_mass(mass_b, edges, n_quantiles)
    u = np.linspace(0.0, 1.0, n_quantiles)
    return float(np.trapezoid((q_a - q_b) ** 2, u))


def build_neumann_laplacian(n: int) -> np.ndarray:
    """Finite-difference Laplacian stencil (no 1/dx^2 factor), Neumann via ghost points."""
    L = np.zeros((n, n), dtype=float)
    L[0, 0] = -2.0
    L[0, 1] = 2.0
    for i in range(1, n - 1):
        L[i, i - 1] = 1.0
        L[i, i] = -2.0
        L[i, i + 1] = 1.0
    L[n - 1, n - 2] = 2.0
    L[n - 1, n - 1] = -2.0
    return L


def heat_step_implicit_euler(mass_prev: np.ndarray, tau: float, dx: float, laplacian: np.ndarray) -> np.ndarray:
    """
    Implicit heat step on masses:
        (I - τ/dx^2 L) m^{k+1} = m^k.

    NOTE: We do NOT unconditionally clip/renormalize. We only correct tiny numerical drift.
    """
    m_prev = np.asarray(mass_prev, float)
    n = m_prev.size
    A = np.eye(n) - (tau / (dx * dx)) * laplacian
    m_next = np.linalg.solve(A, m_prev)

    # Correct tiny negative roundoff only; if it's meaningfully negative, warn+clip.
    min_val = float(m_next.min())
    if min_val < -1e-10:
        print(f"[Euler] warning: significant negative mass detected (min={min_val:.3e}); clipping to 0.")
        m_next = np.clip(m_next, 0.0, None)
    elif min_val < 0.0:
        m_next[m_next < 0.0] = 0.0

    s = float(m_next.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise RuntimeError("Euler step produced invalid mass vector.")
    if abs(s - 1.0) > 1e-12:
        m_next /= s
    return m_next


def jko_objective(mass: np.ndarray, mass_prev: np.ndarray, tau: float, dx: float, edges: np.ndarray) -> float:
    """JKO functional: F(m) + (1/(2τ)) W2^2(m, m_prev)."""
    return entropy(mass, dx) + 0.5 / tau * w2_squared_1d(mass, mass_prev, edges)


def initialize_mass(x: np.ndarray, dx: float, alpha: float, beta: float, sigma1: float, sigma2: float) -> np.ndarray:
    """Two-bump density, converted to cell masses via `density*dx`."""
    rho = alpha * np.exp(-((x - 0.3) ** 2) / sigma1) + beta * np.exp(-((x - 0.7) ** 2) / sigma2)
    m = rho * dx
    m /= m.sum()
    return m


def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm with dense fallback for safety."""
    n = len(b)
    ac = np.asarray(a, dtype=float).copy()
    bc = np.asarray(b, dtype=float).copy()
    cc = np.asarray(c, dtype=float).copy()
    dc = np.asarray(d, dtype=float).copy()

    if not np.all(np.isfinite(bc)) or np.any(np.abs(bc) < 1e-14):
        M = np.diag(bc) + np.diag(ac, -1) + np.diag(cc, 1)
        return np.linalg.solve(M, dc)

    for i in range(1, n):
        piv = bc[i - 1]
        if abs(piv) < 1e-14 or not np.isfinite(piv):
            M = np.diag(bc) + np.diag(ac, -1) + np.diag(cc, 1)
            return np.linalg.solve(M, dc)
        w = ac[i - 1] / piv
        bc[i] -= w * cc[i - 1]
        dc[i] -= w * dc[i - 1]

    x = np.zeros(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        piv = bc[i]
        if abs(piv) < 1e-14 or not np.isfinite(piv):
            M = np.diag(bc) + np.diag(ac, -1) + np.diag(cc, 1)
            return np.linalg.solve(M, dc)
        x[i] = (dc[i] - cc[i] * x[i + 1]) / piv
    return x


def _project_increasing_fixed_ends(Q: np.ndarray, lo: float, hi: float, min_gap: float) -> np.ndarray:
    """Simple feasibility projection used to keep Newton iterates admissible."""
    Qp = np.asarray(Q, float).copy()
    n = Qp.size
    Qp[0] = lo
    Qp[-1] = hi
    for i in range(1, n):
        Qp[i] = max(Qp[i], Qp[i - 1] + min_gap)
    for i in range(n - 2, -1, -1):
        Qp[i] = min(Qp[i], Qp[i + 1] - min_gap)
    if lo + (n - 1) * min_gap > hi:
        Qp = np.linspace(lo, hi, n)
    return Qp


def jko_step_entropy_1d_via_quantiles(
    Q_prev: np.ndarray,
    tau: float,
    x_bounds: tuple[float, float],
    max_newton: int = 200,
    tol: float = 1e-11,
    min_gap: float = 1e-9,
) -> np.ndarray:
    """
    JKO step in quantile coordinates with endpoints included.

    Objective:
      ent(Q)  = -∫ log Q'(u) du   ≈  du * (-Σ log(Q_{i+1}-Q_i))  (constant omitted)
      w2(Q)   = (1/(2τ)) ∫ |Q-Q_prev|^2 du  (trapezoid in u)
    """
    Q_prev = np.asarray(Q_prev, float)
    n = Q_prev.size
    if n < 3:
        raise ValueError("Need at least 3 quantile points.")
    lo, hi = x_bounds
    du = 1.0 / (n - 1)
    w = np.ones(n, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5

    Q_prev = _project_increasing_fixed_ends(Q_prev, lo, hi, min_gap=min_gap)
    Q = Q_prev.copy()
    min_d_safe = max(min_gap, 1e-12)

    def objective(Qv: np.ndarray) -> float:
        d = np.diff(Qv)
        if np.any(d <= 0):
            return np.inf
        d = np.maximum(d, min_d_safe)
        ent = du * (-np.sum(np.log(d)))
        quad = (du / (2.0 * tau)) * float(np.sum(w * (Qv - Q_prev) ** 2))
        return float(ent + quad)

    def grad_full(Qv: np.ndarray) -> np.ndarray:
        d = np.maximum(np.diff(Qv), min_d_safe)
        g = (du / tau) * (w * (Qv - Q_prev))
        g[0] += du / d[0]
        g[-1] += -du / d[-1]
        if n > 2:
            g[1:-1] += du * ((-1.0 / d[:-1]) + (1.0 / d[1:]))
        return g

    J_old = objective(Q)
    m = n - 2

    for _ in range(max_newton):
        Q = _project_increasing_fixed_ends(Q, lo, hi, min_gap=min_gap)
        d = np.maximum(np.diff(Q), min_d_safe)
        g = grad_full(Q)[1:-1]
        if np.linalg.norm(g, ord=np.inf) < tol:
            break

        inv_d2 = du * (1.0 / (d * d))
        inv_d2 = np.minimum(inv_d2, 1e12)

        a = np.zeros(m - 1)
        b = np.zeros(m)
        c = np.zeros(m - 1)
        for j in range(m):
            i = j + 1
            b[j] = (du / tau) * w[i] + inv_d2[i - 1] + inv_d2[i]
            if j > 0:
                a[j - 1] = -inv_d2[i - 1]
            if j < m - 1:
                c[j] = -inv_d2[i]

        p = solve_tridiagonal(a, b, c, -g)

        step = 1.0
        while step > 1e-12:
            Q_try = Q.copy()
            Q_try[1:-1] += step * p
            Q_try = _project_increasing_fixed_ends(Q_try, lo, hi, min_gap=min_gap)
            J_try = objective(Q_try)
            if J_try <= J_old - 1e-4 * step * float(np.dot(g, p)):
                Q = Q_try
                J_old = J_try
                break
            step *= 0.5

        if step <= 1e-12:
            # return last feasible iterate (matches your current behavior)
            return Q

    return Q


def mass_from_quantile(Q: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Convert endpoint-quantiles Q(u_i) into cell masses:
      m_i = F(edges[i+1]) - F(edges[i])
    where F is approximated by inverting the quantile map.
    """
    Q = np.asarray(Q, float)
    u = np.linspace(0.0, 1.0, Q.size)
    u_edges = np.interp(edges, Q, u, left=0.0, right=1.0)
    u_edges = np.clip(u_edges, 0.0, 1.0)
    u_edges = np.maximum.accumulate(u_edges)
    m = np.diff(u_edges)
    m = np.clip(m, 0.0, None)
    # tiny floor then renormalize
    m += 1e-12
    m /= m.sum()
    return m


def jko_step_quantile(mass_prev: np.ndarray, edges: np.ndarray, tau: float, n_quantiles: int = 256) -> np.ndarray:
    Q_prev = quantile_from_mass(mass_prev, edges, n_quantiles)
    Q_next = jko_step_entropy_1d_via_quantiles(Q_prev, tau=tau, x_bounds=(edges[0], edges[-1]))
    m_next = mass_from_quantile(Q_next, edges)
    m_next = np.clip(m_next, 0.0, None)
    m_next /= m_next.sum()
    return m_next


def _diagnostics(name: str, t: float, x: np.ndarray, dx: float, mass: np.ndarray):
    s = float(mass.sum())
    mmin = float(mass.min())
    ent = entropy(mass, dx)
    mean_x = float(np.dot(mass, x))
    print(f"[{name:4s}] t={t: .4f}  sum={s:.16f}  min={mmin:+.3e}  mean={mean_x:.6f}  entropy={ent:.6f}")


def plot_snapshots(x: np.ndarray, dx: float, snapshots: list[np.ndarray], times: list[float], title: str, save_path: str):
    fig = plt.figure(figsize=(7, 4))
    for mass, t in zip(snapshots, times):
        plt.plot(x, mass / dx, label=f"t={t:.3f}")
    plt.xlabel("x")
    plt.ylabel("density ρ(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def run_euler(n_cells: int, tau: float, steps: int, snapshot_every: int, *, alpha: float, beta: float, sigma1: float, sigma2: float):
    x, dx, edges = make_grid(n_cells)
    m = initialize_mass(x, dx, alpha, beta, sigma1, sigma2)
    L = build_neumann_laplacian(n_cells)
    snaps: list[np.ndarray] = []
    times: list[float] = []
    for k in range(steps + 1):
        if k % snapshot_every == 0:
            snaps.append(m.copy())
            times.append(k * tau)
        if k < steps:
            m = heat_step_implicit_euler(m, tau=tau, dx=dx, laplacian=L)
    return x, dx, edges, snaps, times


def run_jko(n_cells: int, tau: float, steps: int, snapshot_every: int, n_quantiles: int, *, alpha: float, beta: float, sigma1: float, sigma2: float):
    x, dx, edges = make_grid(n_cells)
    m = initialize_mass(x, dx, alpha, beta, sigma1, sigma2)
    snaps: list[np.ndarray] = []
    times: list[float] = []
    for k in range(steps + 1):
        if k % snapshot_every == 0:
            snaps.append(m.copy())
            times.append(k * tau)
        if k < steps:
            m = jko_step_quantile(m, edges=edges, tau=tau, n_quantiles=n_quantiles)
    return x, dx, edges, snaps, times


def main():
    parser = argparse.ArgumentParser(description="1D heat: Euler vs JKO (quantile Newton)")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--sigma1", type=float, default=0.9)
    parser.add_argument("--sigma2", type=float, default=0.16)
    parser.add_argument("--n-cells", type=int, default=400)
    parser.add_argument("--tau", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--n-quantiles", type=int, default=400)
    args = parser.parse_args()

    alpha = args.alpha
    beta = args.beta
    sigma1 = args.sigma1
    sigma2 = args.sigma2
    n_cells = args.n_cells
    tau = args.tau
    steps = args.steps
    snapshot_every = args.snapshot_every
    n_quantiles = args.n_quantiles

    xE, dxE, edgesE, snapsE, timesE = run_euler(
        n_cells, tau, steps, snapshot_every, alpha=alpha, beta=beta, sigma1=sigma1, sigma2=sigma2
    )
    xJ, dxJ, edgesJ, snapsJ, timesJ = run_jko(
        n_cells, tau, steps, snapshot_every, n_quantiles, alpha=alpha, beta=beta, sigma1=sigma1, sigma2=sigma2
    )

    # Diagnostics at snapshots
    for t, mE, mJ in zip(timesE, snapsE, snapsJ):
        _diagnostics("Euler", t, xE, dxE, mE)
        _diagnostics("JKO", t, xJ, dxJ, mJ)

    euler_png = f"plot_euler_alpha{alpha}_beta{beta}.png"
    jko_png = f"plot_jko_alpha{alpha}_beta{beta}.png"
    plot_snapshots(xE, dxE, snapsE, timesE, "Euler (implicit heat step)", euler_png)
    plot_snapshots(xJ, dxJ, snapsJ, timesJ, "JKO (quantile)", jko_png)
    print(f"Saved: {euler_png}")
    print(f"Saved: {jko_png}")


if __name__ == "__main__":
    main()


