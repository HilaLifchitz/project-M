"""
1D heat equation on [0,1] with Neumann BCs.

Two solvers:
- Implicit Euler PDE step (sanity check)
- "True JKO" step for entropy in 1D quantile coordinates

This file is based on `1D_b.py` and keeps its cleanup, but updates the JKO step
to optimize over *log-gaps* (softmax parameterization) rather than projecting
Newton iterates back onto the monotone set. This removes a major source of
boundary artifacts.

It also saves:
- per-method plots (filenames include method + alpha/beta)
- a side-by-side comparison plot whose filename is ONLY the parameters
  (alpha, beta, sigma1, sigma2, tau, n_cells).

---------------------------------------------------------------------------
Notes / comments copied from `1D.py`
---------------------------------------------------------------------------

Physical meaning (heat equation, Neumann BCs):
- The PDE is the 1D heat equation: ∂_t u = ∂_{xx} u on x∈[0,1]
- Neumann BCs: ∂_x u = 0 at both ends (insulated boundaries, no flux)
- Total mass/heat is conserved; solutions flatten to a uniform equilibrium.

What the “true JKO” algorithm is doing (quantile formulation, 1D):

In theory, one JKO step is the infinite-dimensional minimization

    J_τ(m | m^k) = F(m) + (1 / (2τ)) W_2^2(m, m^k),

where F is a functional on probability densities (here F(ρ)=∫ ρ log ρ dx).
In one spatial dimension, rewrite in terms of the quantile function

    Q(u) = F^{-1}(u),   u ∈ (0, 1),

which is monotone increasing.

Why the entropy becomes a log(Q') term:
Let ρ have CDF F(x)=∫_{-∞}^x ρ(s)ds and quantile Q=F^{-1}. Then F(Q(u))=u, so
F'(Q(u)) Q'(u) = 1 and since F'(x)=ρ(x) this gives ρ(Q(u)) = 1/Q'(u).
Changing variables u=F(x) (so du=ρ(x)dx) yields

    ∫ ρ(x) log ρ(x) dx = ∫_0^1 log(ρ(Q(u))) du = -∫_0^1 log Q'(u) du  (+const).

Wasserstein term in quantile variables:
In 1D, W_2^2(m, m^k) = ∫_0^1 |Q(u) - Q^k(u)|^2 du.

Discrete formulation (one common choice):
With Δu, Q'(u_i) ≈ (Q_{i+1}-Q_i)/Δu, and the discrete objective becomes

    J(Q) = -∑ log(Q_{i+1}-Q_i) + (Δu/(2τ)) ∑ (Q_i - Q_i^k)^2

on the admissible set {Q_{i+1} > Q_i}.
"""

from __future__ import annotations

import argparse
import numpy as np  # type: ignore

# Non-interactive backend (avoid Qt/Wayland issues)
import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore


def make_grid(n_cells: int = 200):
    edges = np.linspace(0.0, 1.0, n_cells + 1)
    dx = edges[1] - edges[0]
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, dx, edges


def entropy(mass: np.ndarray, dx: float, eps: float = 1e-12) -> float:
    m = np.asarray(mass, float)
    m = np.maximum(m, eps)
    return float(np.sum(m * (np.log(m) - np.log(dx))))


def quantile_from_mass(mass: np.ndarray, edges: np.ndarray, n_quantiles: int = 512) -> np.ndarray:
    """Exact inverse CDF Q(u)=F^{-1}(u) for piecewise-constant density on cells."""
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
    q_a = quantile_from_mass(mass_a, edges, n_quantiles)
    q_b = quantile_from_mass(mass_b, edges, n_quantiles)
    u = np.linspace(0.0, 1.0, n_quantiles)
    return float(np.trapezoid((q_a - q_b) ** 2, u))


def build_neumann_laplacian(n: int) -> np.ndarray:
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
    m_prev = np.asarray(mass_prev, float)
    n = m_prev.size
    A = np.eye(n) - (tau / (dx * dx)) * laplacian
    m_next = np.linalg.solve(A, m_prev)

    # Only fix tiny roundoff
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


def initialize_mass(x: np.ndarray, dx: float, alpha: float, beta: float, sigma1: float, sigma2: float) -> np.ndarray:
    rho = alpha * np.exp(-((x - 0.3) ** 2) / sigma1) + beta * np.exp(-((x - 0.7) ** 2) / sigma2)
    m = rho * dx
    m /= m.sum()
    return m


def jko_objective(mass: np.ndarray, mass_prev: np.ndarray, tau: float, dx: float, edges: np.ndarray) -> float:
    return entropy(mass, dx) + 0.5 / tau * w2_squared_1d(mass, mass_prev, edges)


def mass_from_quantile(Q: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """m_i = F(edges[i+1]) - F(edges[i]), where F is obtained by inverting Q."""
    Q = np.asarray(Q, float)
    u = np.linspace(0.0, 1.0, Q.size)
    u_edges = np.interp(edges, Q, u, left=0.0, right=1.0)
    u_edges = np.clip(u_edges, 0.0, 1.0)
    u_edges = np.maximum.accumulate(u_edges)
    m = np.diff(u_edges)
    m = np.clip(m, 0.0, None)
    m += 1e-12
    m /= m.sum()
    return m


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def jko_step_entropy_1d_via_loggaps(
    Q_prev: np.ndarray,
    tau: float,
    x_bounds: tuple[float, float],
    max_iter: int = 500,
    tol: float = 1e-10,
    z_init: np.ndarray | None = None,
) -> np.ndarray:
    """
    JKO step in 1D quantile coordinates using a log-gap parameterization.

    Let gaps d_i = Q_{i+1} - Q_i, i=0..n-2, with d_i > 0 and sum d_i = hi-lo.
    Parameterize:
        p = softmax(z)  (p_i>0, sum p_i=1)
        d_i = (hi-lo) * p_i
        Q_0 = lo, Q_{k} = lo + sum_{i<k} d_i

    This removes the inequality constraints (no projection needed).
    """
    Q_prev = np.asarray(Q_prev, float)
    n = Q_prev.size
    if n < 3:
        raise ValueError("Need at least 3 quantile points.")
    lo, hi = x_bounds
    L = hi - lo
    du = 1.0 / (n - 1)
    w = np.ones(n, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5

    # Initialize z either from warm start (previous step) or from previous gaps.
    if z_init is not None:
        z = np.asarray(z_init, float).copy()
        if z.size != n - 1:
            raise ValueError("z_init must have length n_quantiles-1.")
    else:
        d_prev = np.diff(Q_prev)
        d_prev = np.maximum(d_prev, 1e-12)
        d_prev = L * d_prev / d_prev.sum()
        p0 = d_prev / L
        p0 = np.clip(p0, 1e-15, None)
        p0 = p0 / p0.sum()
        z = np.log(p0)

    def z_to_Q(zv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = _softmax(zv)
        d = L * p
        Q = np.empty(n, dtype=float)
        Q[0] = lo
        Q[1:] = lo + np.cumsum(d)
        # numerical cleanup
        Q[-1] = hi
        return Q, d

    def objective(Qv: np.ndarray, dv: np.ndarray) -> float:
        # entropy part: du * (-sum log d_i) (constant dropped)
        ent = du * (-float(np.sum(np.log(np.maximum(dv, 1e-300)))))
        r = Qv - Q_prev
        quad = (du / (2.0 * tau)) * float(np.sum(w * (r * r)))
        return ent + quad

    def grad_z(Qv: np.ndarray, dv: np.ndarray, zv: np.ndarray) -> np.ndarray:
        # gradient w.r.t. gaps d
        # ent: du * (-1/d_i)
        g_d = du * (-1.0 / np.maximum(dv, 1e-300))
        # quad: (du/tau) * sum_{i>k} w_i r_i
        r = Qv - Q_prev
        # Vectorized suffix sum: suffix[k] = Σ_{i>=k+1} w[i]*r[i]
        suffix = np.cumsum((w[1:] * r[1:])[::-1])[::-1]
        g_d = g_d + (du / tau) * suffix

        # chain to z through d = L*softmax(z)
        p = dv / L
        dot_pg = float(np.dot(p, g_d))
        g_z = L * (p * (g_d - dot_pg))
        return g_z

    Q, d = z_to_Q(z)
    J = objective(Q, d)

    for _ in range(max_iter):
        g = grad_z(Q, d, z)
        gnorm = float(np.linalg.norm(g, ord=np.inf))
        if gnorm < tol:
            break

        step = 1.0
        # backtracking
        while step > 1e-12:
            z_try = z - step * g
            Q_try, d_try = z_to_Q(z_try)
            J_try = objective(Q_try, d_try)
            if J_try <= J - 1e-4 * step * float(np.dot(g, g)):
                z = z_try
                Q, d = Q_try, d_try
                J = J_try
                break
            step *= 0.5

        if step <= 1e-12:
            break

    return Q


def jko_step_quantile_loggaps(
    mass_prev: np.ndarray,
    edges: np.ndarray,
    tau: float,
    n_quantiles: int = 256,
    *,
    z_init: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    Q_prev = quantile_from_mass(mass_prev, edges, n_quantiles)
    Q_next = jko_step_entropy_1d_via_loggaps(
        Q_prev,
        tau=tau,
        x_bounds=(edges[0], edges[-1]),
        max_iter=max_iter,
        tol=tol,
        z_init=z_init,
    )
    m_next = mass_from_quantile(Q_next, edges)
    m_next = np.clip(m_next, 0.0, None)
    m_next /= m_next.sum()
    # Return z for warm start: recover z from current gaps
    d = np.diff(Q_next)
    d = np.maximum(d, 1e-15)
    p = d / d.sum()
    z_next = np.log(np.clip(p, 1e-15, None))
    return m_next, z_next


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


def plot_compare_side_by_side(
    x: np.ndarray,
    dx: float,
    snapshots_euler: list[np.ndarray],
    times_euler: list[float],
    snapshots_jko: list[np.ndarray],
    times_jko: list[float],
    title_left: str,
    title_right: str,
    save_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_l, ax_r = axes
    for mass, t in zip(snapshots_euler, times_euler):
        ax_l.plot(x, mass / dx, label=f"t={t:.3f}")
    ax_l.set_title(title_left)
    ax_l.set_xlabel("x")
    ax_l.set_ylabel("density ρ(x)")
    ax_l.legend()

    for mass, t in zip(snapshots_jko, times_jko):
        ax_r.plot(x, mass / dx, label=f"t={t:.3f}")
    ax_r.set_title(title_right)
    ax_r.set_xlabel("x")
    ax_r.legend()

    fig.tight_layout()
    fig.savefig(save_path)
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
    z: np.ndarray | None = None
    for k in range(steps + 1):
        if k % snapshot_every == 0:
            snaps.append(m.copy())
            times.append(k * tau)
        if k < steps:
            m, z = jko_step_quantile_loggaps(
                m,
                edges=edges,
                tau=tau,
                n_quantiles=n_quantiles,
                z_init=z,
            )
    return x, dx, edges, snaps, times


def main():
    parser = argparse.ArgumentParser(description="1D heat: Euler vs JKO")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--sigma1", type=float, default=0.01)
    parser.add_argument("--sigma2", type=float, default=0.20)
    parser.add_argument("--n-cells", type=int, default=400)
    parser.add_argument("--tau", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=480, help="Number of time steps.")
    parser.add_argument("--snapshot-every", type=int, default=80)
    parser.add_argument("--n-quantiles", type=int, default=400)
    parser.add_argument("--jko-max-iter", type=int, default=200, help="Max inner iterations per JKO step.")
    parser.add_argument("--jko-tol", type=float, default=1e-10, help="Infinity-norm tolerance for JKO log-gaps gradient.")
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
    jko_max_iter = args.jko_max_iter
    jko_tol = args.jko_tol

    xE, dxE, edgesE, snapsE, timesE = run_euler(
        n_cells, tau, steps, snapshot_every, alpha=alpha, beta=beta, sigma1=sigma1, sigma2=sigma2
    )
    # run_jko warm-starts internally; pass settings via closure below
    def run_jko_with_settings():
        x, dx, edges = make_grid(n_cells)
        m = initialize_mass(x, dx, alpha, beta, sigma1, sigma2)
        snaps: list[np.ndarray] = []
        times: list[float] = []
        z: np.ndarray | None = None
        for k in range(steps + 1):
            if k % snapshot_every == 0:
                snaps.append(m.copy())
                times.append(k * tau)
            if k < steps:
                m, z = jko_step_quantile_loggaps(
                    m,
                    edges=edges,
                    tau=tau,
                    n_quantiles=n_quantiles,
                    z_init=z,
                    max_iter=jko_max_iter,
                    tol=jko_tol,
                )
        return x, dx, edges, snaps, times

    xJ, dxJ, edgesJ, snapsJ, timesJ = run_jko_with_settings()

    for t, mE, mJ in zip(timesE, snapsE, snapsJ):
        _diagnostics("Euler", t, xE, dxE, mE)
        _diagnostics("JKO", t, xJ, dxJ, mJ)

    euler_png = f"plot_euler_alpha{alpha}_beta{beta}.png"
    jko_png = f"plot_jko_alpha{alpha}_beta{beta}.png"
    plot_snapshots(xE, dxE, snapsE, timesE, "Euler (implicit heat step)", euler_png)
    plot_snapshots(xJ, dxJ, snapsJ, timesJ, "JKO", jko_png)

    # Comparison plot: filename is ONLY the parameters (per request)
    compare_png = f"alpha{alpha}_beta{beta}_sigma1_{sigma1}_sigma2_{sigma2}_tau{tau}_n{n_cells}.png"
    plot_compare_side_by_side(
        xE,
        dxE,
        snapsE,
        timesE,
        snapsJ,
        timesJ,
        title_left="Euler",
        title_right="JKO",
        save_path=compare_png,
    )

    print(f"Saved: {euler_png}")
    print(f"Saved: {jko_png}")
    print(f"Saved: {compare_png}")


if __name__ == "__main__":
    main()


