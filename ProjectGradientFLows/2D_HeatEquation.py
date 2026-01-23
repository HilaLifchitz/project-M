"""
2D heat equation on a square with Neumann (reflecting) boundary conditions,
implemented in two ways:

1) PDE sanity check: an implicit diffusion solver (ADI / split implicit Euler)
2) JKO step (one time step) solved via a *dynamic OT* (Benamou–Brenier) formulation
   and a primal–dual (Chambolle–Pock) algorithm.

Why this is the "next rung" after 1D:
- In 1D, W2 is easy via quantiles.
- In 2D, OT is the bottleneck; the dynamic OT formulation replaces W2^2 by a convex
  problem over (rho_t, m_t) subject to a discrete continuity equation.

We work with *cell masses* m[i,j] >= 0 that sum to 1.
On a uniform grid dx=dy, the density is rho = m / (dx*dy).
"""

from __future__ import annotations

import argparse
import numpy as np  # type: ignore

# Avoid GUI backends; we only save images.
import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore


# -----------------------------
# Grid + basic functionals
# -----------------------------


def make_grid(n: int):
    """Uniform n×n cell grid on [0,1]×[0,1]. Returns centers (X,Y), dx, edges."""
    edges = np.linspace(0.0, 1.0, n + 1)
    dx = edges[1] - edges[0]
    xc = 0.5 * (edges[:-1] + edges[1:])
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    return X, Y, dx, edges


def entropy_mass(mass: np.ndarray, dx: float, eps: float = 1e-12) -> float:
    """
    Discrete entropy for cell masses:
      F(m) = ∫ ρ log ρ dx  ≈  Σ m_ij log(m_ij / dx^2).
    """
    m = np.asarray(mass, float)
    m = np.maximum(m, eps)
    return float(np.sum(m * (np.log(m) - 2.0 * np.log(dx))))


def normalize_mass(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, float)
    m = np.clip(m, 0.0, None)
    s = float(m.sum())
    if s <= 0.0 or not np.isfinite(s):
        raise RuntimeError("Invalid mass.")
    return m / s


def diagnostics(tag: str, t: float, m: np.ndarray, dx: float):
    s = float(m.sum())
    mn = float(m.min())
    ent = entropy_mass(m, dx)
    print(f"[{tag:5s}] t={t: .4f}  sum={s:.16f}  min={mn:+.3e}  entropy={ent:.6f}")


# -----------------------------
# 2D Neumann Laplacian + ADI solver (sanity check)
# -----------------------------


def build_neumann_laplacian_1d(n: int) -> np.ndarray:
    """1D Laplacian stencil matrix (no 1/dx^2 factor) with Neumann BCs."""
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


def adi_heat_step_neumann(mass: np.ndarray, tau: float, dx: float, L1d: np.ndarray) -> np.ndarray:
    """
    One approximate implicit Euler step for ∂_t ρ = Δρ (Neumann) using ADI.

    We evolve cell masses (sum to 1). The scheme is:
      (I - r Lx) U* = (I + r Ly) U^k
      (I - r Ly) U^{k+1} = (I + r Lx) U*
    with r = tau/(2 dx^2) and Lx,Ly the 1D Neumann Laplacian applied along each axis.

    This is a standard stable split for diffusion and is a good sanity check.
    """
    U = np.asarray(mass, float)
    n = U.shape[0]
    r = tau / (2.0 * dx * dx)
    A = np.eye(n) - r * L1d
    B = np.eye(n) + r * L1d

    # (I - r Lx) U* = (I + r Ly) U
    RHS = (U @ B.T)  # apply Ly on right (along y)
    Ustar = np.empty_like(U)
    for j in range(n):
        Ustar[:, j] = np.linalg.solve(A, RHS[:, j])

    # (I - r Ly) Unew = (I + r Lx) U*
    RHS2 = (B @ Ustar)  # apply Lx on left (along x)
    Unew = np.empty_like(U)
    for i in range(n):
        Unew[i, :] = np.linalg.solve(A, RHS2[i, :])

    # Small numerical cleanup only.
    if float(Unew.min()) < -1e-10:
        print(f"[Euler] warning: negative mass (min={float(Unew.min()):.3e}), clipping.")
        Unew = np.clip(Unew, 0.0, None)
    else:
        Unew[Unew < 0.0] = 0.0
    if abs(float(Unew.sum()) - 1.0) > 1e-12:
        Unew = Unew / float(Unew.sum())
    return Unew


def implicit_heat_step_full(mass: np.ndarray, tau: float, dx: float, L1d: np.ndarray) -> np.ndarray:
    """
    Full (unsplit) implicit Euler step for 2D heat with Neumann BCs:
        (I - τ Δ) ρ^{k+1} = ρ^k,  Δ ≈ (Lx⊗I + I⊗Ly)/dx^2.

    Implemented by building the Kronecker-sum matrix explicitly (dense).
    This is O(n^6) worst-case if you go huge, but fine for n<=64-ish.
    """
    U = np.asarray(mass, float)
    n = U.shape[0]
    # Work directly with masses; the operator acts componentwise the same (dx^2 factor cancels).
    I = np.eye(n)
    L2 = np.kron(L1d, I) + np.kron(I, L1d)
    A = np.eye(n * n) - (tau / (dx * dx)) * L2
    u_new = np.linalg.solve(A, U.reshape(-1))
    Unew = u_new.reshape(n, n)
    # Numerical cleanup only
    if float(Unew.min()) < -1e-10:
        print(f"[Euler-full] warning: negative mass (min={float(Unew.min()):.3e}), clipping.")
        Unew = np.clip(Unew, 0.0, None)
    else:
        Unew[Unew < 0.0] = 0.0
    if abs(float(Unew.sum()) - 1.0) > 1e-12:
        Unew = Unew / float(Unew.sum())
    return Unew


# -----------------------------
# Dynamic OT (Benamou–Brenier) JKO step via Chambolle–Pock
# -----------------------------


def div_neumann(mx: np.ndarray, my: np.ndarray, dx: float) -> np.ndarray:
    """
    Discrete divergence at cell centers with a *matched adjoint* gradient.

    IMPORTANT: This divergence is chosen so that
        <div(m), p> = - <m, grad(p)>
    under the standard discrete inner product (plain sum), with Neumann no-flux
    enforced by setting the "outgoing" components to 0 in `grad_neumann`.

    mx,my are cell-centered momentum components of shape (n,n).
    """
    # backward differences. For true no-flux boundaries we must enforce
    # mx=0 at x-boundaries and my=0 at y-boundaries (handled elsewhere too).
    div = np.zeros_like(mx, dtype=float)

    # x-part
    div[0, :] += mx[0, :] / dx
    div[1:, :] += (mx[1:, :] - mx[:-1, :]) / dx

    # y-part
    div[:, 0] += my[:, 0] / dx
    div[:, 1:] += (my[:, 1:] - my[:, :-1]) / dx

    return div


def div_adjoint_neumann(p: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact adjoint of `div_neumann` under the plain sum inner product:
        <div_neumann(mx,my), p> = <(mx,my), div_adjoint_neumann(p)>

    For the backward-difference divergence used in `div_neumann`, the adjoint is
    a *negative forward difference*:
        gx[i] = (p[i] - p[i+1]) / dx, with gx[-1]=0,
        gy[j] = (p[j] - p[j+1]) / dx, with gy[:,-1]=0.
    """
    gx = np.zeros_like(p, dtype=float)
    gy = np.zeros_like(p, dtype=float)
    gx[:-1, :] = (p[:-1, :] - p[1:, :]) / dx
    gx[-1, :] = 0.0
    gy[:, :-1] = (p[:, :-1] - p[:, 1:]) / dx
    gy[:, -1] = 0.0
    return gx, gy


def adjoint_test(dx: float, n: int = 32, seed: int = 0) -> float:
    """Return relative adjointness error for <div(m),p> == <m,div^*(p)>."""
    rng = np.random.default_rng(seed)
    mx = rng.standard_normal((n, n))
    my = rng.standard_normal((n, n))
    # Enforce the same no-flux constraints used in the algorithm
    mx[0, :] = 0.0
    mx[-1, :] = 0.0
    my[:, 0] = 0.0
    my[:, -1] = 0.0
    p = rng.standard_normal((n, n))
    lhs = float(np.sum(div_neumann(mx, my, dx) * p))
    gx, gy = div_adjoint_neumann(p, dx)
    rhs = float(np.sum(mx * gx + my * gy))
    denom = max(1e-12, abs(lhs), abs(rhs))
    return abs(lhs - rhs) / denom


def prox_kinetic_cell(m: np.ndarray, mx: np.ndarray, my: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prox for the perspective kinetic term at each cell:

      prox_{lam * (|m|^2/mass)} applied to (mass, mx, my).

    More precisely, for each cell we solve:
      min_{r>0, q} 0.5 (r-r0)^2 + 0.5||q-q0||^2 + lam * (|q|^2 / r)

    This is separable per cell:
    - For fixed r, q has a closed-form.
    - Then solve for r with 1D Newton (few iterations).
    """
    # Correct, stable prox for the perspective kinetic term:
    # min_{r>0,q} 0.5(r-r0)^2 + 0.5||q-q0||^2 + lam * ||q||^2 / r
    # Eliminating q gives a 1D convex problem for r with derivative
    #   φ'(r) = (r-r0) - lam*||q0||^2 / (r + 2 lam)^2
    # and φ''(r) = 1 + 2 lam*||q0||^2 / (r+2 lam)^3.
    r0 = np.asarray(m, float)
    qx0 = np.asarray(mx, float)
    qy0 = np.asarray(my, float)
    s = qx0 * qx0 + qy0 * qy0

    k = 2.0 * lam
    r = np.maximum(r0, 1e-12).copy()
    for _ in range(30):
        denom = np.maximum(r + k, 1e-12)
        denom2 = denom * denom
        denom3 = denom2 * denom
        fp = (r - r0) - (lam * s) / denom2
        fpp = 1.0 + (2.0 * lam * s) / denom3
        step = fp / np.maximum(fpp, 1e-12)
        r_new = np.maximum(r - step, 1e-12)
        if float(np.max(np.abs(step))) < 1e-10:
            r = r_new
            break
        r = r_new

    scale = r / np.maximum(r + k, 1e-12)
    qx = qx0 * scale
    qy = qy0 * scale
    return r, qx, qy


def prox_entropy(m: np.ndarray, lam: float) -> np.ndarray:
    """
    Prox for lam * Σ m log m  (m>=0), applied elementwise via Newton:
      minimize 0.5 (x - a)^2 + lam * x log x
    => x - a + lam (log x + 1) = 0.
    """
    a = np.asarray(m, float)
    x = np.maximum(a, 1e-12).copy()
    for _ in range(30):
        f = x - a + lam * (np.log(x) + 1.0)
        fp = 1.0 + lam / x
        step = f / fp
        x_new = x - step
        x = np.maximum(x_new, 1e-12)
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return x


def action_bb(rho: np.ndarray, mx: np.ndarray, my: np.ndarray, dx: float, dt: float, eps: float = 1e-12) -> float:
    """
    Discrete Benamou–Brenier action:
        A = ∫_0^1 ∫ |m|^2 / rho  dx dt
      ≈ Σ_t dt * Σ_cells dx^2 * (mx^2 + my^2) / rho

    Here:
      - rho has shape (T+1,n,n)
      - mx,my have shape (T,n,n)
    """
    rho_safe = np.maximum(rho[:-1], eps)
    num = mx * mx + my * my
    return float(np.sum(dt * (dx * dx) * num / rho_safe))


def entropy_density(rho: np.ndarray, dx: float, eps: float = 1e-12) -> float:
    """
    Discrete entropy for density:
        E = ∫ rho log rho dx  ≈ Σ dx^2 * rho log rho
    """
    r = np.maximum(rho, eps)
    return float(np.sum((dx * dx) * r * np.log(r)))


def jko_step_dynamic_ot(
    m0: np.ndarray,
    tau: float,
    dx: float,
    n_transport: int = 8,
    n_iters: int = 800,
    theta: float = 1.0,
    primal_step: float = 0.01,
    dual_step: float = 0.01,
    res_check_every: int = 50,
    res_tol: float = 1e-4,
    verbose: bool = False,
    debug_objective: bool = False,
) -> np.ndarray:
    """
    One 2D JKO step for the heat equation using a dynamic OT formulation.

    We solve the convex problem over (m_t, M_t) for t=0..T:

      minimize  Entropy(m_T) + (1/(2τ)) ∫_0^1 ∫ |M_t|^2 / m_t dx dt
      subject to  m_0 = given, and  (m_{t+1}-m_t)/dt + div(M_t) = 0.

    Discretization:
    - time: t=0..n_transport, dt = 1/n_transport
    - space: n×n uniform grid, dx
    - div is a Neumann (no-flux) finite difference divergence

    Algorithm:
    - Chambolle–Pock primal–dual on the constraint K(m,M)=0
    - prox for kinetic term is separable per cell (uses prox_kinetic_cell)
    - prox for entropy at final time uses prox_entropy
    """
    n = m0.shape[0]
    dt = 1.0 / n_transport
    # Chambolle–Pock stability requires primal_step * dual_step * ||K||^2 < 1.
    # A conservative bound for our discrete K = (time-diff)/dt + div/dx is:
    #   ||K||^2 <= 2/dt^2 + 8/dx^2
    K_norm2_bound = 2.0 / (dt * dt) + 8.0 / (dx * dx)
    if primal_step * dual_step * K_norm2_bound >= 0.9:
        # If user picked aggressive steps, automatically scale them down.
        scale = np.sqrt((0.5) / (primal_step * dual_step * K_norm2_bound))
        primal_step *= scale
        dual_step *= scale

    # We solve the dynamic OT problem in terms of *densities* rho and momentum densities (mx,my).
    # Convert initial masses to density so that sum(rho)*dx^2 = 1.
    rho0 = np.asarray(m0, float) / (dx * dx)

    # primal variables
    rho = np.repeat(rho0[None, :, :], n_transport + 1, axis=0)  # (T+1,n,n)
    mx = np.zeros((n_transport, n, n), dtype=float)  # momentum density
    my = np.zeros((n_transport, n, n), dtype=float)

    # dual variable for the continuity constraint (one per time-slab)
    p = np.zeros((n_transport, n, n), dtype=float)

    # Over-relaxed copies
    m_bar = rho.copy()
    mx_bar = mx.copy()
    my_bar = my.copy()

    # Precompute scaling:
    # kinetic term coefficient: (1/(2τ)) * dt * dx^2 * |m|^2/rho
    alpha = (dt * dx * dx) / (2.0 * tau)
    # entropy term coefficient: dx^2 * rho log rho (in density variables)
    ent_scale = dx * dx

    # IMPORTANT SCALING CHOICE:
    # We enforce continuity in the numerically-balanced form
    #     rho_{t+1} - rho_t + dt * div(m_t) = 0
    # instead of dividing by dt (which makes ||K|| huge and harms CP behavior).
    for it in range(n_iters):
        # --- Dual ascent: p <- p + sigma * K(rho_bar, m_bar)
        # K_t = rho_{t+1} - rho_t + dt * div(m_t)
        for t in range(n_transport):
            p[t] += dual_step * ((m_bar[t + 1] - m_bar[t]) + dt * div_neumann(mx_bar[t], my_bar[t], dx))

        # --- Primal descent: (m,M) <- prox_G( (m,M) - tau_p * K^*(p) )
        m_old = rho.copy()
        mx_old = mx.copy()
        my_old = my.copy()

        # K^* on rho (time-difference adjoint for K_t = rho_{t+1}-rho_t + ...)
        g_m = np.zeros_like(rho)
        # rho0 is fixed -> do not update it (treat as hard constraint)
        g_m[0] = 0.0
        for t in range(1, n_transport):
            g_m[t] = (p[t - 1] - p[t])
        g_m[n_transport] = p[n_transport - 1]

        # K^* on momentum: adjoint of dt*div is dt*div^*
        # Primal update: m <- m - tau_p * (dt * div^*(p)).
        for t in range(n_transport):
            gx, gy = div_adjoint_neumann(p[t], dx)
            mx[t] -= primal_step * (dt * gx)
            my[t] -= primal_step * (dt * gy)

        rho -= primal_step * g_m

        # Prox for kinetic term on t=0..T-1 (m_t, M_t)
        lam_kin = primal_step * alpha
        for t in range(n_transport):
            rt, mxt, myt = prox_kinetic_cell(rho[t], mx[t], my[t], lam=lam_kin)
            rho[t] = rt
            mx[t] = mxt
            my[t] = myt

        # Prox for entropy on final slice rho_T:
        # objective contribution is ent_scale * rho log rho, so prox uses lam = primal_step*ent_scale
        rho[n_transport] = prox_entropy(rho[n_transport], lam=primal_step * ent_scale)

        # Enforce rho0 fixed and only clip negatives
        rho[0] = rho0
        min_m = float(np.min(rho))
        if min_m < -1e-8:
            print(f"[JKO] warning: negative mass in iterate (min={min_m:.3e}); clipping to 0.")
        rho[rho < 0.0] = 0.0

        # Enforce no-flux momentum on boundaries (Neumann): both sides 0 for our div stencil
        mx[:, 0, :] = 0.0
        mx[:, -1, :] = 0.0
        my[:, :, 0] = 0.0
        my[:, :, -1] = 0.0

        # Over-relaxation
        m_bar = rho + theta * (rho - m_old)
        mx_bar = mx + theta * (mx - mx_old)
        my_bar = my + theta * (my - my_old)

        # NaN/Inf guard: if the iterates blow up, stop early with a clear error.
        if not np.isfinite(m_bar).all() or not np.isfinite(p).all() or not np.isfinite(mx_bar).all() or not np.isfinite(my_bar).all():
            raise RuntimeError(
                "Dynamic-OT primal-dual diverged (NaNs/Infs). "
                "Try smaller --pd-iters, smaller primal/dual steps, or larger --tau."
            )

        # Residual check: continuity constraint should approach 0 as the inner solve converges.
        if res_check_every > 0 and (it + 1) % res_check_every == 0:
            res = 0.0
            for t in range(n_transport):
                Kt = (rho[t + 1] - rho[t]) + dt * div_neumann(mx[t], my[t], dx)
                res = max(res, float(np.max(np.abs(Kt))))
            if verbose:
                print(f"[JKO inner] it={it+1:4d}  cont_res_inf={res:.3e}")
            if debug_objective:
                A = action_bb(rho, mx, my, dx=dx, dt=dt)
                E = entropy_density(rho[n_transport], dx=dx)
                J = E + (1.0 / (2.0 * tau)) * A
                print(f"[JKO obj  ] it={it+1:4d}  E={E:.6e}  A={A:.6e}  J={J:.6e}")
            if res < res_tol:
                break

    # Convert rho_T back to masses, and renormalize to sum 1.
    mT = normalize_mass(rho[n_transport] * (dx * dx))
    return mT


# -----------------------------
# Plotting + main loop
# -----------------------------


def plot_density(X: np.ndarray, Y: np.ndarray, m: np.ndarray, dx: float, title: str, save_path: str):
    rho = m / (dx * dx)
    fig = plt.figure(figsize=(5, 4))
    plt.pcolormesh(X, Y, rho, shading="auto")
    plt.colorbar(label="density ρ")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_compare_side_by_side(
    X: np.ndarray,
    Y: np.ndarray,
    m_euler: np.ndarray,
    m_jko: np.ndarray,
    dx: float,
    title_left: str,
    title_right: str,
    save_path: str,
    *,
    smooth: bool = True,
    common_scale: bool = True,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    rhoL = m_euler / (dx * dx)
    rhoR = m_jko / (dx * dx)

    # IMPORTANT: without a shared color scale, the two panels can look "similar"
    # even if magnitudes differ (each gets its own auto-scaled colorbar).
    vmin = 0.0
    vmax = float(max(rhoL.max(), rhoR.max())) if common_scale else None
    interp = "bilinear" if smooth else "nearest"
    extent = (0.0, 1.0, 0.0, 1.0)

    im0 = axes[0].imshow(
        rhoL.T,
        origin="lower",
        extent=extent,
        interpolation=interp,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[0].set_title(title_left)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(
        rhoR.T,
        origin="lower",
        extent=extent,
        interpolation=interp,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axes[1].set_title(title_right)
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="2D heat: Euler (ADI) vs JKO (dynamic OT)")
    parser.add_argument("--n", type=int, default=48, help="Grid size n×n.")
    parser.add_argument("--tau", type=float, default=5e-4, help="Outer time step (JKO/PDE).")
    parser.add_argument("--steps", type=int, default=30, help="Number of outer time steps.")
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--transport-slices", type=int, default=8, help="Inner transport time slices for dynamic OT.")
    parser.add_argument("--pd-iters", type=int, default=800, help="Max primal-dual iterations per JKO step.")
    parser.add_argument("--pd-primal-step", type=float, default=0.01, help="Chambolle–Pock primal step size.")
    parser.add_argument("--pd-dual-step", type=float, default=0.01, help="Chambolle–Pock dual step size.")
    parser.add_argument("--pd-res-tol", type=float, default=1e-4, help="Inner continuity residual tolerance (inf-norm).")
    parser.add_argument("--pd-check-every", type=int, default=50, help="Check residual every N inner iterations (0 disables).")
    parser.add_argument("--pd-verbose", action="store_true", help="Print inner residual progress.")
    parser.add_argument("--pd-debug-objective", action="store_true", help="Print inner JKO objective components (E, action, J).")
    parser.add_argument("--alpha", type=float, default=0.8, help="Weight of bump 1.")
    parser.add_argument("--beta", type=float, default=0.2, help="Weight of bump 2.")
    parser.add_argument("--plot-smooth", action="store_true", help="Use smooth interpolation in plots (visual only).")
    parser.add_argument("--plot-common-scale", action="store_true", help="Use the same color scale for Euler/JKO at each snapshot.")
    parser.add_argument("--euler-solver", choices=["adi", "full"], default="adi", help="Euler sanity solver: ADI split or full implicit (dense).")
    args = parser.parse_args()

    n = args.n
    tau = args.tau
    steps = args.steps
    snapshot_every = args.snapshot_every
    T = args.transport_slices
    pd_iters = args.pd_iters
    pd_primal = args.pd_primal_step
    pd_dual = args.pd_dual_step
    pd_res_tol = args.pd_res_tol
    pd_check_every = args.pd_check_every
    pd_verbose = args.pd_verbose
    pd_dbg_obj = args.pd_debug_objective
    alpha = args.alpha
    beta = args.beta
    plot_smooth = args.plot_smooth
    plot_common = args.plot_common_scale
    euler_solver = args.euler_solver

    X, Y, dx, edges = make_grid(n)
    err = adjoint_test(dx=dx, n=n, seed=0)
    print(f"[adjoint] relative_error={err:.3e} (should be ~1e-12..1e-8)")
    # Two Gaussian bumps as initial density, then convert to masses
    rho0 = alpha * np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 0.01) + beta * np.exp(
        -((X - 0.1) ** 2 + (Y - 0.7) ** 2) / 0.02
    )
    m0 = normalize_mass(rho0 * (dx * dx))

    # Euler sanity check state
    m_e = m0.copy()
    L1d = build_neumann_laplacian_1d(n)

    # JKO state
    m_j = m0.copy()

    # Snapshot outputs
    for k in range(steps + 1):
        t = k * tau
        if k % snapshot_every == 0:
            diagnostics("Euler", t, m_e, dx)
            diagnostics("JKO", t, m_j, dx)
            print(f"[max  ] t={t: .4f}  max_rho_euler={float((m_e/(dx*dx)).max()):.3e}  max_rho_jko={float((m_j/(dx*dx)).max()):.3e}")
            plot_compare_side_by_side(
                X,
                Y,
                m_e,
                m_j,
                dx,
                title_left=f"Euler t={t:.3f}",
                title_right=f"JKO t={t:.3f}",
                save_path=f"compare2d_t{t:.3f}_n{n}.png",
                smooth=plot_smooth,
                common_scale=plot_common,
            )

        if k < steps:
            if euler_solver == "adi":
                m_e = adi_heat_step_neumann(m_e, tau=tau, dx=dx, L1d=L1d)
            else:
                if n > 80:
                    raise RuntimeError("Full implicit Euler is too expensive for n>80; use --euler-solver adi.")
                m_e = implicit_heat_step_full(m_e, tau=tau, dx=dx, L1d=L1d)
            m_j = jko_step_dynamic_ot(
                m_j,
                tau=tau,
                dx=dx,
                n_transport=T,
                n_iters=pd_iters,
                primal_step=pd_primal,
                dual_step=pd_dual,
                res_check_every=pd_check_every,
                res_tol=pd_res_tol,
                verbose=pd_verbose,
                debug_objective=pd_dbg_obj,
            )


if __name__ == "__main__":
    main()


