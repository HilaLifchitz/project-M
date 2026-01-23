"""
A code to implemet the 1D JKO scheme for the heat equation (or Euler scheme for the heat equation)
"""
# Not why were doing it, but good to know: physical meaning of this solver (1D heat equation, Neumann BCs)
#
# This code evolves a temperature profile u(x, t) along a 1D rod.
# The PDE is the heat equation:
#
#     ∂_t u =  ∂_{xx} u
#
# with Neumann (insulated) boundary conditions:
#
#     ∂_x u = 0 at both ends of the rod.
#
# Interpretation:
# - u(x, t) is the temperature at position x and time t.
# - The term ∂_{xx} u models diffusion of heat: hot regions spread into cold ones.
# - Neumann BCs mean no heat flux across the boundaries (an insulated rod).
#
# Key physical facts:
# - Total heat (the integral of u over the rod) is conserved in time.
# - Temperature gradients are smoothed out: sharp peaks flatten.
# - As t → ∞, u(x, t) converges to a uniform temperature equal to the initial
#   average temperature along the rod.
#
# Numerically, this solver approximates that diffusion process step by step,
# preserving mass (total heat) and progressively flattening the profile until
# it reaches this equilibrium state.


import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def make_grid(n_cells: int = 200):
    """
    Uniform cell grid on [0, 1].

    - x: cell centers (length n_cells)
    - dx: cell width
    - edges: cell edges (length n_cells+1), edges[0]=0 and edges[-1]=1
    """
    edges = np.linspace(0.0, 1.0, n_cells + 1)
    dx = edges[1] - edges[0]
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, dx, edges


def entropy(mass: np.ndarray, dx: float, eps: float = 1e-12) -> float:
    """
    Discrete entropy F(ρ) = ∫ ρ log ρ.

    The input `mass` stores cell masses (summing to 1). The density in a cell
    is mass / dx. define m_i = mass[i]= probability of being in the i-th cell.
    Then the density is roughly p[i] ~= m_i / dx.
    So the Riemann sum yields sum_i m_i log(p[i]) = sum_i m_i log(m_i / dx).
    """
    mass = np.asarray(mass, dtype=float)
    safe_mass = np.maximum(mass, eps) # avoid log(0)
    return float(np.sum(safe_mass * (np.log(safe_mass) - np.log(dx))))


def quantile_from_mass(mass: np.ndarray, edges: np.ndarray, n_quantiles: int = 512) -> np.ndarray:
    """
    Compute Q(u)=F^{-1}(u) from *cell masses* on a grid with edges.

    We assume the density is piecewise-constant on each cell, so the inverse CDF
    can be computed exactly by locating u in the cumulative masses and then
    linearly mapping inside the corresponding cell.
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
    """
    Compute W2^2 between two 1D probability distributions using the quantile formula.
    Both input arrays represent probability masses and must sum to 1.

    In one dimension, the squared 2-Wasserstein distance admits the exact representation
        W2^2 = ∫_0^1 |F_a^{-1}(u) − F_b^{-1}(u)|^2 du,
    where F^{-1} denotes the quantile (generalized inverse CDF).

    Numerically, computing W2^2 in 1D reduces to three steps:
    (i) compute the discrete cumulative distribution functions (CDFs) via cumulative sums,
    (ii) invert the CDFs to obtain the quantile functions F^{-1}(u) (here using interpolation),
    and (iii) approximate the integral over u ∈ (0, 1) by uniform quadrature.

    This is why the code samples many values of u in (0, 1) and interpolates the inverse CDFs
    to evaluate the integral defining W2^2.
    """

    q_a = quantile_from_mass(mass_a, edges, n_quantiles)
    q_b = quantile_from_mass(mass_b, edges, n_quantiles)


    # (q_a - q_b)**2 evaluates the integrand |F_a^{-1}(u) - F_b^{-1}(u)|^2 at the sampled u-values. 
    # Taking np.mean(...) computes
    #     (1/n_quantiles) * sum_k |F_a^{-1}(u_k) - F_b^{-1}(u_k)|^2,
    # which is exactly the midpoint Riemann-sum approximation of
    #     ∫_0^1 |F_a^{-1}(u) - F_b^{-1}(u)|^2 du.
    # The factor du = 1/n_quantiles is implicitly included by taking the mean!
    u = np.linspace(0.0, 1.0, n_quantiles)
    return float(np.trapz((q_a - q_b) ** 2, u))

def build_neumann_laplacian(n: int) -> np.ndarray:
    """ standard numerical PDE machinery: classical finite-difference PDE solver for the Laplacian operator. (NOT JKO SCHEME)
    Finite-difference Laplacian with Neumann (reflecting) i.e;
     We work on the interval x ∈ [0, 1]. In one dimension, the Laplacian is simply the second derivative:
        Δρ = ∂_{xx} ρ.
    Neumann (reflecting) boundary conditions impose zero flux at the boundaries, mathematically-
        ∂_x ρ(0) = 0 and ∂_x ρ(1) = 0.


     So: 
       Δ : rho ↦ ∂_{xx} rho
     On a grid, a function becomes a vector, so L: R--> R is a linear map
     we compute here then L as a matrix, and its application over rho 
     is going to be plainly L @ rho
 
    Lets compute it:
    Discrete Neumann Laplacian stencil (without the 1/dx^2 factor):

    For a grid function ρ = (ρ_0, ..., ρ_{n-1}),

   
                    ρ_{i-1} - 2ρ_i + ρ_{i+1}        for i = 1, ..., n-2   (interior)
     (L ρ)_i =     -2ρ_0     + 2ρ_1                for i = 0            (left boundary)
                    2ρ_{n-2} - 2ρ_{n-1}             for i = n-1          (right boundary)

    These boundary stencils come from enforcing Neumann BCs (∂_x ρ = 0)
    via ghost points. The actual discrete Laplacian is (1/dx^2) * L,
    with the factor 1/dx^2 applied later when the operator is used.
    """


    # initiating our matrix (Laplacian operator)
    L = np.zeros((n, n), dtype=float)
    


    # At the left boundary x = 0 we want to approximate the second derivative ∂_{xx}ρ(0),
    # but the standard centered stencil would involve an out-of-domain value ρ_{-1}.
    # The Neumann boundary condition ∂_xρ(0) = 0 is enforced using a ghost point:
    #     (ρ_1 - ρ_{-1}) / (2 dx) = 0  ⇒  ρ_{-1} = ρ_1.
    # Substituting this into the second-difference formula gives
    #     ∂_{xx}ρ(0) ≈ (ρ_{-1} - 2ρ_0 + ρ_1) / dx^2 = (2ρ_1 - 2ρ_0) / dx^2.
    # The first row of L encodes this stencil (without the 1/dx^2 factor):
    #     (Lρ)_0 = -2ρ_0 + 2ρ_1.
    L[0, 0] = -2.0
    L[0, 1] = 2.0

    # In the continuum, the second derivative (Laplacian in 1D) is approximated
    # by a centered finite difference. At a grid point x_i:
    #
    #   (∂_{xx} rho)(x_i) ≈ (rho(x_{i-1}) - 2*rho(x_i) + rho(x_{i+1})) / dx**2
    #
    # In discrete notation, this is written as:
    #
    #   (Δ rho)_i ≈ (rho_{i-1} - 2*rho_i + rho_{i+1}) / dx**2
    for i in range(1, n - 1):
        L[i, i - 1] = 1.0
        L[i, i] = -2.0
        L[i, i + 1] = 1.0
    
    # At the right boundary x = 1 we want to approximate the second derivative ∂_{xx}ρ(1),
    # but the standard centered stencil would involve an out-of-domain value ρ_{n}.
    # The Neumann boundary condition ∂_xρ(1) = 0 is enforced using a ghost point:
    #     (ρ_n - ρ_{n-2}) / (2 dx) = 0  ⇒  ρ_n = ρ_{n-2}.
    # Substituting this into the second-difference formula gives
    #     ∂_{xx}ρ(1) ≈ (ρ_{n-2} - 2ρ_{n-1} + ρ_n) / dx^2
    #              = (2ρ_{n-2} - 2ρ_{n-1}) / dx^2.
    # The last row of L encodes this stencil (without the 1/dx^2 factor):
    #     (Lρ)_{n-1} = 2ρ_{n-2} - 2ρ_{n-1}.
    L[n - 1, n - 2] = 2.0
    L[n - 1, n - 1] = -2.0
    return L


"""
# --- Why this "implicit Euler with a Laplacian" corresponds to the JKO step for heat ---
#
# Heat equation (with reflecting / Neumann BCs on [0,1]):
#     ∂_t ρ = Δρ,     with ∂_x ρ(0) = ∂_x ρ(1) = 0.
#
# Backward Euler time discretization of the PDE:
#     (ρ^{k+1} - ρ^k) / τ = Δρ^{k+1}
#   ⇔ (I - τΔ) ρ^{k+1} = ρ^k.
#
# JKO scheme (variational time discretization in Wasserstein space):
#     ρ^{k+1} = argmin_ρ {  F(ρ) + (1/(2τ)) W_2^2(ρ, ρ^k)  },
# where for the heat equation the driving functional is the entropy
#     F(ρ) = ∫ ρ log ρ dx.
#
# Key idea: in Wasserstein geometry, admissible first variations of ρ come from
# *transporting* mass. If we perturb ρ by pushing it along a velocity field v,
# then the infinitesimal variation is constrained to the "divergence form"
#     δρ = -∇·(ρ v)    (continuity equation / mass conservation).
#
# For F(ρ)=∫ρ log ρ, the Wasserstein gradient flow PDE is:
#     ∂_t ρ = ∇·(ρ ∇(δF/δρ)) with δF/δρ = 1 + log ρ
#            = ∇·(ρ ∇log ρ)
#            = ∇·(∇ρ)
#            = Δρ,
# i.e. the heat equation.
#
# At the discrete-time (one-step) level, the Euler–Lagrange condition for the
# JKO minimizer implies a discrete continuity equation
#     (ρ^{k+1} - ρ^k)/τ + ∇·(ρ^{k+1} v^{k+1}) = 0,
# together with the optimality relation (for entropy)
#     v^{k+1} = -∇log(ρ^{k+1}).
# Plugging this into the continuity equation gives
#     (ρ^{k+1} - ρ^k)/τ = ∇·(ρ^{k+1} ∇log ρ^{k+1}) = Δρ^{k+1},
# which is exactly the backward Euler update for the heat equation.
#
# Therefore, although the JKO step is defined as a minimization involving W_2,
# for the heat equation it produces the same evolution as solving the implicit
# Euler PDE step. In this code we use the PDE form as a convenient baseline
# (and later, an OT-based "true JKO" implementation can be sanity-checked
# against this implicit heat solver).
"""

def jko_step_Euler(mass_prev: np.ndarray, tau: float, dx: float, laplacian: np.ndarray | None = None) -> np.ndarray:
    """
     Despite the name `jko_step_Euler`, this function does not numerically minimize
     F(m) + (1/(2τ)) W2^2(m, m^k).
    Instead, it exploits the fact that for the heat equation the JKO minimizer
    coincides with the backward Euler discretization of the PDE, and implements
    that implicit PDE step directly. In this sense, this is a classical implicit
    diffusion step used as a proxy for JKO in the heat case.


    For pure diffusion the minimizer solves the implicit Euler update
        (I - τ Δ) ρ^{k+1} = ρ^k.
    Here we work with cell masses m = ρ dx, so the discrete system becomes
        (I - τ/dx^2 L) m^{k+1} = m^k, (because the factor dx is at both sides of the equation)
    with L the Neumann Laplacian.
    """
    mass_prev = np.asarray(mass_prev, dtype=float) #previous mass
    n = mass_prev.size #number of cells 
    L = laplacian if laplacian is not None else build_neumann_laplacian(n)

    A = np.eye(n) - (tau / (dx * dx)) * L #A is the matrix of the system    
    mass_next = np.linalg.solve(A, mass_prev) #solving the system for the next mass

    # Enforce positivity and conservation of mass.
    mass_next = np.clip(mass_next, 0.0, None) #ensuring the mass is positive
    total = mass_next.sum() #ensuring the mass is conserved
    if total <= 0.0:
        raise ValueError("Mass vanished; consider reducing the time step.")
    mass_next /= total
    return mass_next


# This function evaluates the discrete JKO objective
#     J_tau(m | m^k) = F(m) + (1/(2τ)) W2^2(m, m^k)- 
# which is the quantity that the JKO scheme says should be minimized at each step!

# Remind that F(m) is the entropy and W2 is the 2-Wasserstein distance.
# It represents the variational functional that the JKO scheme minimizes
# at each time step. In the current code this is not used to evolve the
# solution; it is provided for diagnostics and as a reference for a future
# implementation of a true (OT-based) JKO minimization.

def jko_objective(mass: np.ndarray, mass_prev: np.ndarray, tau: float, dx: float, x: np.ndarray) -> float:
    """Evaluate the discrete JKO functional at `mass`."""
    # Note: x here is expected to be edges for the W2 computation.
    return entropy(mass, dx) + 0.5 / tau * w2_squared_1d(mass, mass_prev, x)


def initialize_mass(x: np.ndarray,alpha: float = 0.6, beta: float = 0.4, sigma1: float = 0.01, sigma2: float = 0.02) -> np.ndarray:
    """Non-uniform initial probability on [0,1] (mixture of bumps)."""
    density = alpha * np.exp(-((x - 0.3) ** 2) / sigma1) + beta * np.exp(-((x - 0.7) ** 2) / sigma2)
    mass = density
    mass /= mass.sum() #normalizing the mass (dx cancels on uniform grid)
    return mass

def run_simulation_Euler(
    n_points: int = 200,
    tau: float = 5e-4,
    steps: int = 200,
    snapshot_every: int = 25,
    alpha: float = 0.6,
    beta: float = 0.4,
    sigma1: float = 0.01,
    sigma2: float = 0.02,
):
    x, dx, edges = make_grid(n_points)
    mass = initialize_mass(x, alpha, beta, sigma1, sigma2)
    L = build_neumann_laplacian(n_points)

    snapshots = []
    times = []
    for k in range(steps + 1):
        if k % snapshot_every == 0:
            snapshots.append(mass.copy())
            times.append(k * tau)
        if k < steps:
            mass = jko_step_Euler(mass, tau=tau, dx=dx, laplacian=L)
    return x, dx, snapshots, times


def plot_snapshots(
    x: np.ndarray,
    dx: float,
    snapshots: list[np.ndarray],
    times: list[float],
    title: str = "Density snapshots",
    save_path: str | None = None,
    show: bool = True,
):
    fig = plt.figure(figsize=(7, 4))
    for mass, t in zip(snapshots, times):
        plt.plot(x, mass / dx, label=f"t = {t:.3f}")
    plt.xlabel("x")
    plt.ylabel("density ρ(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_snapshots_side_by_side(
    x: np.ndarray,
    dx: float,
    snapshots_left: list[np.ndarray],
    times_left: list[float],
    snapshots_right: list[np.ndarray],
    times_right: list[float],
    title_left: str = "Euler",
    title_right: str = "JKO (quantile)",
):
    """Compare two solution families on shared axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax_l, ax_r = axes

    for mass, t in zip(snapshots_left, times_left):
        ax_l.plot(x, mass / dx, label=f"t = {t:.3f}")
    ax_l.set_title(title_left)
    ax_l.set_xlabel("x")
    ax_l.set_ylabel("density ρ(x)")
    ax_l.legend()

    for mass, t in zip(snapshots_right, times_right):
        ax_r.plot(x, mass / dx, label=f"t = {t:.3f}")
    ax_r.set_title(title_right)
    ax_r.set_xlabel("x")
    ax_r.legend()

    fig.suptitle("Heat flow: Euler vs JKO (quantile)")
    fig.tight_layout()
    plt.show()



# NOW LETS ADD THE JKO SCHEME
"""
What the “true JKO” algorithm is doing (quantile formulation, 1D)

In theory, one JKO step is the infinite-dimensional minimization

    J_τ(m | m^k) = F(m) + (1 / (2τ)) W_2^2(m, m^k),

where F is a functional on probability densities (here
    F(ρ) = ∫ ρ(x) log ρ(x) dx ).
The minimization is over all admissible probability densities m (or ρ).

In one spatial dimension, it is convenient to rewrite this problem in terms
of the quantile function

    Q(u) = F^{-1}(u),   u ∈ (0, 1),

which is monotone increasing.

Lets translate our JKO scheme to the quantile coordinates
Why the entropy becomes a log(Q') term:
--------------------------------------

Let ρ be a positive density with cumulative distribution function

    F(x) = ∫_{-∞}^x ρ(s) ds,

and quantile function Q(u) = F^{-1}(u).
By definition, F(Q(u)) = u. Differentiating gives

    F'(Q(u)) · Q'(u) = 1.

Since F'(x) = ρ(x), this implies the key identity

    ρ(Q(u)) = 1 / Q'(u).

Using the change of variables u = F(x), for which du = ρ(x) dx,
the entropy functional can be rewritten as

    ∫ ρ(x) log ρ(x) dx
      = ∫_0^1 log(ρ(Q(u))) du
      = ∫_0^1 log(1 / Q'(u)) du
      = -∫_0^1 log Q'(u) du.

Thus, in 1D quantile coordinates, the entropy is (up to an additive constant)
the integral of -log Q'(u).

Wasserstein term in quantile variables:
---------------------------------------

In one dimension, the squared Wasserstein distance simplifies to

    W_2^2(m, m^k) = ∫_0^1 |Q(u) - Q^k(u)|^2 du.

Discrete formulation:
---------------------

We discretize u ∈ (0, 1) on a uniform grid u_i and represent Q(u) by a finite
vector

    Q = (Q_0, ..., Q_{n-1}) ∈ R^n,

with the monotonicity constraint

    Q_{i+1} > Q_i.

With Δu = 1 / n, the discrete JKO functional becomes

    J(Q)
      = -∑_{i=0}^{n-2} log(Q_{i+1} - Q_i)
        + (Δu / (2τ)) ∑_{i=0}^{n-1} (Q_i - Q_i^k)^2.

Here the entropy term comes from approximating

    Q'(u_i) ≈ (Q_{i+1} - Q_i) / Δu,

and the additive constant involving log(Δu) drops out of the minimization.

This discrete objective is strictly convex on the admissible set
{Q_{i+1} > Q_i}, so it has a unique minimizer.


Newton’s method and tridiagonal structure:
-------------------------------------------

To find this minimizer, Newton’s method is applied:

    H(Q) p = -∇J(Q),

where H(Q) is the Hessian of J.
Because J only couples neighboring differences (Q_{i+1} - Q_i),
the Hessian is tridiagonal.

Each Newton step therefore reduces to solving a tridiagonal linear system,
which can be done efficiently in O(n) time using the Thomas algorithm.

Meaning of the solution:
------------------------

Conceptually, there are infinitely many admissible quantile functions Q(u).
The algorithm searches for the best one within a finite-dimensional
approximation. If Newton’s method converges, it finds the unique minimizer
of the discrete JKO problem. As the u-grid is refined (n → ∞), this discrete
minimizer converges to the true infinite-dimensional JKO minimizer.

The resulting Q^{k+1} is then mapped back to a density (or masses on an
x-grid) to obtain the next step m^{k+1}.
"""



def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve tridiagonal system with lower diag a (len n-1), main diag b (len n),
    upper diag c (len n-1), rhs d (len n).
    """
    n = len(b)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))

    # Safety: if the main diagonal has zeros/NaNs, fall back to dense solve.
    if not np.all(np.isfinite(bc)) or np.any(np.abs(bc) < 1e-14):
        M = np.zeros((n, n), dtype=float)
        np.fill_diagonal(M, bc)
        np.fill_diagonal(M[1:], ac)
        np.fill_diagonal(M[:, 1:], cc)
        return np.linalg.solve(M, dc)

    for i in range(1, n):
        if abs(bc[i - 1]) < 1e-14 or not np.isfinite(bc[i - 1]):
            return np.linalg.solve(
                np.diag(bc) + np.diag(ac, -1) + np.diag(cc, 1),
                dc,
            )
        w = ac[i - 1] / bc[i - 1]
        bc[i] -= w * cc[i - 1]
        dc[i] -= w * dc[i - 1]
    x = np.zeros(n)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        if abs(bc[i]) < 1e-14 or not np.isfinite(bc[i]):
            return np.linalg.solve(
                np.diag(bc) + np.diag(ac, -1) + np.diag(cc, 1),
                dc,
            )
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


def _project_increasing_fixed_ends(Q: np.ndarray, lo: float, hi: float, min_gap: float) -> np.ndarray:
    """
    Enforce Q[0]=lo, Q[-1]=hi, and Q strictly increasing with gaps >= min_gap
    using a forward/backward pass. If infeasible, fall back to uniform spacing.
    """
    Qp = np.asarray(Q, float).copy()
    n = Qp.size
    Qp[0] = lo
    Qp[-1] = hi
    # forward
    for i in range(1, n):
        Qp[i] = max(Qp[i], Qp[i - 1] + min_gap)
    # backward (keep end fixed)
    for i in range(n - 2, -1, -1):
        Qp[i] = min(Qp[i], Qp[i + 1] - min_gap)
    # If min_gap too large, reset
    if lo + (n - 1) * min_gap > hi:
        Qp = np.linspace(lo, hi, n)
    return Qp


def jko_step_entropy_1d_via_quantiles(
    Q_prev: np.ndarray,
    tau: float,
    max_newton: int = 200,
    tol: float = 1e-11,
    min_gap: float = 1e-9,
    x_bounds: tuple[float, float] | None = (0.0, 1.0),
) -> np.ndarray:
    """
    True JKO minimization for F(ρ)=∫ρ log ρ in 1D, in quantile coordinates.

    Minimize over increasing Q:
        J(Q) = Δu * [ -∑ log(Q_{i+1}-Q_i) ] + (Δu/(2τ)) ∑ (Q_i - Q_prev_i)^2

    Returns Q_next.
    """
    # This solver assumes Q is sampled on a u-grid INCLUDING endpoints u=0 and u=1.
    # We enforce Q(0)=lo and Q(1)=hi to reflect the bounded domain.
    Q_prev = np.asarray(Q_prev, float)
    n = Q_prev.size
    if n < 3:
        raise ValueError("Need at least 3 quantile points (including endpoints).")
    lo, hi = (0.0, 1.0) if x_bounds is None else x_bounds
    du = 1.0 / (n - 1)
    # trapezoid weights for integral in u
    w = np.ones(n, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5

    Q_prev = _project_increasing_fixed_ends(Q_prev, lo=lo, hi=hi, min_gap=min_gap)
    Q = Q_prev.copy()

    # Numerical safety for tiny gaps to avoid inf/NaN in Hessian.
    min_d_safe = max(min_gap, 1e-12)

    def objective(Qv: np.ndarray) -> float:
        d = np.diff(Qv)
        if np.any(d <= 0):
            return np.inf
        d = np.maximum(d, min_d_safe)
        ent = du * (-np.sum(np.log(d)))  # constant -du*sum(log du) omitted
        quad = (du / (2.0 * tau)) * float(np.sum(w * (Qv - Q_prev) ** 2))
        return float(ent + quad)

    def gradient(Qv: np.ndarray) -> np.ndarray:
        """Gradient of the discrete functional at Qv (assumes strictly increasing)."""
        d = np.maximum(np.diff(Qv), min_d_safe)
        g = (du / tau) * (w * (Qv - Q_prev))
        g[0] += du / d[0]
        g[-1] += -du / d[-1]
        if n > 2:
            g[1:-1] += du * ((-1.0 / d[:-1]) + (1.0 / d[1:]))
        return g

    J_old = objective(Q)

    m = n - 2  # interior unknowns
    for it in range(max_newton):
        Q = _project_increasing_fixed_ends(Q, lo=lo, hi=hi, min_gap=min_gap)
        d = np.maximum(np.diff(Q), min_d_safe)

        g_full = gradient(Q)
        g = g_full[1:-1]
        grad_norm = np.linalg.norm(g, ord=np.inf)
        if grad_norm < tol:
            break

        # Hessian on interior is tridiagonal.
        inv_d2 = du * (1.0 / (d * d))
        inv_d2 = np.minimum(inv_d2, 1e12)
        a = np.zeros(m - 1)
        b = np.zeros(m)
        c = np.zeros(m - 1)
        for j in range(m):
            i = j + 1  # global index
            b[j] = (du / tau) * w[i] + inv_d2[i - 1] + inv_d2[i]
            if j > 0:
                a[j - 1] = -inv_d2[i - 1]
            if j < m - 1:
                c[j] = -inv_d2[i]
        p_int = solve_tridiagonal(a, b, c, -g)

        # Backtracking line search to maintain strict monotonicity and decrease J
        step = 1.0
        while step > 1e-12:
            Q_try = Q.copy()
            Q_try[1:-1] = Q_try[1:-1] + step * p_int
            Q_try = _project_increasing_fixed_ends(Q_try, lo=lo, hi=hi, min_gap=min_gap)
            J_try = objective(Q_try)
            if J_try <= J_old - 1e-4 * step * float(np.dot(g, p_int)):
                Q = Q_try
                J_old = J_try
                break
            step *= 0.5

        if step <= 1e-12:
            # Keep old behavior: return last feasible iterate (avoids crashing).
            return Q

    return Q


def mass_from_quantile(Q: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Convert quantiles Q(u_i) (u INCLUDING endpoints) into *cell masses* on the
    cell grid defined by `edges`, using CDF differences:
        m_i = F(edges[i+1]) - F(edges[i]).
    """
    Q = np.asarray(Q, float)
    n = Q.size
    u = np.linspace(0.0, 1.0, n)

    u_edges = np.interp(edges, Q, u, left=0.0, right=1.0)
    u_edges = np.clip(u_edges, 0.0, 1.0)
    u_edges = np.maximum.accumulate(u_edges)  # ensure monotone
    mass = np.diff(u_edges)
    mass = np.clip(mass, 0.0, None)

    # Small uniform floor, then renormalize
    eps = 1e-12
    mass += eps
    total = mass.sum()
    if total <= 0.0:
        raise RuntimeError("Reconstructed mass vanished; check quantiles.")
    mass /= total
    return mass


def jko_step_quantile(
    mass_prev: np.ndarray,
    edges: np.ndarray,
    tau: float,
    n_quantiles: int = 256,
    max_newton: int = 200,
    tol: float = 1e-11,
    min_gap: float = 1e-9,
) -> np.ndarray:
    """Wrapper to perform one JKO step via quantile Newton solve."""
    Q_prev = quantile_from_mass(mass_prev, edges, n_quantiles)
    Q_next = jko_step_entropy_1d_via_quantiles(
        Q_prev,
        tau=tau,
        max_newton=max_newton,
        tol=tol,
        min_gap=min_gap,
        x_bounds=(edges[0], edges[-1]),
    )
    mass_next = mass_from_quantile(Q_next, edges)
    mass_next = np.clip(mass_next, 0.0, None)
    mass_next /= mass_next.sum()
    return mass_next


def run_simulation_quantile(
    n_points: int = 200,
    tau: float = 5e-4,
    steps: int = 200,
    snapshot_every: int = 25,
    n_quantiles: int = 256,
    alpha: float = 0.6,
    beta: float = 0.4,
    sigma1: float = 0.01,
    sigma2: float = 0.02,
):
    x, dx, edges = make_grid(n_points)
    mass = initialize_mass(x, alpha, beta, sigma1, sigma2)

    snapshots = []
    times = []
    for k in range(steps + 1):
        if k % snapshot_every == 0:
            snapshots.append(mass.copy())
            times.append(k * tau)
        if k < steps:
            mass = jko_step_quantile(
                mass,
                edges=edges,
                tau=tau,
                n_quantiles=n_quantiles,
            )
    return x, dx, edges, snapshots, times




def main():
    # Parameters shared by both runs (so filenames can encode them)
    alpha = 0.6
    beta = 0.4
    sigma1 = 0.01
    sigma2 = 0.02
    n_points = 400
    tau = 2e-4
    steps = 300
    snapshot_every = 50



##################################################################################
#            MAIN: RUNNING THE SIMULATIONS
##################################################################################

if __name__ == "__main__":
    # Parameters shared by both runs (so filenames can encode them)
    alpha = 0.9
    beta = 0.4
    sigma1 = 0.09
    sigma2 = 0.1
    n_points = 400
    tau = 2e-4
    steps = 300
    snapshot_every = 50

    x, dx, snapshots, times = run_simulation_Euler(
        n_points=n_points,
        tau=tau,
        steps=steps,
        snapshot_every=snapshot_every,
        alpha=alpha,
        beta=beta,
        sigma1=sigma1,
        sigma2=sigma2,
    )
    euler_fname = f"plot_euler_alpha{alpha}_beta{beta}.png"
    plot_snapshots(
        x,
        dx,
        snapshots,
        times,
        title="Euler (implicit heat step)",
        save_path=euler_fname,
        show=False,
    )

    xq, dxq, edges_q, snapshots_q, times_q = run_simulation_quantile(
        n_points=n_points,
        tau=tau,
        steps=steps,
        snapshot_every=snapshot_every,
        alpha=alpha,
        beta=beta,
        sigma1=sigma1,
        sigma2=sigma2,
    )
    jko_fname = f"plot_jko_alpha{alpha}_beta{beta}.png"
    plot_snapshots(
        xq,
        dxq,
        snapshots_q,
        times_q,
        title="JKO (quantile)",
        save_path=jko_fname,
        show=False,
    )

