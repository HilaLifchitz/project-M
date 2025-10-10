#!/usr/bin/env python3

"""
Two-locus Wright-Fisher Markov chain utilities (independent loci).

This module leverages the single-locus `WrightFisherMarkovChain` from `Markov_1_locus.py`
to construct and analyze a two-locus process under the assumption that the loci evolve
independently (no epistasis or linkage used here). The joint distribution is thus the
outer product of identical marginals.
"""

from __future__ import annotations

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from Markov_1_locus import WrightFisherMarkovChain


class MarkovTwoLoci:
    """
    Two-locus wrapper using the single-locus Wright-Fisher Markov chain.

    Assumes both loci share N and μ, but can have distinct s1 and s2, and evolve independently.
    Provides:
    - Access to the single-locus transition matrix P and stationary distribution π
    - Evolution of a marginal distribution via repeated applications of P
    - Construction of a joint 2D distribution as outer product when independence holds
    - Plotting a 2D heatmap given a marginal π (outer product π ⊗ π)
    """

    def __init__(self, N: int, mu: float, s1: float, s2: float, quiet: bool = True):
        self.N = int(N)
        self.mu = float(mu)
        self.s1 = float(s1)
        self.s2 = float(s2)
        self.quiet = quiet

        self.chain1 = WrightFisherMarkovChain(N=self.N, s=self.s1, mu=self.mu, quiet=self.quiet)
        self.chain2 = WrightFisherMarkovChain(N=self.N, s=self.s2, mu=self.mu, quiet=self.quiet)
        self.P1 = None
        self.P2 = None
        self.pi1_star = None
        self.pi2_star = None

    def construct_transition_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        if self.P1 is None:
            self.P1 = self.chain1.construct_transition_matrix()
        if self.P2 is None:
            self.P2 = self.chain2.construct_transition_matrix()
        return self.P1, self.P2

    def stationary_distributions(self, method: str = 'direct') -> tuple[np.ndarray, np.ndarray]:
        if self.pi1_star is None or self.P1 is None:
            self.construct_transition_matrices()
            self.pi1_star = np.asarray(self.chain1.compute_stationary_distribution(method=method), dtype=float)
        if self.pi2_star is None or self.P2 is None:
            self.construct_transition_matrices()
            self.pi2_star = np.asarray(self.chain2.compute_stationary_distribution(method=method), dtype=float)
        return self.pi1_star, self.pi2_star

    # def evolve_marginals(self, pi1_0: np.ndarray, pi2_0: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    #     """Evolve both marginals independently for T steps: π_T = P^T π_0 for each locus."""
    #     P1, P2 = self.construct_transition_matrices()
    #     pi1 = np.asarray(pi1_0, dtype=float)
    #     pi2 = np.asarray(pi2_0, dtype=float)
    #     pi1 = pi1 / np.sum(pi1)
    #     pi2 = pi2 / np.sum(pi2)
    #     for _ in range(int(T)):
    #         pi1 = P1 @ pi1
    #         pi2 = P2 @ pi2
    #         pi1 = pi1 / np.sum(pi1)
    #         pi2 = pi2 / np.sum(pi2)
    #     return pi1, pi2

    def joint_from_marginals(self, pi1: np.ndarray, pi2: np.ndarray) -> np.ndarray:
        """Return joint distribution under independence: Π = π1 ⊗ π2 (outer product)."""
        pi1 = np.asarray(pi1, dtype=float)
        pi2 = np.asarray(pi2, dtype=float)
        pi1 = pi1 / np.sum(pi1)
        pi2 = pi2 / np.sum(pi2)
        return np.outer(pi1, pi2)

    def plot_joint_heatmap(self, pi1: np.ndarray, pi2: np.ndarray, title: str | None = None) -> plt.Figure:
        """Plot a 2D heatmap of Π = π1 ⊗ π2 (independent loci)."""
        sns.set_style("white")
        sns.set_palette("husl")

        joint = self.joint_from_marginals(pi1, pi2)
        states = np.arange(self.N + 1)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        im = ax.imshow(joint, cmap='Blues', origin='lower', aspect='equal', interpolation='nearest')
        ax.set_xlabel('Locus 1: number of "1" alleles')
        ax.set_ylabel('Locus 2: number of "1" alleles')
        if title is None:
            title = (
                f'Two-Locus Joint Distribution (independent)\n'
                f'N={self.N}, μ={self.mu}, s1={self.s1}, s2={self.s2}'
            )
        ax.set_title(title)

        # ticks
        tick_positions = np.arange(0, self.N + 1, max(1, self.N // 5))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions)
        ax.set_yticklabels(tick_positions)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Joint probability')
        plt.tight_layout()
        return fig

    def plot_joint_3d(self, pi1: np.ndarray, pi2: np.ndarray, title: str | None = None,
                      elev: float = 30.0, azim: float = -60.0) -> plt.Figure:
        """
        Plot a 3D histogram (bar3d) of Π = π1 ⊗ π2.

        Parameters:
        -----------
        pi1, pi2 : np.ndarray
            Marginal distributions for locus 1 and locus 2 (length N+1 each)
        title : str | None
            Optional custom title. If None, a default title is constructed
        elev, azim : float
            Elevation and azimuth angles for the 3D view
        """
        joint = self.joint_from_marginals(pi1, pi2)
        states = np.arange(self.N + 1)
        X, Y = np.meshgrid(states, states)
        Xf = X.ravel()
        Yf = Y.ravel()
        Zf = joint.ravel()

        # Bar sizes
        dx = np.full_like(Xf, 0.8, dtype=float)
        dy = np.full_like(Yf, 0.8, dtype=float)
        z0 = np.zeros_like(Zf)

        # Colors based on height
        max_z = np.max(Zf) if np.max(Zf) > 0 else 1.0
        colors = cm.Blues(0.3 + 0.7 * (Zf / max_z))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(Xf - 0.4, Yf - 0.4, z0, dx, dy, Zf, shade=True, color=colors, alpha=0.95)

        ax.set_xlabel('Locus 1: number of "1" alleles')
        ax.set_ylabel('Locus 2: number of "1" alleles')
        ax.set_zlabel('Joint probability')

        if title is None:
            title = (
                f'Two-Locus Joint Distribution (3D)\n'
                f'N={self.N}, μ={self.mu}, s1={self.s1}, s2={self.s2}'
            )
        ax.set_title(title)

        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        return fig


def phi_star(N: int, mu: float, method: str = 'direct') -> np.ndarray:
    """Return 1D stationary distribution at s=0 for given N, mu."""
    chain = WrightFisherMarkovChain(N=int(N), s=0.0, mu=float(mu), quiet=True)
    chain.construct_transition_matrix()
    phi = np.asarray(chain.compute_stationary_distribution(method=method), dtype=float)
    phi = phi / np.sum(phi)
    return phi


def evolve_p_list(P_list: list[np.ndarray], p0_list: list[np.ndarray], T: int) -> list[np.ndarray]:
    """
    Generalized evolution for multiple independent loci.
    For each k, returns p_k(T) = P_k^T @ p_k(0), normalized.

    Parameters:
    -----------
    P_list : list of (N+1)x(N+1) transition matrices
    p0_list : list of initial marginal distributions (length N+1 each)
    T : int generations
    """
    assert len(P_list) == len(p0_list), "P_list and p0_list must have same length"
    pT_list: list[np.ndarray] = []
    for P, p0 in zip(P_list, p0_list):
        p = np.asarray(p0, dtype=float)
        p = p / np.sum(p)
        PT = np.linalg.matrix_power(np.asarray(P, dtype=float), int(T))
        pT = PT @ p
        pT = pT / np.sum(pT)
        pT_list.append(pT)
    return pT_list


def evolve_p1_p2(P1: np.ndarray, P2: np.ndarray, p1_0: np.ndarray, p2_0: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible 2-locus wrapper around evolve_p_list."""
    p1_T, p2_T = evolve_p_list([P1, P2], [p1_0, p2_0], T)
    return p1_T, p2_T


def marginals_from_s(N: int, mu: float, s1: float, s2: float, T: int, method: str = 'direct') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function:
    - Builds P1 (s=s1) and P2 (s=s2) for shared (N, mu)
    - Uses phi_star(N, mu) as initial condition for both marginals
    - Returns p1(T), p2(T) along with P1, P2
    """
    # Transition matrices for the two loci
    chain1 = WrightFisherMarkovChain(N=int(N), s=float(s1), mu=float(mu), quiet=True)
    chain2 = WrightFisherMarkovChain(N=int(N), s=float(s2), mu=float(mu), quiet=True)
    P1 = chain1.construct_transition_matrix()
    P2 = chain2.construct_transition_matrix()

    # Initial condition: neutral stationary distribution
    phi = phi_star(N, mu, method=method)

    # Evolve to time T
    p1_T, p2_T = evolve_p1_p2(P1, P2, phi, phi, T)
    return p1_T, p2_T



def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-16
    p_ = p + eps
    q_ = q + eps
    p_ = p_ / np.sum(p_)
    q_ = q_ / np.sum(q_)
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))) / math.log(2.0))  # bits


# -----------------------------
# Simplex grid enumeration
# -----------------------------

def _compositions(n: int, k: int):
    """
    Yield all k-tuples of nonnegative integers summing to n (integer compositions with k parts).
    This is a generator with no filtering (stars-and-bars recursion).
    Mathematicaly, for any n >= 0 and k >= 2,
              C(n, k) = ⋃_{i=0}^n { (i, r1, ..., r_{k-1}) : (r1, ..., r_{k-1}) ∈ C(n - i, k - 1) }.
    """
    if k == 1:
        #Yield pauses a function's execution and returns a value temporarily
        # any function that contains yield is a generator function. open this as touples_list = list(_compositions(n, k))
        yield (n,) 
        return
    for i in range(n + 1):
        # (i,) is a one-element tuple; 'rest' is one of the (k-1)-tuples from the recursive call.
        # Tuple concatenation (i,) + rest prepends i to each smaller composition,
        # building full k-tuples like (i, r1, r2, ...).
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


def simplex_grid_points(N: int, dim: int):
    """
    Iterate over the gridded simplex of dimension `dim` with grid size 1/N.

    - dim=1 → points on [0,1] as pairs (x, 1-x) with x ∈ {0,1/N,...,1}
    - dim=2 → triples (x,y,z) with x+y+z=1 on the 1/N grid
    - dim=3 → quadruples (A,B,C,D) with A+B+C+D=1 on the 1/N grid

    Yields tuples of length dim+1 with components in {0,1/N,...,1} summing to 1.
    """
    k = int(dim) + 1 # cause we have k+1 coordinates in a k-dimensional simplex

    for comp in _compositions(int(N), k): # comp = e.g. (2, 1, 3) 
        yield tuple(c / float(N) for c in comp)  # -> (2/N, 1/N, 3/N), tuple(...)builds a tuple from any iterable.

# shortcuts functions
def simplex2_grid(N: int): # dim=2, returning a list of tuples
    """Convenience wrapper for dim=2 (triples)."""
    return list(simplex_grid_points(N, dim=2))


def simplex3_grid(N: int): # dim=3, returning a list of tuples
    """Convenience wrapper for dim=3 (quadruples)."""
    return list(simplex_grid_points(N, dim=3))


#########################################################################################################################
# Information theory functions
#########################################################################################################################

def D(T,N,mu,s1,s2): #D(P**2(T)) the information stored by natural selection in the allele frequency space after T generations  
    p1_T, p2_T = marginals_from_s(N, mu, s1, s2, T)
    phi = phi_star(N, mu) # phi is the stationary distribution of the neutral locus which is here always p1_0 = p2_0
    return kl_divergence(p1_T,phi)*kl_divergence(p2_T,phi) # we get to multiply the measures cause of their independece


def I(T,N,mu,s1,s2): #I(T) the free-fitness after T generations, I stands for Iwasa  
    p1_T, p2_T = marginals_from_s(N, mu, s1, s2, T)
    model = MarkovTwoLoci(N=N, mu=mu, s1=s1, s2=s2, quiet=True)
    psi_1, psi_2 = model.stationary_distributions(method='direct') # psi_1 and psi_2 are the stationary distributions of the two loci-- to which we are going to 
    return kl_divergence(p1_T,psi_1)*kl_divergence(p2_T,psi_2) # we get to multiply the measures cause of their independece



    
if __name__ == '__main__':
    N = 40
    mu = 5e-4
    s1 = 0.005
    s2 = 0.010
    model = MarkovTwoLoci(N=N, mu=mu, s1=s1, s2=s2, quiet=True)
    pi1_star, pi2_star = model.stationary_distributions(method='direct')
    phi = phi_star(N, mu)
    print(kl_divergence(pi1_star,phi)*kl_divergence(pi2_star,phi))
    
    #fig_2d = model.plot_joint_heatmap(pi1_star, pi2_star)
    #plt.show()
    #fig_3d = model.plot_joint_3d(pi1_star, pi2_star)
    #plt.show()

    # Nice, this works
    # D_t = []
    # for t in range(0, 5000, 100):
    #     D_t.append(D(t,N,mu,s1,s2))
    # plt.plot(D_t)
    # plt.show()


