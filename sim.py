"""QRN toy model: generate Figures 2–4 (PNG + SVG).

This script generates the numerical simulations used for Fig. 2–4 in the QRN draft.

Model (toy, GKSL/Lindblad-consistent):
- Graph: chain of N nodes (hypotheses).
- Hamiltonian: H = -gamma * L + V, where L is the graph Laplacian and V is a fixed
  diagonal potential (proxy for local free energy / prediction error).
- Open dynamics (Lindblad):
  * Pure dephasing on site projectors |i><i| at rate kappa.
  * Irreversible sink/readout on the target node at rate eta.

Implementation detail:
- The sink is implemented as an explicit additional basis state. This keeps Tr(rho)=1,
  and the success probability is simply the sink population.
- Time propagation uses expm_multiply (action of the matrix exponential) instead of
  solve_ivp, improving numerical stability and reproducibility.

Outputs (kept identical to legacy names expected by the manuscript):
- Fig2_QRN_Targeting_300dpi.png / .svg
- Fig3_QRN_ExpectedPotential_300dpi.png / .svg
- Fig4_QRN_EfficiencyLandscape_300dpi.png / .svg

Figure 4 plotting fixes:
- The heatmap extent is expanded by half a grid step so grid points correspond to pixel
  centers.
- Axes limits are padded so the maximum marker is comfortably inside the frame.
- The maximum annotation is auto-positioned so it never overlaps the colorbar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import expm_multiply


# ======================
# Parameters (Table 1)
# ======================
N = 10
T_HORIZON = 20.0
GAMMA_DEFAULT = 2.0
ETA = 0.8
TARGET = N - 1


# ----------------------
# Reproducible potential
# ----------------------
np.random.seed(42)
epsilon = np.random.uniform(-0.5, 0.5, N)
V_pot = 0.3 * (np.arange(N)[::-1] + epsilon)
V_pot = V_pot - np.min(V_pot)


def laplacian_chain(n: int) -> np.ndarray:
    """Graph Laplacian L = D - A for an undirected chain of n nodes."""
    if n < 2:
        raise ValueError("n must be >= 2")
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


# ----------------------
# Linear-algebra helpers
# ----------------------

def vecF(M: np.ndarray) -> np.ndarray:
    """Column-stacking vec operator (Fortran order)."""
    return np.asarray(M, dtype=complex).reshape((-1,), order="F")


def matF(v: np.ndarray, dim: int) -> np.ndarray:
    """Inverse of vecF: reshape vector into (dim, dim) (Fortran order)."""
    return np.asarray(v, dtype=complex).reshape((dim, dim), order="F")


def embed_system_operator(op_sys: np.ndarray, dim_sink: int = 1) -> np.ndarray:
    """Embed an NxN system operator into (N+dim_sink)x(N+dim_sink) with zeros on sink."""
    n = op_sys.shape[0]
    d = n + dim_sink
    out = np.zeros((d, d), dtype=complex)
    out[:n, :n] = np.asarray(op_sys, dtype=complex)
    return out


def projector(dim: int, i: int) -> np.ndarray:
    """Projector |i><i| in dimension dim."""
    P = np.zeros((dim, dim), dtype=complex)
    P[i, i] = 1.0
    return P


def commutator_superop(H: np.ndarray) -> np.ndarray:
    """Superoperator for -i[H, rho] under vecF convention."""
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    return (-1j) * (np.kron(I, H) - np.kron(H.T, I))


def dissipator_superop(L: np.ndarray) -> np.ndarray:
    """Superoperator for D_L[rho] = L rho L† - 1/2 {L†L, rho} under vecF convention."""
    d = L.shape[0]
    I = np.eye(d, dtype=complex)
    LdL = L.conj().T @ L
    return np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I)


def precompute_liouvillian_components(
    n: int,
    V: np.ndarray,
    target: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute components so that L = gamma*C_L + C_V + kappa*D_deph + eta*D_sink."""
    d = n + 1
    sink = n

    # Hamiltonian pieces (embedded; sink has zero Hamiltonian)
    H_L = embed_system_operator(-laplacian_chain(n))  # corresponds to gamma * (-L)
    H_V = embed_system_operator(np.diag(V))

    C_L = commutator_superop(H_L)
    C_V = commutator_superop(H_V)

    # Dephasing base: sum_i D_{|i><i|}
    D_deph = np.zeros((d * d, d * d), dtype=complex)
    for i in range(n):
        D_deph += dissipator_superop(projector(d, i))

    # Sink base: Lindblad operator |sink><target| (rate=1)
    Ls = np.zeros((d, d), dtype=complex)
    Ls[sink, target] = 1.0
    D_sink = dissipator_superop(Ls)

    return C_L, C_V, D_deph, D_sink


def propagate_density_timeseries(
    L_super: np.ndarray,
    rho0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Return rho(t) for all t in 'times' using expm_multiply."""
    d = rho0.shape[0]
    v0 = vecF(rho0)

    t0 = float(times[0])
    t1 = float(times[-1])
    num = int(len(times))

    # expm_multiply supports evenly-spaced sampling via start/stop/num
    if not np.allclose(times, np.linspace(t0, t1, num)):
        raise ValueError("'times' must be an evenly spaced linspace")

    vt = expm_multiply(L_super, v0, start=t0, stop=t1, num=num, endpoint=True)  # (num, d^2)

    rhos = np.empty((num, d, d), dtype=complex)
    for k in range(num):
        rhos[k] = matF(vt[k], d)
    return rhos


def propagate_density_final(L_super: np.ndarray, rho0: np.ndarray, T: float) -> np.ndarray:
    """Return rho(T)."""
    d = rho0.shape[0]
    v0 = vecF(rho0)
    vT = expm_multiply(L_super * float(T), v0)
    return matF(vT, d)


def crw_generator_chain_with_sink(n: int, gamma: float, eta: float, target: int) -> np.ndarray:
    """Classical continuous-time random walk generator for chain + absorbing sink."""
    d = n + 1
    sink = n
    L = laplacian_chain(n)

    Q = np.zeros((d, d), dtype=float)
    Q[:n, :n] = -gamma * L

    # Flow target -> sink
    Q[target, target] -= eta
    Q[sink, target] += eta

    return Q


def propagate_prob_timeseries(Q: np.ndarray, p0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return p(t) for all t in 'times' using expm_multiply."""
    p0 = np.asarray(p0, dtype=float).reshape((-1,))

    t0 = float(times[0])
    t1 = float(times[-1])
    num = int(len(times))

    if not np.allclose(times, np.linspace(t0, t1, num)):
        raise ValueError("'times' must be an evenly spaced linspace")

    pt = expm_multiply(Q, p0, start=t0, stop=t1, num=num, endpoint=True)  # (num, d)
    return np.asarray(pt, dtype=float)


def expected_potential_conditional(rho: np.ndarray, V: np.ndarray) -> float:
    """E[V] conditioned on not yet being in the sink: Tr(rho_sys V) / Tr(rho_sys)."""
    n = V.shape[0]
    rho_sys = rho[:n, :n]
    tr_sys = float(np.real(np.trace(rho_sys)))
    if tr_sys <= 1e-12:
        return 0.0
    num = float(np.real(np.trace(rho_sys @ np.diag(V))))
    return num / tr_sys


def main() -> None:
    out_dir = Path(".").resolve()

    # Global readability
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    # Time grid (Figures 2–3)
    times = np.linspace(0.0, T_HORIZON, 200)

    # Dimensions (system + sink)
    d = N + 1
    sink = N

    # Initial density matrix: |0><0|
    rho0 = np.zeros((d, d), dtype=complex)
    rho0[0, 0] = 1.0 + 0j

    # Precompute Liouvillian components
    C_L, C_V, D_deph, D_sink = precompute_liouvillian_components(N, V_pot, TARGET)

    # Base part independent of (gamma, kappa)
    L_base = C_V + ETA * D_sink

    def L_super(gamma: float, kappa: float) -> np.ndarray:
        return L_base + float(gamma) * C_L + float(kappa) * D_deph

    # -----------------
    # Fig 2–3 time series
    # -----------------
    rhos_coh = propagate_density_timeseries(L_super(GAMMA_DEFAULT, 0.0), rho0, times)
    rhos_qrn = propagate_density_timeseries(L_super(GAMMA_DEFAULT, 1.0), rho0, times)

    p_coh = np.real(rhos_coh[:, sink, sink])
    p_qrn = np.real(rhos_qrn[:, sink, sink])

    ev_coh = np.array([expected_potential_conditional(r, V_pot) for r in rhos_coh], dtype=float)
    ev_qrn = np.array([expected_potential_conditional(r, V_pot) for r in rhos_qrn], dtype=float)

    # Classical RW baseline (same chain + sink)
    Q = crw_generator_chain_with_sink(N, GAMMA_DEFAULT, ETA, TARGET)
    p0 = np.zeros(d, dtype=float)
    p0[0] = 1.0
    pt = propagate_prob_timeseries(Q, p0, times)
    p_crw = pt[:, sink]

    ev_crw = np.zeros_like(p_crw)
    for k in range(len(times)):
        p_sys = pt[k, :N]
        rem = float(np.sum(p_sys))
        ev_crw[k] = float(np.dot(p_sys, V_pot) / rem) if rem > 1e-12 else 0.0

    # --------
    # Figure 2
    # --------
    plt.figure(figsize=(8, 5))
    plt.plot(times, p_coh, label="Coherent ($\\kappa=0$)", linestyle="--", linewidth=2)
    plt.plot(times, p_qrn, label="QRN Netting ($\\kappa=1$)", linewidth=3)
    plt.plot(times, p_crw, label="Classical RW", linestyle=":", linewidth=2)
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Cumulative Probability at Sink")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "Fig2_QRN_Targeting_300dpi.png", dpi=300)
    plt.savefig(out_dir / "Fig2_QRN_Targeting.svg")
    plt.close()

    # --------
    # Figure 3
    # --------
    plt.figure(figsize=(8, 5))
    plt.plot(times, ev_coh, label="Coherent ($\\kappa=0$)", linestyle="--", linewidth=2)
    plt.plot(times, ev_qrn, label="QRN Netting ($\\kappa=1$)", linewidth=3)
    plt.plot(times, ev_crw, label="Classical RW", linestyle=":", linewidth=2, alpha=0.8)
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Expected Potential (conditional on not-yet-readout)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "Fig3_QRN_ExpectedPotential_300dpi.png", dpi=300)
    plt.savefig(out_dir / "Fig3_QRN_ExpectedPotential.svg")
    plt.close()

    # -----------------
    # Figure 4: heatmap
    # -----------------
    gammas = np.linspace(0.5, 4.0, 20)
    kappas = np.linspace(0.0, 2.5, 20)
    heatmap = np.zeros((len(kappas), len(gammas)), dtype=float)

    v0 = vecF(rho0)
    sink_vec_index = sink + sink * d  # index of (sink, sink) in vecF

    for i, kappa in enumerate(kappas):
        for j, gamma in enumerate(gammas):
            Lgk = L_super(gamma, kappa)
            vT = expm_multiply(Lgk * float(T_HORIZON), v0)
            heatmap[i, j] = float(np.real(vT[sink_vec_index]))

    # Locate maximum for annotation/marker
    i_max, j_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    max_kappa = float(kappas[i_max])
    max_gamma = float(gammas[j_max])
    max_val = float(heatmap[i_max, j_max])

    # Corrected extent: align grid points with pixel centers
    dg = float(gammas[1] - gammas[0]) if len(gammas) > 1 else 1.0
    dk = float(kappas[1] - kappas[0]) if len(kappas) > 1 else 1.0
    extent = [
        float(gammas[0] - 0.5 * dg),
        float(gammas[-1] + 0.5 * dg),
        float(kappas[0] - 0.5 * dk),
        float(kappas[-1] + 0.5 * dk),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$P_{success}(T)$")

    ax.set_xlabel("Diffusion / Connectivity $\\gamma$")
    ax.set_ylabel("Dephasing / Noise $\\kappa$")

    # Mark maximum
    ax.scatter(
        [max_gamma],
        [max_kappa],
        s=80,
        marker="o",
        edgecolors="k",
        linewidths=1.0,
        zorder=5,
    )

    # Auto-place annotation so it stays inside the main axes and never overlaps the colorbar.
    # Heuristic: if argmax is on the right half, place text to the *left* of the marker (ha='right').
    # If argmax is on the left half, place it to the right (ha='left'). Similarly for vertical.
    gx_mid = 0.5 * (extent[0] + extent[1])
    ky_mid = 0.5 * (extent[2] + extent[3])

    dx = -10 if max_gamma >= gx_mid else 10
    ha = "right" if dx < 0 else "left"

    dy = -10 if max_kappa >= ky_mid else 10
    va = "top" if dy < 0 else "bottom"

    label = f"max={max_val:.3f}\n(γ={max_gamma:.2f}, κ={max_kappa:.2f})"
    ax.annotate(
        label,
        xy=(max_gamma, max_kappa),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9),
        arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8, shrinkA=0, shrinkB=6),
        zorder=6,
    )

    # Axis padding: ensures the max marker doesn't sit on the plot frame
    # (even if optimum ends up at the boundary of the scanned box).
    pad_g = max(0.08 * (gammas[-1] - gammas[0]), 1.5 * dg)
    pad_k = max(0.08 * (kappas[-1] - kappas[0]), 1.5 * dk)
    ax.set_xlim(extent[0] - pad_g, extent[1] + pad_g)
    ax.set_ylim(extent[2] - pad_k, extent[3] + pad_k)

    plt.tight_layout()
    fig.savefig(out_dir / "Fig4_QRN_EfficiencyLandscape_300dpi.png", dpi=300)
    fig.savefig(out_dir / "Fig4_QRN_EfficiencyLandscape.svg")
    plt.close(fig)

    print("Done. Figures 2–4 were generated.")


main()
