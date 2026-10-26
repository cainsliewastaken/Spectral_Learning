"""
Figures illustrating Section 6 (scalar-input, scalar-output linear regression)
of "A curious case in the dynamics of spectral bias".

Three conceptual moves are illustrated:

  (A) Shallow scalar linear model  -> one effective coordinate, a single
      geometric contraction factor; failure on oscillatory targets is
      representational, not spectral-dynamical.  (Sec 6.1-6.3)

  (B) Deep / factorized linear model -> same function class as the shallow
      model, but depth reshapes the optimization geometry.  (Sec 6.4-6.9)

  (C) Fixed sine-basis model -> spectral learning order becomes meaningful and
      is governed by the Gram matrix G = Phi^T Phi.  Orthogonal sampling gives
      uniform rates; non-orthogonal sampling spreads them.  (Sec 6.10-6.14)

Each figure is written to ./figs as both PNG (for slides / quick viewing) and
PDF (vector, for LaTeX \\includegraphics).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ----------------------------------------------------------------------------
# Global style: clean, consistent, paper-friendly.
# ----------------------------------------------------------------------------
rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
    "figure.constrained_layout.use": True,
})

OUTDIR = "figs"
os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.default_rng(0)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ============================================================================
# Figure 1  --  Shallow scalar model: a single geometric contraction factor
# Sec 6.1-6.2:  w^{n+1} - w* = (1 - eta * S_xx)(w^n - w*)
# ============================================================================
def fig_single_mode():
    # synthetic 1-D regression data y ~ w_true x
    m = 200
    x = rng.uniform(0.0, 1.0, size=m)
    y = 2.0 * x + 0.1 * rng.standard_normal(m)
    Sxx = np.sum(x * x)
    Sxy = np.sum(x * y)
    wstar = Sxy / Sxx

    n_steps = 45
    # choose eta so that the contraction factor (1 - eta*Sxx) takes clean values
    factors = [0.8, 0.5, 0.2]
    etas = [(1.0 - c) / Sxx for c in factors]

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(etas)))
    for eta, c, col in zip(etas, factors, colors):
        w = 0.0
        err = []
        for _ in range(n_steps):
            err.append(abs(w - wstar))
            w = w - eta * (Sxx * w - Sxy)
        ax.semilogy(err, "o-", ms=3.5, color=col,
                    label=fr"$1-\eta S_{{xx}} = {c:.1f}$")

    ax.set_xlabel("gradient step $n$")
    ax.set_ylabel(r"$|w^n - w^\ast|$")
    ax.set_title("Shallow scalar model: a single geometric mode")
    ax.legend(title="contraction factor", loc="upper right")
    ax.text(0.02, 0.04,
            "one trainable coordinate $\\Rightarrow$ one straight-line rate\n"
            "(no nontrivial spectral bias)",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    save(fig, "fig1_single_mode")


# ============================================================================
# Figure 2  --  Representational failure on an oscillatory target
# Sec 6.3:  the model can only realize w* x; it has no coordinate for any mode.
# ============================================================================
def fig_representational_failure():
    xs = np.linspace(0.0, 1.0, 400)
    target = lambda x: np.sin(np.pi * x) + 0.5 * np.sin(5 * np.pi * x) \
        + 0.2 * np.sin(15 * np.pi * x)

    # least-squares fit of y = w x on a sample
    xi = np.linspace(0.02, 0.98, 120)
    yi = target(xi)
    wstar = np.sum(xi * yi) / np.sum(xi * xi)

    # sine-mode coefficients
    K = 16
    ks = np.arange(1, K + 1)
    target_coeff = np.zeros(K)
    target_coeff[0] = 1.0       # k = 1
    target_coeff[4] = 0.5       # k = 5
    target_coeff[14] = 0.2      # k = 15
    # coeffs of the straight line w* x in the sin(k pi x) basis: 2(-1)^{k+1}/(k pi)
    line_coeff = wstar * 2.0 * ((-1.0) ** (ks + 1)) / (ks * np.pi)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    axL.plot(xs, target(xs), color="#1f77b4", label=r"target $u^\ast(x)$")
    axL.plot(xs, wstar * xs, color="#d62728", lw=2.4,
             label=fr"best fit $w^\ast x,\ w^\ast={wstar:.2f}$")
    axL.set_xlabel("$x$")
    axL.set_ylabel("$u(x)$")
    axL.set_title("Function space: model cannot bend")
    axL.legend(loc="upper left")

    width = 0.4
    axR.bar(ks - width / 2, np.abs(target_coeff), width,
            color="#1f77b4", label="target modal energy")
    axR.bar(ks + width / 2, np.abs(line_coeff), width,
            color="#d62728", label=r"fit $w^\ast x$ modal energy")
    axR.set_xlabel("sine mode $k$")
    axR.set_ylabel(r"$|$coefficient$|$")
    axR.set_title("Modal space: no coordinate for target modes")
    axR.set_xticks(ks)
    axR.legend(loc="upper right")
    # axR.text(0.5, 0.78,
    #          "target has spikes at $k=1,5,15$;\n"
    #          "the fit is forced to the spectrum of a line.",
    #          transform=axR.transAxes, fontsize=9, ha="center",
    #          bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.suptitle("Oscillatory target: the failure is representational, "
                 "not spectral-dynamical", fontsize=12.5)
    save(fig, "fig2_representational_failure")


# ============================================================================
# Figure 3  --  Depth reshapes optimization geometry (same function class)
# Sec 6.6-6.9:  m = prod_l w_l ;  dL/dw_l = (Sxx m - Sxy) m / w_l
# ============================================================================
def fig_depth_geometry():
    # decoupled, normalized scalar problem: minimize over effective m
    #   dL/dm = Sxx m - Sxy,   with Sxx = 1, Sxy = 2  =>  m* = 2
    Sxx, Sxy = 1.0, 2.0
    mstar = Sxy / Sxx

    depths = [1, 2, 3, 4]
    m0 = 1e-3          # small balanced initialization (exposes depth effect)
    eta = 0.02
    n_steps = 700

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    colors = plt.cm.plasma(np.linspace(0.08, 0.78, len(depths)))
    for L, col in zip(depths, colors):
        w = np.full(L, m0 ** (1.0 / L))   # balanced init so factors stay equal
        m_traj, err_traj = [], []
        for _ in range(n_steps):
            m = np.prod(w)
            m_traj.append(m)
            err_traj.append(abs(m - mstar))
            g = (Sxx * m - Sxy)
            grad = g * m / w               # = g * prod_{j!=l} w_j
            w = w - eta * grad
        lbl = "shallow ($L=1$)" if L == 1 else fr"depth $L={L}$"
        axL.plot(m_traj, color=col, label=lbl)
        axR.semilogy(err_traj, color=col, label=lbl)

    axL.axhline(mstar, ls="--", lw=1.2, color="0.4")
    axL.text(n_steps * 0.97, mstar + 0.06, r"$m^\ast$", ha="right",
             va="bottom", color="0.4")
    axL.set_xlabel("gradient step $n$")
    axL.set_ylabel(r"effective parameter $m^n$")
    axL.legend(loc="lower right")

    axR.set_xlabel("gradient step $n$")
    axR.set_ylabel(r"$|m^n - m^\ast|$")
    axR.legend(loc="lower left")

    save(fig, "fig3_depth_geometry")


# ----------------------------------------------------------------------------
# helpers for the fixed sine-basis model (Figures 4 and 5)
# ----------------------------------------------------------------------------
def design_matrix(x, K):
    ks = np.arange(1, K + 1)
    return np.sin(np.outer(x, ks) * np.pi)   # Phi[i,k] = sin(k pi x_i)


def orthogonal_grid(N):
    # discrete sine-transform grid: x_i = i/(N+1) makes sin(k pi x_i) orthogonal
    i = np.arange(1, N + 1)
    return i / (N + 1)


# ============================================================================
# Figure 4  --  Orthogonal vs non-orthogonal sampling: modal learning rates
# Sec 6.12-6.14:  a^{n+1} - c = (I - eta G)(a^n - c),  G = Phi^T Phi
# ============================================================================
def fig_modal_rates():
    K = 6
    c = np.array([1.0, -0.8, 0.6, -0.5, 0.4, -0.3])   # target modal amplitudes
    n_steps = 120
    modes_to_plot = list(range(K))
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, K))

    # --- orthogonal sampling on the DST grid: G = ((N+1)/2) I ---
    N_orth = 64
    x_orth = orthogonal_grid(N_orth)
    G_orth = design_matrix(x_orth, K).T @ design_matrix(x_orth, K)
    alpha = np.mean(np.diag(G_orth))
    eta_orth = 0.1 / alpha          # gentle: contraction factor ~0.9 for every mode

    # --- moderately non-orthogonal sampling: random points in (0,1) ---
    rloc = np.random.default_rng(1)
    x_non = np.sort(rloc.uniform(0.0, 1.0, 10))
    G_non = design_matrix(x_non, K).T @ design_matrix(x_non, K)
    eta_non = 0.086                 # spans factors ~0.26 (fast) to ~0.95 (slow)

    def run(G, eta):
        a = np.zeros(K)
        hist = []
        I = np.eye(K)
        M = I - eta * G
        for _ in range(n_steps):
            hist.append(np.abs(a - c).copy())
            a = c + M @ (a - c)
        return np.array(hist)            # (n_steps, K)

    H_orth = run(G_orth, eta_orth)
    H_non = run(G_non, eta_non)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    for k in modes_to_plot:
        axL.semilogy(H_orth[:, k], color=cmap[k], label=fr"$k={k+1}$")
        axR.semilogy(H_non[:, k], color=cmap[k], label=fr"$k={k+1}$")

    for ax in (axL, axR):
        ax.set_xlabel("gradient step $n$")
        ax.set_ylim(1e-6, 3)
    axL.set_ylabel(r"modal error $|a_k^n - c_k|$")
    axL.legend(ncol=2, loc="upper right")
    axL.text(0.03, 0.05, "sine modes diagonalize $G$:\nidentical rate for every mode",
             transform=axL.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axR.text(0.03, 0.05, "sine modes are coupled;\nclean rates live in $G$'s eigenbasis",
             transform=axR.transAxes, fontsize=9, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    save(fig, "fig4_modal_rates")


# ============================================================================
# Figure 5  --  The Gram matrix G = Phi^T Phi and its spectrum
# Sec 6.13-6.14:  only when G ∝ I do all modes learn uniformly.
# ============================================================================
def fig_gram_matrix():
    K = 12

    x_orth = orthogonal_grid(96)
    Phi_o = design_matrix(x_orth, K)
    G_o = Phi_o.T @ Phi_o

    x_non = np.sort(np.concatenate([
        rng.uniform(0.0, 0.4, 14),
        rng.uniform(0.5, 1.0, 10),
    ]))
    Phi_n = design_matrix(x_non, K)
    G_n = Phi_n.T @ Phi_n

    # normalize each by its mean diagonal so colour scales are comparable
    Go = G_o / np.mean(np.diag(G_o))
    Gn = G_n / np.mean(np.diag(G_n))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax in axes:
        ax.grid(False)

    im0 = axes[0].imshow(np.abs(Go), cmap="viridis", vmin=0, vmax=1.0,
                         origin="lower", extent=[1, K, 1, K])
    axes[0].set_title("Orthogonal sampling\n$G \\propto I$")
    axes[0].set_xlabel("mode $j$"); axes[0].set_ylabel("mode $k$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.abs(Gn), cmap="viridis", vmin=0, vmax=1.0,
                         origin="lower", extent=[1, K, 1, K])
    axes[1].set_title("Non-orthogonal sampling\noff-diagonal coupling")
    axes[1].set_xlabel("mode $j$"); axes[1].set_ylabel("mode $k$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    ev_o = np.sort(np.linalg.eigvalsh(Go))[::-1]
    ev_n = np.sort(np.linalg.eigvalsh(Gn))[::-1]
    idx = np.arange(1, K + 1)
    axes[2].grid(True, alpha=0.3, which="both")
    axes[2].semilogy(idx, ev_o, "o-", color="#2ca02c", label="orthogonal")
    axes[2].semilogy(idx, np.clip(ev_n, 1e-3, None), "s-", color="#d62728",
                     label="non-orthogonal")
    axes[2].set_title("Eigenvalues of $G$\n(= modal learning rates $\\propto \\mu_j$)")
    axes[2].set_xlabel("index $j$")
    axes[2].set_ylabel(r"normalized eigenvalue $\mu_j$")
    axes[2].legend()

    fig.suptitle(r"Why the rates differ: the Gram matrix $G=\Phi^\top\Phi$ and "
                 r"its eigenvalue spread", fontsize=12.5)
    save(fig, "fig5_gram_matrix")


if __name__ == "__main__":
    fig_single_mode()
    fig_representational_failure()
    fig_depth_geometry()
    fig_modal_rates()
    fig_gram_matrix()
    print("done")
