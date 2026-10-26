#!/usr/bin/env python3
"""Generate the spectral-bias paper figures.

Reads checkpoints from SAVE_ROOT (see paths.py). Writes png/pdf to
finalized_figures/. Intermediate kernels / Lanczos products go to
_cache/, next to this script.

Default figures: spectral gap, final spectra, misalignment, tail (GELU
plus the full-model suite), phenomenon, and loss. Setup, diag-failure,
alignment, and error-loading are off unless requested with --only.

    shifter python generate_finalized_figures.py
    shifter python generate_finalized_figures.py --only gap spectra
    shifter python generate_finalized_figures.py --only loss
    shifter python generate_finalized_figures.py --only tail
    shifter python generate_finalized_figures.py --only all
    shifter python generate_finalized_figures.py --only phenomenon
    shifter python generate_finalized_figures.py --only phenomenon misalign tail --forcing smooth
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import models  # noqa: F401  — needed to unpickle model.pt
from paths import SCRIPT_DIR, SAVE_ROOT as MODEL_DICTS_DIR

# ============================================================
# Style
# ============================================================
FONT_SMALL = 8
FONT_BASE = 9
FONT_LABEL = 9.5

mpl.rcParams.update({
    "font.size": FONT_BASE,
    "axes.labelsize": FONT_LABEL,
    "axes.titlesize": FONT_LABEL,
    "xtick.labelsize": FONT_SMALL,
    "ytick.labelsize": FONT_SMALL,
    "legend.fontsize": FONT_SMALL,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 110,
    "savefig.dpi": 300,
})

LABEL_KW = dict(fontsize=FONT_LABEL, fontweight="bold", va="bottom", ha="left")
FLOOR = 1e-16
EPS = 1e-30
EPS64 = float(np.finfo(np.float64).eps)
SAVE_DPI = 300
FIG_WIDTH = 7.4  # inches; shared so fonts match when figures span the paper width

# ============================================================
# Paths / registry
# ============================================================
dtype = torch.float64
# FIG_OUT_DIR redirects output (e.g. a review folder) without touching
# finalized_figures/.
OUT_DIR = Path(os.environ.get("FIG_OUT_DIR", SCRIPT_DIR / "finalized_figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = SCRIPT_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FORCINGS = ["smooth", "white"]
FORCING_COLORS = {"smooth": "#e8853a", "white": "#6c757d"}

# Five-model hierarchy (paper). Slug linear_dense is the deep linear MLP.
MODELS = [
    ("ideal_diagonal_spectral", "Ideal Diagonal Spectral", "Ideal Diag.", "#2ca02c"),
    ("linear_fno_k64",          "Linear FNO",              "Lin. FNO",    "#17becf"),
    ("nonlinear_fno_k64",       "Nonlinear FNO",           "Nonlin. FNO", "#e377c2"),
    ("gelu_dense",              "GELU Dense",              "GELU Dense",  "#d62728"),
    ("linear_dense",            "Deep Linear Dense",       "Deep Lin.",   "#8c564b"),
]
MODEL_COLOR = {name: color for _, name, _, color in MODELS}
HEATMAP_MODELS = [
    "Ideal Diagonal Spectral",
    "Linear FNO",
    "Deep Linear Dense",
    "GELU Dense",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

N = None
h = None
S = None
MODE_INDEX = None
lambda_A = None
A_inv = None


def panel_label(ax, text, dx=0.0, dy=0.0):
    ax.text(0.0 + dx, 1.05 + dy, text, transform=ax.transAxes, **LABEL_KW)


def loglog_fine_grid(ax):
    """Major + minor log grid (the dense decade grid used on log–log panels)."""
    subs = np.arange(2, 10)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=subs, numticks=100))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=subs, numticks=100))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.3)


def save_both(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=SAVE_DPI,
                    bbox_inches="tight", pad_inches=0.04)
    print(f"[save] {OUT_DIR / (stem + '.png')}  (+ .pdf)")


def place_legend(fig, axes, handles, labels, ncol, offset_in=0.38):
    """Shared legend immediately under the x-axis labels (inch-based offset)."""
    fig.canvas.draw()
    axs = [ax for ax in np.asarray(axes).ravel() if ax.get_visible()]
    y0 = min(ax.get_position().y0 for ax in axs)
    x_mid = 0.5 * (min(ax.get_position().x0 for ax in axs)
                   + max(ax.get_position().x1 for ax in axs))
    y_leg = y0 - offset_in / fig.get_size_inches()[1]
    fig.legend(
        handles, labels, ncol=ncol, loc="upper center",
        bbox_to_anchor=(x_mid, y_leg), bbox_transform=fig.transFigure,
        framealpha=0.92, handlelength=1.4, columnspacing=1.1,
        borderaxespad=0.0, fancybox=False)


# ============================================================
# Geometry + loading
# ============================================================
def configure_geometry():
    global N, h, S, MODE_INDEX, lambda_A, A_inv
    sizes = {}
    for forcing in FORCINGS:
        for slug, *_ in MODELS:
            label = f"{forcing}_{slug}"
            d = MODEL_DICTS_DIR / label
            # Prefer small histories over 400MB ntk_history for size inference.
            for fname in ("modal_abs_err_history.pt", "modal_err_history.pt",
                          "output_history.pt"):
                fp = d / fname
                if not fp.exists():
                    continue
                hist = load_pt(fp) or {}
                if not hist:
                    continue
                v = next(iter(hist.values()))
                sizes[label] = int(np.asarray(v).shape[-1])
                break
    if not sizes:
        raise RuntimeError(f"No artifacts under {MODEL_DICTS_DIR}")
    unique = set(sizes.values())
    if len(unique) != 1:
        raise RuntimeError(f"Inconsistent N: {sizes}")
    N = unique.pop()
    h = 1.0 / (N + 1)
    _j = torch.arange(1, N + 1, dtype=dtype).view(-1, 1)
    _k = torch.arange(1, N + 1, dtype=dtype).view(1, -1)
    S = torch.sqrt(torch.tensor(2.0 / (N + 1), dtype=dtype)) * torch.sin(
        math.pi * _j * _k / (N + 1)
    )
    MODE_INDEX = torch.arange(1, N + 1, dtype=dtype)
    lambda_A = (4.0 / h ** 2) * torch.sin(
        0.5 * math.pi * MODE_INDEX / (N + 1)
    ) ** 2
    A = torch.zeros((N, N), dtype=dtype)
    for i in range(N):
        A[i, i] = 2.0 / h ** 2
        if i > 0:
            A[i, i - 1] = -1.0 / h ** 2
        if i < N - 1:
            A[i, i + 1] = -1.0 / h ** 2
    A_inv = torch.linalg.inv(A)
    print(f"[geom] spatial N={N} (sine basis); forcings={FORCINGS}", flush=True)
    print(f"[geom] note: paper text 'N=64' refers to FNO bandwidth naming "
          f"(slug *_k64); trained spatial grid / sine basis is N={N}.",
          flush=True)


def run_dir(forcing, slug):
    return MODEL_DICTS_DIR / f"{forcing}_{slug}"


def load_pt(path):
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def late_ntk_keys(ntk_history):
    return sorted(k for k in ntk_history if isinstance(k, (int, np.integer)) and k > 0)


def mean_late_ntk(ntk_history):
    keys = late_ntk_keys(ntk_history)
    if not keys:
        raise ValueError("no late NTK keys (steps > 0)")
    acc = None
    for k in keys:
        t = ntk_history[k].to(dtype)
        acc = t.clone() if acc is None else acc + t
    return acc / len(keys), keys


def load_run(forcing, slug, name, load_ntk=False):
    d = run_dir(forcing, slug)
    r = {"name": name, "forcing": forcing, "run_dir": d, "slug": slug}
    for key in ("modal_err_history", "modal_abs_err_history",
                "output_history", "truth_history"):
        r[key] = load_pt(d / f"{key}.pt") or {}
    r["loss_history"] = load_pt(d / "loss_history.pt") or []
    r["ntk_history"] = {}
    r["ntk_path"] = d / "ntk_history.pt"
    if load_ntk and r["ntk_path"].exists():
        r["ntk_history"] = load_pt(r["ntk_path"]) or {}
    r["n_final_ntk"] = load_pt(d / "n_final_ntk.pt")
    r["save_steps"] = load_pt(d / "save_steps.pt")
    return r


def ensure_ntk(result):
    """Load NTK for one run (lazy); return late-mean K and keys.

    Caches the late mean under _cache/ to avoid
    re-reading ~400MB ntk_history.pt on every panel.
    """
    cache_fp = CACHE_DIR / f"late_mean_ntk_{result['forcing']}_{result['slug']}.pt"
    if cache_fp.exists():
        obj = load_pt(cache_fp)
        return obj["K"].to(dtype), list(obj["keys"])
    if not result["ntk_path"].exists():
        raise FileNotFoundError(result["ntk_path"])
    print(f"[ntk] loading {result['ntk_path']} ...", flush=True)
    ntk_history = load_pt(result["ntk_path"]) or {}
    Kmean, keys = mean_late_ntk(ntk_history)
    torch.save({"K": Kmean.cpu(), "keys": keys}, cache_fp)
    print(f"[ntk] cached late mean -> {cache_fp.name}  "
          f"steps {keys[0]}..{keys[-1]} (n={len(keys)})", flush=True)
    del ntk_history
    gc.collect()
    return Kmean, keys


def load_all_runs(load_ntk=False):
    out = {f: [] for f in FORCINGS}
    for forcing in FORCINGS:
        for slug, name, *_ in MODELS:
            r = load_run(forcing, slug, name, load_ntk=load_ntk)
            if not r["modal_abs_err_history"] and not r["output_history"]:
                print(f"[warn] missing histories for {forcing}_{slug}")
                continue
            out[forcing].append(r)
            print(f"[load] {forcing}_{slug}: n_final_ntk={r['n_final_ntk']}; "
                  f"ntk_file={'yes' if r['ntk_path'].exists() else 'no'}")
    return out


def modal_kernel(K):
    if K.ndim == 2:
        K = K.unsqueeze(0)
    K = K.to(dtype)
    Sc = S.to(K.device)
    return torch.einsum("mk,bmn,nj->bkj", Sc, K, Sc)


def diag_mass_fraction(K):
    Kmod = modal_kernel(K)
    diag = torch.diagonal(Kmod, dim1=1, dim2=2)
    num = (diag ** 2).sum(dim=1)
    den = (Kmod ** 2).sum(dim=(1, 2)).clamp_min(EPS)
    return float((num / den).mean())


def rayleigh_diagonal(K):
    """Mean over samples of diag(S^T K S) = Rayleigh quotients R_k."""
    Kmod = modal_kernel(K)
    return torch.diagonal(Kmod, dim1=1, dim2=2).mean(dim=0)


def eig_modal_kernel(K):
    Kmod = modal_kernel(K)
    Kmod = 0.5 * (Kmod + Kmod.transpose(-1, -2))
    lam, psi = torch.linalg.eigh(Kmod)
    return lam.clamp_min(0.0), psi


def centroid_and_pr(psi):
    w = psi ** 2
    w = w / w.sum(dim=1, keepdim=True).clamp_min(EPS)
    cen = torch.einsum("k,bka->ba", MODE_INDEX.to(psi.device), w)
    pr = 1.0 / (w ** 2).sum(dim=1).clamp_min(EPS)
    return cen, pr


def final_abs_spectrum(result):
    abs_hist = result.get("modal_abs_err_history") or {}
    if abs_hist:
        s = max(k for k in abs_hist if isinstance(k, (int, np.integer)))
        return abs_hist[s].to(dtype).numpy(), s
    oh = result.get("output_history") or {}
    th = result.get("truth_history") or {}
    common = sorted(set(oh) & set(th))
    if not common:
        return None, None
    s = common[-1]
    ehat = (oh[s].to(dtype) - th[s].to(dtype)) @ S
    return torch.sqrt((ehat ** 2).mean(0).clamp_min(EPS)).numpy(), s


def final_rel_spectrum(result):
    meh = result.get("modal_err_history") or {}
    if meh:
        s = max(k for k in meh if isinstance(k, (int, np.integer)))
        return meh[s].to(dtype).numpy(), s
    E, s = final_abs_spectrum(result)
    U, _ = truth_spectrum(result)
    if E is None or U is None:
        return None, None
    return E / np.maximum(U, 1e-30), s


def truth_spectrum(result):
    th = result.get("truth_history") or {}
    if not th:
        return None, None
    s = max(k for k in th if isinstance(k, (int, np.integer)))
    u = th[s].to(dtype)
    hat = u @ S
    return torch.sqrt((hat ** 2).mean(0).clamp_min(EPS)).numpy(), s


def output_spectrum(result):
    oh = result.get("output_history") or {}
    if not oh:
        return None, None
    s = max(k for k in oh if isinstance(k, (int, np.integer)))
    u = oh[s].to(dtype)
    hat = u @ S
    return torch.sqrt((hat ** 2).mean(0).clamp_min(EPS)).numpy(), s


# ============================================================
# Fig 3 — setup
# ============================================================
def fig_setup():
    print("\n=== Fig 3: paper_fig_setup ===")
    print("[provenance] analytic: 1/λ_k from discrete Poisson; "
          "forcing variances σ²_smooth=k^{-4}, σ²_white=1")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.28),
                                     constrained_layout=True)
    k = MODE_INDEX.numpy()
    ax_a.semilogy(k, (1.0 / lambda_A).numpy(), color="#333333")
    ax_a.set_xlabel(r"Mode $k$")
    ax_a.set_ylabel(r"True Inverse Gain $1/\lambda_k$")
    ax_a.grid(True, alpha=0.3)
    panel_label(ax_a, "(a)")

    profiles = {"smooth": k ** (-4.0), "white": np.ones(N)}
    for f in FORCINGS:
        ax_b.semilogy(k, profiles[f], color=FORCING_COLORS[f],
                      label=f"{f.capitalize()} Forcing")
    ax_b.set_xlabel(r"Mode $k$")
    ax_b.set_ylabel(r"Forcing Modal Variance $\sigma_k^2$")
    ax_b.grid(True, alpha=0.3)
    ax_b.set_ylim(3e-10, 3.0)
    leg = ax_b.legend(framealpha=1.0, loc="center left",
                      bbox_to_anchor=(1.04, 0.5), borderaxespad=0.0)
    leg.get_frame().set_facecolor("white")
    panel_label(ax_b, "(b)")
    save_both(fig, "paper_fig_setup")
    plt.close(fig)


# ============================================================
# Fig 2 — spectral gap
# ============================================================
def gap_mode(E, U, smooth=3):
    """First k where smoothed Ẽ_k ≳ U_k. Returns 1-based k or None."""
    if smooth > 1:
        pad = smooth // 2
        Ep = np.pad(E, (pad, pad), mode="edge")
        kernel = np.ones(smooth) / smooth
        # 3-point median as specified
        Es = np.array([np.median(Ep[i:i + smooth]) for i in range(len(E))])
    else:
        Es = E
    cross = np.where(Es >= U * (1.0 - 1e-12))[0]
    if len(cross) == 0:
        return None
    return int(cross[0] + 1)


def _fresh_model(slug):
    """Untrained architecture matching train_networks.main()."""
    Sd = S.to(device=device, dtype=dtype)
    if slug == "linear_dense":
        m = models.LinearDense(N, width=512, depth=4)
    elif slug == "gelu_dense":
        m = models.GELUDense(N, width=512, depth=4)
    elif slug == "ideal_diagonal_spectral":
        m = models.IdealDiagonalSpectralOperator(N, S=Sd, K_retained=32)
    elif slug == "linear_fno_k64":
        m = models.LinearFNO1D(N, S=Sd, width=32, layers=3, K_retained=N)
    elif slug == "nonlinear_fno_k64":
        m = models.NonlinearFNO1D(N, S=Sd, width=32, layers=3, K_retained=N)
    else:
        raise ValueError(slug)
    return m.to(device=device, dtype=dtype).eval()


def fig_spectral_gap(results, sample=0):
    """(a,b) u(x) at init; (c,d) u(x) from final model.pt; (e,f) modal spectra.

    Physical panels use one shared forcing per column (same f as the
    black truth). Init is a freshly constructed network; the last row is
    the trained weights. Columns are smooth / white.
    """
    print("\n=== Fig 2: fig_spectral_gap ===")
    x = h * np.arange(1, N + 1, dtype=float)
    modes = np.arange(1, N + 1)
    fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH, 6.70),
                             gridspec_kw=dict(left=0.12, right=0.99,
                                              top=0.97, bottom=0.09,
                                              hspace=0.22, wspace=0.22))
    handles, labels = [], []
    ymaxs = []
    PHYS_SEED = 17

    for col, forcing in enumerate(FORCINGS):
        ax_sp = axes[2, col]
        U_ref = None
        U_step = None
        for slug, name, short, color in MODELS:
            r = next((x for x in results[forcing] if x["name"] == name), None)
            if r is None:
                continue
            U_out, sO = output_spectrum(r)
            U, sU = truth_spectrum(r)
            E, sE = final_abs_spectrum(r)
            if U_out is None or U is None:
                continue
            print(f"[provenance] {name}/{forcing}: û_k from "
                  f"output_history step={sO}; "
                  f"U_k from truth_history step={sU}; "
                  f"gap from Ẽ_k step={sE} "
                  f"(validation B={next(iter(r['truth_history'].values())).shape[0]})")
            # The aligned models track the truth closely, so draw them
            # thicker: green, then blue, with the dotted truth on top.
            z_emph = {"Ideal Diagonal Spectral": 4.3, "Linear FNO": 4.5}
            emph = name in z_emph
            (ln,) = ax_sp.loglog(modes, np.maximum(U_out, FLOOR), color=color,
                                 lw=2.0 if emph else 1.15, alpha=0.95,
                                 zorder=z_emph.get(name, 2), label=name)
            if forcing == FORCINGS[0]:
                handles.append(ln)
                labels.append(name)
            ymaxs.append(U_out.max())
            if U_ref is None:
                U_ref, U_step = U, sU
            if E is not None:
                kgap = gap_mode(E, U_ref)
                if kgap is not None:
                    ax_sp.plot(kgap, U_ref[kgap - 1], marker="|", color=color,
                               ms=10, mew=1.6, zorder=5)
                    print(f"  k_gap({name},{forcing})={kgap}")

        if U_ref is not None:
            (lnU,) = ax_sp.loglog(modes, np.maximum(U_ref, FLOOR), color="k",
                                  lw=2.2, linestyle=":",
                                  label=r"True Solution $U_k$", zorder=6)
            if forcing == FORCINGS[0]:
                handles.append(lnU)
                labels.append(r"True Solution $U_k$")
            ymaxs.append(U_ref.max())
            print(f"[provenance] {forcing}: heavy black U_k from "
                  f"truth_history step={U_step}")

        ax_sp.set_xlabel(r"Sine Mode $k$")
        ax_sp.grid(True, alpha=0.3, which="both")
        panel_label(ax_sp, f"({chr(ord('e') + col)})")

        f_shared = _forcing_batch(1, forcing, PHYS_SEED + col)
        with torch.no_grad():
            u_true = (f_shared @ A_inv.to(device).T)[0].detach().cpu().numpy()
        print(f"[physical] {forcing}: shared f seed={PHYS_SEED + col}  "
              f"sample=0  u_true RMS={float(np.sqrt((u_true ** 2).mean())):.3e}")

        for row, kind, lab in ((0, "init", chr(ord('a') + col)),
                               (1, "final", chr(ord('c') + col))):
            ax = axes[row, col]
            t_lab = 0 if kind == "init" else 12000
            for slug, name, short, color in MODELS:
                r = next((x for x in results[forcing] if x["name"] == name),
                         None)
                if r is None:
                    continue
                if kind == "init":
                    torch.manual_seed(PHYS_SEED)
                    model = _fresh_model(slug)
                else:
                    if not (r["run_dir"] / "model.pt").exists():
                        print(f"[warn] no model.pt for {forcing}_{slug}")
                        continue
                    model = _load_model(r["run_dir"])
                with torch.no_grad():
                    u = model(f_shared)[0].detach().cpu().numpy()
                ax.plot(x, u, color=color, lw=1.05, alpha=0.95)
                del model
                gc.collect()
            ax.plot(x, u_true, color="k", lw=2.0, linestyle=":", zorder=4)
            ax.set_xlabel(r"$x$")
            ax.grid(True, alpha=0.3)
            ax.text(0.04, 0.96, rf"$t={t_lab}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=FONT_SMALL)
            panel_label(ax, f"({lab})")

    axes[0, 0].set_ylabel(r"$u(x)$")
    axes[1, 0].set_ylabel(r"$u(x)$")
    axes[2, 0].set_ylabel(
        r"$\langle (S^\top u_\theta^\mu)_k^2 \rangle_\mu^{1/2}$")
    hi = max(ymaxs) * 2.0 if ymaxs else 1.0
    axes[2, 0].set_ylim(1e-9, hi)
    axes[2, 1].set_ylim(1e-5, hi)
    for ax in axes[2]:
        ax.set_yscale("log")
        ax.set_xscale("log")
    print("[caption] physical rows: one shared f per column; "
          "black = A^{-1}f; colored = model(f) at init / trained model.pt")
    place_legend(fig, axes, handles, labels, ncol=3, offset_in=0.46)
    save_both(fig, "fig_spectral_gap")
    plt.close(fig)


# ============================================================
# Loss curves over training
# ============================================================
def _rolling_mean(y, w=51):
    y = np.asarray(y, dtype=float)
    if y.size < w:
        return np.arange(1, y.size + 1, dtype=float), y
    k = np.ones(w) / w
    ys = np.convolve(y, k, mode="valid")
    x = np.arange((w // 2) + 1, (w // 2) + 1 + len(ys), dtype=float)
    return x, ys


def fig_loss_curves(results, smooth_window=51):
    """(a,b) training MSE vs step for smooth / white."""
    print("\n=== fig_loss_curves ===")
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.35), sharey=True,
                             gridspec_kw=dict(left=0.12, right=0.99,
                                              top=0.90, bottom=0.22,
                                              wspace=0.18))
    handles, labels = [], []
    for ax, forcing, lab in zip(axes, FORCINGS, "ab"):
        for slug, name, short, color in MODELS:
            r = next((x for x in results[forcing] if x["name"] == name), None)
            if r is None:
                continue
            loss = r.get("loss_history") or []
            if not loss:
                print(f"[warn] no loss_history for {forcing}_{slug}")
                continue
            x, y = _rolling_mean(loss, w=smooth_window)
            y = np.maximum(y, FLOOR)
            (ln,) = ax.semilogy(x, y, color=color, lw=1.2, label=name)
            if forcing == FORCINGS[0]:
                handles.append(ln)
                labels.append(name)
            print(f"[provenance] {name}/{forcing}: train MSE from "
                  f"loss_history.pt  n={len(loss)}  "
                  f"rolling mean w={min(smooth_window, len(loss))}  "
                  f"final={float(y[-1]):.3e}")
        ax.set_xlabel("Training Step")
        ax.set_xlim(left=1)
        ax.grid(True, alpha=0.3, which="both")
        panel_label(ax, f"({lab})")
    axes[0].set_ylabel(r"Training MSE")
    place_legend(fig, axes, handles, labels, ncol=3)
    save_both(fig, "fig_loss_curves")
    plt.close(fig)


# ============================================================
# Fig 4 — absolute + relative final spectra
# ============================================================
def slope_triangle(ax, x0, x1, y0, p):
    """Log-log slope triangle of exponent p from (x0, y0) to x1."""
    y1 = y0 * (x1 / x0) ** p
    ax.plot([x0, x1, x1, x0], [y0, y0, y1, y0], color="k", lw=0.9,
            zorder=6)
    ax.text(math.sqrt(x0 * x1), y0 / 1.6, "1", ha="center", va="top",
            fontsize=FONT_SMALL)
    ax.text(x1 * 1.08, math.sqrt(y0 * y1), f"{p:g}", ha="left",
            va="center", fontsize=FONT_SMALL)


def fig_final_spectra(results):
    print("\n=== Fig 4: paper_fig_final_spectra ===")
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 6.2),
                             sharex="col", sharey="row",
                             gridspec_kw=dict(left=0.12, right=0.99,
                                              top=0.94, bottom=0.14,
                                              hspace=0.30, wspace=0.18))
    modes = np.arange(1, N + 1)
    handles, labels = [], []
    rows = (
        (final_abs_spectrum, r"Absolute RMS Modal Error $E_k$",
         "E_k", "modal_abs_err_history"),
        (final_rel_spectrum, r"Relative RMS Modal Error $\varepsilon_k$",
         "ε_k", "modal_err_history"),
    )
    for row, (spectrum_fn, ylabel, tag, hist_name) in enumerate(rows):
        for col, forcing in enumerate(FORCINGS):
            ax = axes[row, col]
            lab = chr(ord("a") + 2 * row + col)
            for slug, name, short, color in MODELS:
                r = next((x for x in results[forcing] if x["name"] == name),
                         None)
                if r is None:
                    continue
                eps, s = spectrum_fn(r)
                if eps is None:
                    continue
                print(f"[provenance] {name}/{forcing}: {tag} from "
                      f"{hist_name} step={s} (val set)")
                (ln,) = ax.loglog(modes, np.maximum(eps, FLOOR), color=color,
                                  lw=1.2, label=name)
                if row == 1 and forcing == FORCINGS[0]:
                    handles.append(ln)
                    labels.append(name)
            ax.grid(True, alpha=0.3, which="both")
            panel_label(ax, f"({lab})")
            if row == 1:
                ax.set_xlabel(r"Sine Mode $k$")
        axes[row, 0].set_ylabel(ylabel)
    # eps_k ~ kappa_k / sigma_k on a flat E_k: k^4 (smooth), k^2 (white)
    slope_triangle(axes[1, 0], 3.0, 8.0, 1e-9, 4)
    slope_triangle(axes[1, 1], 3.0, 10.0, 1e-8, 2)
    place_legend(fig, axes, handles, labels, ncol=3)
    save_both(fig, "paper_fig_final_spectra")
    plt.close(fig)


# ============================================================
# Fig 5 — diag failure + Rayleigh + heatmaps
# ============================================================
def fig_diag_failure(results):
    print("\n=== Fig 5: paper_fig_diag_failure ===")
    fig = plt.figure(figsize=(FIG_WIDTH, 5.87))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.45, 1.20],
                          hspace=0.28, wspace=0.32,
                          left=0.08, right=0.90, top=0.94, bottom=0.05)

    # (a) grouped bars — late-training D
    ax_bar = fig.add_subplot(gs[0, :])
    n_models = len(MODELS)
    x = np.arange(n_models)
    width = 0.32
    print("[provenance] D = Σ_k (K̃_kk)² / ||K̃||_F² on late-mean NTK "
          "(avg over diagnostic batch B=32 samples); step 0 excluded")
    for fi, forcing in enumerate(FORCINGS):
        vals = np.full(n_models, np.nan)
        for mi, (slug, name, short, color) in enumerate(MODELS):
            r = next((x for x in results[forcing] if x["name"] == name), None)
            if r is None or not r["ntk_path"].exists():
                continue
            Kmean, keys = ensure_ntk(r)
            vals[mi] = diag_mass_fraction(Kmean)
            print(f"  D({name},{forcing})={vals[mi]:.4f}  "
                  f"avg over late steps {keys[0]}..{keys[-1]} (n={len(keys)}), "
                  f"K shape={tuple(Kmean.shape)}")
            del Kmean
            gc.collect()
        ax_bar.bar(x + (fi - 0.5) * width, vals, width,
                   color=FORCING_COLORS[forcing], edgecolor="black",
                   linewidth=0.5, label=forcing.capitalize())
    ax_bar.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([short for _, _, short, _ in MODELS],
                           rotation=15, ha="right")
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_ylabel(r"Diag.-Mass Fraction $D$ (Late)")
    ax_bar.legend(ncol=2, loc="upper right", framealpha=0.92)
    panel_label(ax_bar, "(a)")

    # (b) Rayleigh quotients R_k — both forcings as linestyles
    ax_r = fig.add_subplot(gs[1, :])
    print("[provenance] R_k = ⟨v_kᵀ K_μ v_k⟩_μ = mean_μ diag(Sᵀ K_μ S); "
          "late-mean NTK")
    model_handles = []
    for slug, name, short, color in MODELS:
        for forcing, ls in zip(FORCINGS, ("-", "--")):
            r = next((x for x in results[forcing] if x["name"] == name), None)
            if r is None or not r["ntk_path"].exists():
                continue
            Kmean, keys = ensure_ntk(r)
            R = rayleigh_diagonal(Kmean).numpy()
            (ln,) = ax_r.loglog(np.arange(1, N + 1), np.maximum(R, FLOOR),
                                ls=ls, color=color, lw=1.15)
            if forcing == "white":
                model_handles.append(ln)
            del Kmean
            gc.collect()
    style_handles = [
        Line2D([], [], color="0.3", ls="-", lw=1.3, label="White"),
        Line2D([], [], color="0.3", ls="--", lw=1.3, label="Smooth"),
    ]
    ax_r.set_xlabel(r"Fourier Mode Index $k$")
    ax_r.set_ylabel(r"Rayleigh Quotient $R_k$")
    ax_r.grid(True, alpha=0.3, which="both")
    panel_label(ax_r, "(b)")

    dummy_models = [Line2D([], [], color=c, lw=1.4, label=n)
                    for _, n, _, c in MODELS]
    leg_f = ax_r.legend(handles=style_handles, loc="lower left", ncol=1,
                        framealpha=0.92, fontsize=FONT_SMALL, title="Forcing",
                        title_fontsize=FONT_SMALL, borderaxespad=0.4)
    ax_r.add_artist(leg_f)
    ax_r.legend(dummy_models, [m[1] for m in MODELS],
                loc="lower left", ncol=1, framealpha=0.92,
                fontsize=FONT_SMALL, bbox_to_anchor=(0.20, 0.0),
                borderaxespad=0.4)

    # (c–f) heatmaps, white late-mean
    mats, present = [], []
    for name in HEATMAP_MODELS:
        r = next((x for x in results["white"] if x["name"] == name), None)
        if r is None or not r["ntk_path"].exists():
            continue
        Kmean, keys = ensure_ntk(r)
        Kmod = modal_kernel(Kmean).abs().mean(dim=0)
        mats.append(torch.log10(Kmod.clamp_min(1e-16)).numpy())
        present.append(name)
        print(f"[provenance] heatmap {name}/white: late-mean NTK "
              f"steps {keys[0]}..{keys[-1]} (n={len(keys)})")
        del Kmean
        gc.collect()
    if mats:
        vmax = max(m.max() for m in mats)
        vmin = max(min(m.min() for m in mats), vmax - 10)
        ims, heatmap_axes = [], []
        for i, (m, name) in enumerate(zip(mats, present)):
            ax = fig.add_subplot(gs[2, i])
            im = ax.imshow(m, origin="lower", cmap="viridis",
                           vmin=vmin, vmax=vmax,
                           extent=[1, N, 1, N], aspect="equal")
            ims.append(im)
            heatmap_axes.append(ax)
            panel_label(ax, f"({chr(ord('c') + i)})")
            ax.text(0.04, 0.96, name, transform=ax.transAxes,
                    ha="left", va="top", fontsize=FONT_SMALL,
                    color="white", fontweight="bold")
            ax.set_xlabel(r"Mode $j$")
            if i == 0:
                ax.set_ylabel(r"Mode $k$")
            else:
                ax.set_yticklabels([])
        cax = inset_axes(
            heatmap_axes[-1], width="4.5%", height="100%", loc="lower left",
            bbox_to_anchor=(1.06, 0.0, 1.0, 1.0),
            bbox_transform=heatmap_axes[-1].transAxes, borderpad=0)
        cbar = fig.colorbar(ims[-1], cax=cax)
        cbar.set_label(r"$\log_{10}\,\langle|\tilde{K}_{kj}|\rangle_\mu$",
                       fontsize=FONT_SMALL)
        cbar.ax.tick_params(labelsize=FONT_SMALL)

    save_both(fig, "paper_fig_diag_failure")
    plt.close(fig)


# ============================================================
# Fig 6 — alignment
# ============================================================
def fig_alignment(results):
    print("\n=== Fig 6: paper_fig_alignment ===")
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.35), sharey=True,
                             constrained_layout=True)
    all_pts = {f: [] for f in FORCINGS}
    pr_max = 2.0
    for forcing in FORCINGS:
        for r in results[forcing]:
            if not r["ntk_path"].exists():
                continue
            Kmean, keys = ensure_ntk(r)
            lam, psi = eig_modal_kernel(Kmean)
            cen, pr = centroid_and_pr(psi)
            all_pts[forcing].append(
                (lam.flatten().cpu(), cen.flatten().cpu(), pr.flatten().cpu()))
            pr_max = max(pr_max, float(pr.max()))
            print(f"[provenance] {r['name']}/{forcing}: eigenpairs of "
                  f"late-mean modal kernel; steps {keys[0]}..{keys[-1]} "
                  f"(n={len(keys)}), batch B={Kmean.shape[0]}")
            del Kmean, lam, psi, cen, pr
            gc.collect()
    norm = LogNorm(vmin=1.0, vmax=pr_max)
    sc = None
    for ax, forcing, lab in zip(axes, FORCINGS, "ab"):
        for lam, cen, pr in all_pts[forcing]:
            sc = ax.scatter(cen.numpy(), lam.clamp_min(FLOOR).numpy(),
                            c=pr.numpy(), cmap="plasma", norm=norm,
                            s=9, alpha=0.7, edgecolors="none")
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 2, None)
        ax.set_xlabel(r"Frequency Centroid $\bar{k}_\alpha$")
        ax.grid(True, alpha=0.3)
        panel_label(ax, f"({lab})")
    axes[0].set_ylabel(r"Eigenvalue $\lambda_\alpha$")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.9, pad=0.012)
    cbar.set_label(r"Participation Ratio $\mathrm{PR}_\alpha$",
                   fontsize=FONT_SMALL)
    cbar.ax.tick_params(labelsize=FONT_SMALL)
    save_both(fig, "paper_fig_alignment")
    plt.close(fig)


# ============================================================
# Fig 6' — misalignment (merged replacement for figs 5 and 6)
# ============================================================
MISALIGN_NTK_BATCH = 32
MISALIGN_SEED = 977


def _batch_precond_ntk(model, f_batch, P):
    """K = J P J^T with J the Jacobian of the batch-mean output.

    J = (1/B) Σ_μ ∂u(f_μ)/∂θ, then K = (J √P)(J √P)^T ∈ R^{N×N}.
    This is not the mean of per-sample kernels (1/B) Σ_μ J_μ P J_μ^T.
    """
    from torch.func import functional_call, jacrev
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    keys = list(params.keys())
    sqrtP = torch.as_tensor(P, dtype=dtype, device=f_batch.device)
    sqrtP = sqrtP.reshape(-1).clamp_min(0.0).sqrt()
    B = f_batch.shape[0]

    def fwd_single(p, f):
        return functional_call(model, ({**p, **buffers},),
                               (f.unsqueeze(0),)).squeeze(0)

    J_sum = None
    for mu in range(B):
        Jd = jacrev(fwd_single)(params, f_batch[mu])
        Jm = torch.cat([Jd[k].reshape(N, -1) for k in keys], dim=1)
        J_sum = Jm if J_sum is None else J_sum + Jm
        del Jd, Jm
    J = (J_sum / B) * sqrtP.view(1, -1)
    return (J @ J.T).detach().cpu().to(dtype)


def ensure_precond_ntk(result):
    """Late Adam-preconditioned batch kernel K = J P J^T (cached).

    J is the Jacobian of the batch-mean network output on a fixed
    diagnostic forcing batch (B=MISALIGN_NTK_BATCH, seed=MISALIGN_SEED),
    built from final-step model.pt and adam_P.pt. Returns a single
    (N, N) kernel. Setting P=I recovers the batch-mean-Jacobian NTK,
    not the saved per-sample ntk_history.
    """
    cache_fp = CACHE_DIR / (f"late_batch_precond_ntk_"
                            f"{result['forcing']}_{result['slug']}.pt")
    if cache_fp.exists():
        obj = load_pt(cache_fp)
        K = obj["K"].to(dtype)
        if K.ndim != 2:
            raise ValueError(
                f"{cache_fp.name}: expected (N, N) batch kernel, "
                f"got {tuple(K.shape)}")
        return K, obj["meta"]
    model = _load_model(result["run_dir"])
    P = _load_adam_P(result["run_dir"], model)
    f_diag = _forcing_batch(MISALIGN_NTK_BATCH, result["forcing"],
                            MISALIGN_SEED)
    print(f"[precond-ntk] computing K = J P J^T (J = batch-mean Jacobian) "
          f"for {result['forcing']}_{result['slug']} "
          f"(B={MISALIGN_NTK_BATCH}, seed={MISALIGN_SEED}) ...", flush=True)
    K = _batch_precond_ntk(model, f_diag, P)
    meta = {"B": MISALIGN_NTK_BATCH, "seed": MISALIGN_SEED,
            "kind": "batch_mean_jacobian",
            "model": str(result["run_dir"] / "model.pt")}
    torch.save({"K": K.cpu(), "meta": meta}, cache_fp)
    print(f"[precond-ntk] cached -> {cache_fp.name}", flush=True)
    del model, P, f_diag
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return K, meta


def _batch_kernel_eigs(K):
    """Eigensystem of S^T K S, sorted by eigenvalue descending.

    K is the single (N, N) batch kernel. Returns (lam, psi2) with lam of
    shape (N,) and psi2 of shape (N, N): psi2[k, a] is the squared
    overlap of eigenvector a with sine mode k+1.
    """
    if K.ndim != 2:
        raise ValueError(f"expected (N, N) batch kernel, got {tuple(K.shape)}")
    Kmod = modal_kernel(K).squeeze(0)
    Kmod = 0.5 * (Kmod + Kmod.T)
    lam, psi = torch.linalg.eigh(Kmod)
    order = torch.argsort(lam, descending=True)
    lam = lam[order].clamp_min(0.0)
    psi2 = (psi[:, order] ** 2)
    return lam, psi2


def _annotate_ideal_aligned_random(ax, n, k_cut=32):
    """Callouts on the Ideal Diag. overlap matrix (panel a, first model).

    The retained block is a 1–1 diagonal of sine modes; the complementary
    null-space eigenvectors fill a Haar-like block. Both boxes sit in the
    dark rectangles so they do not cover either feature.
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.patheffects import withStroke

    box = dict(
        boxstyle="round,pad=0.12",
        facecolor="white",
        edgecolor="0.2",
        linewidth=0.55,
        alpha=0.95,
    )
    text_kw = dict(
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        va="center",
        color="0.12",
        bbox=box,
        zorder=6,
    )
    stroke = [withStroke(linewidth=3.8, foreground="0.08")]

    def _arrow(start, end, lw=1.2, ms=9, z=7):
        patch = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=ms,
            mutation_aspect=1.0, lw=lw, color="white",
            shrinkA=0, shrinkB=1.0, zorder=z, clip_on=False,
            transform=ax.transData,
        )
        patch.set_path_effects(stroke)
        ax.add_patch(patch)

    # Aligned: box in the lower-right dark strip; arrow from the
    # measured box center to the retained diagonal.
    x_a, y_a = 0.62 * n, 0.50 * k_cut
    ann_a = ax.annotate(
        "Aligned", xy=(x_a, y_a), xytext=(x_a, y_a), **text_kw)
    ax.figure.canvas.draw()
    bb_a = ann_a.get_window_extent(ax.figure.canvas.get_renderer())
    bb_a = bb_a.transformed(ax.transData.inverted())
    ax.annotate(
        "",
        xy=(k_cut - 6, 0.5 * (bb_a.y0 + bb_a.y1)),
        xytext=(0.5 * (bb_a.x0 + bb_a.x1), 0.5 * (bb_a.y0 + bb_a.y1)),
        arrowprops=dict(
            arrowstyle="-|>",
            color="white",
            lw=1.15,
            mutation_scale=8,
            shrinkA=0,
            shrinkB=1.5,
        ),
        zorder=5,
    )

    # Box in the left dark strip; measure its data bbox so the shaft
    # starts just to the right instead of hiding under the label.
    y_r = 0.54 * n
    x_text = 0.40 * k_cut
    ann = ax.annotate(
        "Random", xy=(x_text, y_r), xytext=(x_text, y_r),
        **{**text_kw, "zorder": 8},
    )
    ax.figure.canvas.draw()
    bb = ann.get_window_extent(ax.figure.canvas.get_renderer())
    bb = bb.transformed(ax.transData.inverted())
    x_start = bb.x1 + 1.5
    _arrow((x_start, y_r), (x_start + 0.28 * n, y_r), lw=1.6, ms=11, z=7)


def fig_misalignment(results, forcing="white"):
    """Single figure carrying the alignment story of figs 5 and 6.

    All panels use the same Adam-preconditioned batch kernel
    K = J P J^T (ensure_precond_ntk), with J the Jacobian of the
    batch-mean output.

    (a) Squared-overlap matrices psi^2_{k,alpha} of S^T K S, eigenvectors
        sorted by eigenvalue (desc). Alignment, parity banding, mixing,
        and isotropy are read off directly.
    (b) Eigenvalue vs eigenvector frequency centroid of that same
        modal kernel.
    (c) Participation ratio PR_alpha of each eigenvector of S^T K S
        against its eigenvalue: the eigenvector-side invariant that the
        Frobenius-type diagnostics (D, sorted-spectrum comparison) miss
        when the spectrum has near-degenerate shells. Reference lines:
        PR = 1 (one sine mode per eigenvector) and PR = (N+2)/3 (the
        Haar-random expectation for real vectors).
    (legacy) Q-Q comparison: sorted eigenvalues of S^T K S against sorted
        Rayleigh quotients R_k = (S^T K S)_{kk}. Points on the identity
        line iff the kernel is diagonal in the sine basis (Schur-Horn);
        extent along the line is the rate heterogeneity; departure from
        the line is mode mixing.
    """
    print(f"\n=== Fig 6': paper_fig_misalignment ({forcing}) ===")
    print("[provenance] all panels: K = J P J^T with J = (1/B) Σ_μ "
          "∂u(f_μ)/∂θ from final-step model.pt + adam_P.pt; "
          "modal kernel = S^T K S")

    fig = plt.figure(figsize=(FIG_WIDTH, 5.40))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.15, 1.0], hspace=0.38,
        left=0.075, right=0.88, top=0.90, bottom=0.16)
    gs_top = outer[0].subgridspec(1, len(MODELS), wspace=0.18)
    gs_bot = outer[1].subgridspec(1, 2, wspace=0.32)

    # ---- row (a): overlap matrices (all five models) ----
    ims, hm_axes = [], []
    ov_norm = LogNorm(vmin=1e-4, vmax=1.0)
    for i, (slug, name, short, color) in enumerate(MODELS):
        r = next((x for x in results[forcing] if x["name"] == name), None)
        if r is None or not (r["run_dir"] / "model.pt").exists():
            continue
        K, meta = ensure_precond_ntk(r)
        lam, psi2 = _batch_kernel_eigs(K)
        sort_by_k = slug == "ideal_diagonal_spectral"
        if sort_by_k:
            # Retained eigenvectors ordered by their peak sine mode, so the
            # alignment reads as a clean diagonal; null space keeps rank order.
            kept = lam > 1e-12 * max(float(lam.max()), EPS)
            idx = torch.arange(N)
            ret = idx[kept][torch.argsort(psi2[:, kept].argmax(dim=0))]
            psi2 = psi2[:, torch.cat([ret, idx[~kept]])]
        print(f"[provenance] overlap {name}/{forcing}: S^T K S with "
              f"K = J P J^T, J = batch-mean Jacobian, "
              f"B={meta['B']}, seed={meta['seed']}")
        ax = fig.add_subplot(gs_top[0, i])
        im = ax.imshow(psi2.clamp_min(1e-5).numpy(), origin="lower",
                       cmap="magma", norm=ov_norm,
                       extent=[0.5, N + 0.5, 0.5, N + 0.5], aspect="equal")
        ax.set_anchor("S")
        ims.append(im)
        hm_axes.append(ax)
        panel_label(ax, "(a)" if i == 0 else "", dy=0.08)
        ax.set_title(name, fontsize=FONT_SMALL)
        if name == "Nonlinear FNO":
            ax.set_xlabel(r"Eigenvector Rank $\alpha$")
        if sort_by_k:
            ax.set_xlabel(r"Eigenvector (by peak $k$)", fontsize=FONT_SMALL)
        if i == 0:
            ax.set_ylabel(r"Sine Mode $k$")
            if slug == "ideal_diagonal_spectral":
                _annotate_ideal_aligned_random(ax, N)
        else:
            ax.set_yticklabels([])
        del K
        gc.collect()
    if ims:
        cax = inset_axes(
            hm_axes[-1], width="4.5%", height="100%", loc="lower left",
            bbox_to_anchor=(1.06, 0.0, 1.0, 1.0),
            bbox_transform=hm_axes[-1].transAxes, borderpad=0)
        cbar = fig.colorbar(ims[-1], cax=cax)
        cbar.set_label(r"Squared Overlap $\psi^2_{k\alpha}$",
                       fontsize=FONT_SMALL)
        cbar.ax.tick_params(labelsize=FONT_SMALL)

    # ---- panel (b): eigenvalue vs frequency centroid ----
    ax_b = fig.add_subplot(gs_bot[0, 0])
    handles, labels = [], []
    for slug, name, short, color in MODELS:
        r = next((x for x in results[forcing] if x["name"] == name), None)
        if r is None or not (r["run_dir"] / "model.pt").exists():
            continue
        K, meta = ensure_precond_ntk(r)
        lam, psi = eig_modal_kernel(K)
        cen, _ = centroid_and_pr(psi)
        lam = lam.flatten().numpy() * bar_rate_scale()
        cen = cen.flatten().numpy()
        tol = 1e-12 * max(lam.max(), EPS)
        keep = lam > tol
        print(f"[provenance] centroid {name}/{forcing}: eigenpairs of "
              f"S^T K S, K = J P J^T (J = batch-mean Jacobian), "
              f"B={meta['B']}, kept {int(keep.sum())}/{lam.size} "
              f"(null tol {tol:.1e})")
        (ln,) = ax_b.plot(cen[keep], lam[keep], "o", color=color, ms=3.0,
                          ls="none", markeredgecolor="none", alpha=0.8,
                          zorder=3, label=name)
        handles.append(ln)
        labels.append(name)
        del K, lam, psi, cen
        gc.collect()
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"Frequency Centroid $\bar{k}_\alpha$")
    ax_b.set_ylabel(r"Per-Step Rate $\eta\lambda_\alpha$")
    ax_b.grid(True, alpha=0.3, which="both")
    panel_label(ax_b, "(b)")

    # ---- panel (c): participation ratio vs eigenvalue ----
    # Eigenvector-side companion to panel (b): how many sine modes each
    # eigendirection spreads over, against its rate. This is the
    # invariant that Frobenius-type diagnostics (D, sorted-spectrum
    # Q-Q) cannot see when the spectrum has near-degenerate shells:
    # off-diagonals small against the spectral range still scramble
    # eigenvectors within a shell, leaving D ~ 1 while PR ~ N/3.
    ax_c = fig.add_subplot(gs_bot[0, 1])
    pr_haar = (N + 2.0) / 3.0
    lam_lo, lam_hi = np.inf, -np.inf
    for slug, name, short, color in MODELS:
        r = next((x for x in results[forcing] if x["name"] == name), None)
        if r is None or not (r["run_dir"] / "model.pt").exists():
            continue
        K, meta = ensure_precond_ntk(r)
        lam, psi2 = _batch_kernel_eigs(K)
        pr = 1.0 / (psi2 ** 2).sum(dim=0).clamp_min(EPS)
        lam = lam.numpy() * bar_rate_scale()
        pr = pr.numpy()
        tol = 1e-12 * max(lam.max(), EPS)
        keep = lam > tol
        print(f"[provenance] PR {name}/{forcing}: eigenvectors of S^T K S, "
              f"K = J P J^T (J = batch-mean Jacobian); kept "
              f"{int(keep.sum())}/{N} (null tol {tol:.1e}); "
              f"median PR = {np.median(pr[keep]):.1f}, "
              f"Haar expectation (N+2)/3 = {pr_haar:.1f}")
        ax_c.plot(lam[keep], pr[keep], "o", color=color, ms=3.0,
                  markeredgecolor="none", alpha=0.8, zorder=3)
        lam_lo = min(lam_lo, float(lam[keep].min()))
        lam_hi = max(lam_hi, float(lam[keep].max()))
        del K
        gc.collect()
    ax_c.axhline(1.0, color="0.55", lw=1.6, ls="--", zorder=1,
                 label=r"Sine mode ($\mathrm{PR} = 1$)")
    ax_c.axhline(pr_haar, color="0.55", lw=1.6, ls=":", zorder=1,
                 label=r"Random vector ($\mathrm{PR} = (N+2)/3$)")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlim(lam_lo * 0.4, lam_hi * 2.5)
    ax_c.set_ylim(0.7, 5e2)
    ax_c.set_xlabel(r"Per-Step Rate $\eta\lambda_\alpha$")
    ax_c.set_ylabel(r"Participation Ratio $\mathrm{PR}_\alpha$")
    ax_c.grid(True, alpha=0.3, which="both")
    panel_label(ax_c, "(c)")
    ax_c.legend(loc="upper left", framealpha=0.92, fontsize=FONT_SMALL,
                handlelength=1.4, borderaxespad=0.4, fancybox=False)
    place_legend(fig, [ax_b, ax_c], handles, labels, ncol=3)

    save_both(fig, f"paper_fig_misalignment_{forcing}")
    plt.close(fig)


# ============================================================
# Fig 1 — error-loaded G(λ)  (GPU-heavy)
# ============================================================
B_STACK = 2048
B_BLOCK = 512
M_LANCZOS = 400
CHUNK = 1024
N_BATCH = 4


def stack_rate_scale():
    """Eigenvalue of J P J^T (P includes the step size) -> per-step rate.

    Training minimises the mean of the squared error over the B*N entries
    of a batch, so one Adam step contracts the stacked error by
    (2/(B N)) J P J^T. B is the diagnostic batch B_STACK.
    """
    return 2.0 / (B_STACK * N)


def bar_rate_scale():
    """Eigenvalue of Jbar P Jbar^T -> per-step rate of an error shared by
    every sample of the batch: (2/N) Jbar P Jbar^T."""
    return 2.0 / N


def _forcing_batch(batch_size, forcing, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    coeffs = torch.randn(batch_size, N, dtype=dtype, device=device,
                         generator=g)
    k = torch.arange(1, N + 1, dtype=dtype, device=device)
    if forcing == "smooth":
        coeffs = coeffs / (k ** 2)          # σ_k = k^{-2} => σ² = k^{-4}
    elif forcing == "white":
        pass
    else:
        raise ValueError(forcing)
    return coeffs @ S.to(device).T


def _load_model(run_path):
    fp = Path(run_path) / "model.pt"
    try:
        m = torch.load(fp, map_location=device, weights_only=False)
    except TypeError:
        m = torch.load(fp, map_location=device)
    return m.to(device=device, dtype=dtype).eval()


def _load_adam_P(run_dir, model):
    """Load diagonal Adam P (named_parameters order). Falls back to I."""
    n = sum(p.numel() for _, p in model.named_parameters())
    fp = Path(run_dir) / "adam_P.pt"
    if not fp.exists():
        print(f"[warn] {fp.name} missing; Lanczos uses P=I (retrain to save Adam)")
        return torch.ones(n, dtype=dtype, device=device)
    P = load_pt(fp)
    P = torch.as_tensor(P, dtype=dtype, device=device).reshape(-1)
    if P.numel() != n:
        raise ValueError(
            f"adam_P length {P.numel()} != n_params {n} for {run_dir}")
    print(f"[adam] P from {fp}  min={float(P.min()):.3e}  "
          f"median={float(P.median()):.3e}  max={float(P.max()):.3e}")
    return P


def _make_kernel_op(model, f_batch, chunk=CHUNK, P=None):
    """Matrix-free stacked kernel: u ↦ J P Jᵀ u  (P=I recovers J Jᵀ)."""
    from torch.func import functional_call, jvp, vjp
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    keys = list(params.keys())
    n = sum(params[k].numel() for k in keys)
    if P is None:
        P_vec = torch.ones(n, dtype=dtype, device=f_batch.device)
    else:
        P_vec = torch.as_tensor(P, dtype=dtype, device=f_batch.device).reshape(-1)
        if P_vec.numel() != n:
            raise ValueError(f"P length {P_vec.numel()} != {n}")

    def fwd(p, f):
        return functional_call(model, ({**p, **buffers},), (f,))

    def Jv(v_flat):
        out, i = {}, 0
        for k in keys:
            n_ = params[k].numel()
            out[k] = v_flat[i:i + n_].view_as(params[k])
            i += n_
        res = []
        for i0 in range(0, f_batch.shape[0], chunk):
            fc = f_batch[i0:i0 + chunk]
            _, ju = jvp(lambda p: fwd(p, fc), (params,), (out,))
            res.append(ju)
        return torch.cat(res, dim=0)

    def JTu(u_bn):
        acc = None
        for i0 in range(0, f_batch.shape[0], chunk):
            fc = f_batch[i0:i0 + chunk]
            uc = u_bn[i0:i0 + chunk]
            _, vjp_fn = vjp(lambda p: fwd(p, fc), params)
            gf = torch.cat([vjp_fn(uc)[0][k].reshape(-1) for k in keys])
            acc = gf if acc is None else acc + gf
        return acc

    return lambda u: Jv(P_vec * JTu(u))


def _lanczos(K_op, e, m=M_LANCZOS, return_basis=False):
    v = e.reshape(-1)
    beta0 = float(torch.linalg.norm(v))
    Q = [v / beta0]
    alphas, betas = [], []
    for _ in range(m):
        w = K_op(Q[-1].view(e.shape)).reshape(-1)
        alpha = float(Q[-1] @ w)
        w = w - alpha * Q[-1] - (betas[-1] * Q[-2] if betas else 0.0)
        for _ in range(2):
            for qi in Q:
                w = w - (qi @ w) * qi
        beta = float(torch.linalg.norm(w))
        alphas.append(alpha)
        if beta < 1e-13 * beta0:
            break
        betas.append(beta)
        Q.append(w / beta)
    m_eff = len(alphas)
    T = np.diag(np.array(alphas))
    if betas:
        off = np.array(betas[:m_eff - 1])
        T += np.diag(off, 1) + np.diag(off, -1)
    theta, V = np.linalg.eigh(T)
    theta = np.maximum(theta, 0.0)
    weights = (beta0 ** 2) * V[0, :] ** 2
    if return_basis:
        return theta, weights, torch.stack(Q[:m_eff]), V, beta0
    return theta, weights, beta0


def _block_diag_measure(model, f_batch, e, P=None):
    from torch.func import functional_call, jacrev
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    keys = list(params.keys())
    n = sum(params[k].numel() for k in keys)
    if P is None:
        sqrtP = torch.ones(n, dtype=dtype, device=f_batch.device)
    else:
        sqrtP = torch.as_tensor(P, dtype=dtype, device=f_batch.device).reshape(-1)
        if sqrtP.numel() != n:
            raise ValueError(f"P length {sqrtP.numel()} != {n}")
        sqrtP = sqrtP.clamp_min(0.0).sqrt()

    def fwd_single(p, f):
        return functional_call(model, ({**p, **buffers},),
                               (f.unsqueeze(0),)).squeeze(0)

    nodes, weights = [], []
    for mu in range(f_batch.shape[0]):
        Jd = jacrev(fwd_single)(params, f_batch[mu])
        Jm = torch.cat([Jd[k].reshape(N, -1) for k in keys], dim=1)
        Jm = Jm * sqrtP.view(1, -1)
        evals, evecs = torch.linalg.eigh(Jm @ Jm.T)
        a = evecs.T @ e[mu]
        nodes.append(evals.clamp(min=0).cpu().numpy())
        weights.append((a ** 2).cpu().numpy())
        del Jd, Jm
    return np.concatenate(nodes), np.concatenate(weights)


def _G_of(nodes, weights, Lam_grid, BN):
    """Cumulative RMS of error mass on Ritz values with λ ≤ L.

    Every Lanczos/block-diag weight is kept. G(∞) = √(Σw/BN).
    """
    nodes = np.asarray(nodes, float)
    weights = np.asarray(weights, float)
    return np.array([math.sqrt(weights[nodes <= L].sum() / BN)
                     for L in Lam_grid])


def _compute_error_loading_one(model, forcing, seed_base, P=None):
    """Return dict with Lam_grid, G_full_runs, G_bd_runs, rms_runs."""
    G_full_runs, G_bd_runs, rms_runs = [], [], []
    lam_grid = None
    for bi in range(N_BATCH):
        f_full = _forcing_batch(B_STACK, forcing, seed_base + bi)
        with torch.no_grad():
            e_full = model(f_full) - f_full @ A_inv.to(device).T
        rms = float(torch.linalg.norm(e_full) / math.sqrt(B_STACK * N))
        rms_runs.append(rms)
        K_op = _make_kernel_op(model, f_full, P=P)
        nodes_f, w_f, _ = _lanczos(K_op, e_full)
        nodes_b, w_b = _block_diag_measure(
            model, f_full[:B_BLOCK], e_full[:B_BLOCK], P=P)
        if lam_grid is None:
            lam_max = max(float(nodes_f.max()), float(nodes_b.max()), 1e-12)
            lam_grid = np.logspace(np.log10(lam_max) - 8, np.log10(lam_max), 250)
        G_full_runs.append(_G_of(nodes_f, w_f, lam_grid, B_STACK * N))
        G_bd_runs.append(_G_of(nodes_b, w_b, lam_grid, B_BLOCK * N))
        del K_op, f_full, e_full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    batch {bi + 1}/{N_BATCH}: RMS={rms:.3e}  "
              f"G_full(∞)~{G_full_runs[-1][-1]:.3e}")
    return {
        "Lam_grid": lam_grid,
        "G_full": np.mean(G_full_runs, 0),
        "G_full_lo": np.min(G_full_runs, 0),
        "G_full_hi": np.max(G_full_runs, 0),
        "G_bd": np.mean(G_bd_runs, 0),
        "G_bd_lo": np.min(G_bd_runs, 0),
        "G_bd_hi": np.max(G_bd_runs, 0),
        "rms": float(np.mean(rms_runs)),
    }


def fig_error_loading(results):
    print("\n=== Fig 1: fig_error_loading ===")
    print(f"[provenance] late-trained model.pt Jacobians (NOT init); "
          f"stacked K=J P Jᵀ (Adam diagonal P) via matrix-free Lanczos "
          f"m={M_LANCZOS}, B={B_STACK}, {N_BATCH} batches; "
          f"block-diag B={B_BLOCK}; device={device}")
    if device == "cpu":
        print("[warn] no CUDA — error-loading will be extremely slow on CPU")

    cache_fp = CACHE_DIR / "error_loading_cache_adamP.pt"
    cache = load_pt(cache_fp) or {}

    # small-multiples: 5 models × 2 forcings
    fig, axes = plt.subplots(
        len(MODELS), 2, figsize=(FIG_WIDTH, 10.06), sharex=True,
        gridspec_kw=dict(hspace=0.30, wspace=0.22,
                         left=0.12, right=0.98, top=0.96, bottom=0.10))
    style_handles = [
        Line2D([], [], color="0.2", ls="-", lw=1.4,
               label="Full Stacked $K$"),
        Line2D([], [], color="0.2", ls="--", lw=1.4,
               label=r"Block-Diag (No Cross)"),
        Line2D([], [], color="0.2", ls=":", lw=1.2,
               label=r"RMS $\|e\|/\sqrt{BN}$"),
    ]

    for mi, (slug, name, short, color) in enumerate(MODELS):
        for fi, forcing in enumerate(FORCINGS):
            ax = axes[mi, fi]
            key = f"{forcing}_{slug}_adamP"
            r = next((x for x in results[forcing] if x["name"] == name), None)
            if r is None:
                ax.set_visible(False)
                continue
            if key not in cache:
                print(f"[compute] {key} ...")
                model = _load_model(r["run_dir"])
                P = _load_adam_P(r["run_dir"], model)
                seed = 1000 + 17 * mi + 3 * fi
                print(f"[provenance] {key}: model={r['run_dir']/ 'model.pt'}; "
                      f"forcing ensemble seeds={seed}+bi; "
                      f"error e=u_θ(f)-A^{{-1}}f; K=J P Jᵀ")
                cache[key] = _compute_error_loading_one(
                    model, forcing, seed, P=P)
                torch.save(cache, cache_fp)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"[cache] {key}")
            d = cache[key]
            Lam = d["Lam_grid"]
            ax.fill_between(Lam, np.maximum(d["G_full_lo"], FLOOR),
                            np.maximum(d["G_full_hi"], FLOOR),
                            color=color, alpha=0.18, lw=0)
            ax.loglog(Lam, np.maximum(d["G_full"], FLOOR), "-",
                      color=color, lw=1.35)
            ax.loglog(Lam, np.maximum(d["G_bd"], FLOOR), "--",
                      color=color, lw=1.2, alpha=0.85)
            ax.axhline(d["rms"], color=color, ls=":", lw=1.0)
            ys = np.concatenate([
                np.maximum(d["G_full"], FLOOR),
                np.maximum(d["G_bd"], FLOOR),
                [max(d["rms"], FLOOR)],
            ])
            ylo, yhi = float(ys.min()), float(ys.max())
            if yhi / max(ylo, FLOOR) < 20:
                mid = math.sqrt(max(ylo, FLOOR) * yhi)
                ax.set_ylim(mid / 40.0, mid * 40.0)
            else:
                ax.set_ylim(max(ylo * 0.4, FLOOR), yhi * 2.5)
            ax.grid(True, alpha=0.25, which="both")
            if fi == 0:
                ax.text(0.03, 0.96, short, transform=ax.transAxes,
                        va="top", ha="left", fontsize=FONT_SMALL,
                        color=color, fontweight="bold")
            if mi == len(MODELS) // 2 and fi == 0:
                ax.set_ylabel(r"Error Mass Below $\lambda$ (RMS)")
            if mi == len(MODELS) - 1:
                ax.set_xlabel(r"Decay Rate $\lambda$")
            print(f"  {key}: RMS={d['rms']:.3e}  "
                  f"G_full[0]={d['G_full'][0]:.3e}")

    place_legend(fig, axes, style_handles,
                 [h.get_label() for h in style_handles], ncol=3)
    save_both(fig, "fig_error_loading")
    plt.close(fig)


# ============================================================
# Tail causes flatness  (GPU-heavy Lanczos)
# ============================================================
K_GAP = 16
LAM_SLOW = 0.1
G_STAR_FRAC = 0.95  # λ*: this fraction of plotted RMS error mass G(∞) ≡ E
SHARE_FLOOR = 1e-12
TAIL_LANCZOS = 2000
TAIL_CACHE_VER = 4
# GELU Dense has its own figure (`fig_tail_causes_flatness`); the `_all`
# grids show the other four models.
TAIL_MODELS = [
    next(m for m in MODELS if m[0] == slug)
    for slug in (
        "nonlinear_fno_k64",
        "linear_fno_k64",
        "linear_dense",
        "ideal_diagonal_spectral",
    )
]
# Panel (a) legend loc on the single-model figure. Panel (b) stays lower left.
TAIL_LEGEND_A = {
    "gelu_dense": "lower right",
}
# Grid-only. Partial specs; omitted edges stay on matplotlib autoscale.
# x_from_a copies panel-(a) xlim onto panel (b) after (a) is limited.
TAIL_GRID_LIM = {
    "ideal_diagonal_spectral": {
        "a": dict(x1=1e-8, y0=2e-5),
    },
    "linear_dense": {
        "a": dict(y0=6.5e-16, y1=7.5e-16, x0=1e2, x1="lam_max"),
        "b": dict(x1="lam_max"),
    },
}


def _tail_cache_fp(forcing, slug):
    return CACHE_DIR / f"tail_lanczos_{forcing}_{slug}_v{TAIL_CACHE_VER}.pt"


def _resolve_tail_spec(name):
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    for slug, display, *_ in MODELS:
        if slug == key or display.lower() == name.strip().lower() \
                or display.lower().replace(" ", "_") == key:
            return slug, display
    choices = ", ".join(display for _, display, *_ in MODELS)
    sys.exit(f"unknown --tail-model '{name}'. Choose one of: {choices}")


def _lam_star_from_G(G, grid, frac=G_STAR_FRAC):
    """Smallest λ where G(λ) reaches `frac` of the plotted RMS G(∞) ≡ E."""
    G = np.asarray(G, float)
    grid = np.asarray(grid, float)
    G_inf = float(G[-1])
    target = frac * G_inf
    idx = int(np.searchsorted(G, target))
    if idx >= len(grid):
        return float(grid[-1]), 1.0
    if idx == 0:
        return float(grid[0]), float(G[0] / max(G_inf, 1e-300))
    y0, y1 = float(G[idx - 1]), float(G[idx])
    x0, x1 = float(grid[idx - 1]), float(grid[idx])
    if not (y1 > y0 > 0) or not (x1 > x0 > 0):
        return x1, float(y1 / max(G_inf, 1e-300))
    t = float(np.clip((target - y0) / (y1 - y0), 0.0, 1.0))
    lam = x0 * (x1 / x0) ** t
    return float(lam), frac


def _rms_frac_at(G, grid, lam, G_inf):
    g = float(np.interp(lam, grid, G))
    return g / max(G_inf, 1e-300)


def _reconstruct_error(mask, V, Q, beta0):
    V_t = torch.as_tensor(V, dtype=dtype, device=device)
    y0 = V_t[0, :] * beta0
    m = torch.as_tensor(np.asarray(mask, float), dtype=dtype, device=device)
    coef = V_t @ (m * y0)
    return (coef @ Q).view(B_STACK, N)


def _modal_profile(e_rec):
    return torch.sqrt(((e_rec @ S.to(device)) ** 2).mean(dim=0)).cpu().numpy()


def _apply_tail_view(raw):
    """Recompute G(λ) on the observed Ritz range. Axes autoscale from data."""
    sc = stack_rate_scale()
    full_nw = [(np.asarray(n, float) * sc, w_)
               for n, w_ in zip(raw["full_nodes"], raw["full_w"])]
    bd_nw = [(np.asarray(n, float) * sc, w_)
             for n, w_ in zip(raw["bd_nodes"], raw["bd_w"])]
    nodes = np.asarray(raw["nodes"], float) * sc
    w = np.asarray(raw["w"], float)
    share = w / max(float(w.sum()), 1e-300)
    E_now = float(raw["E_now"])
    label = raw["label"]
    slug = raw["slug"]
    forcing = raw["forcing"]

    pos_f = np.concatenate([np.asarray(n, float)[np.asarray(n, float) > 0]
                            for n, _ in full_nw]) if full_nw else nodes[nodes > 0]
    pos_b = np.concatenate([np.asarray(n, float)[np.asarray(n, float) > 0]
                            for n, _ in bd_nw]) if bd_nw else pos_f
    pos_all = np.concatenate([pos_f, pos_b]) if pos_b.size else pos_f
    probe_lo = float(pos_all.min()) if pos_all.size else 1e-12
    probe_hi = float(pos_all.max()) if pos_all.size else 1.0
    if probe_lo >= probe_hi:
        probe_lo = probe_hi / 10
    probe = np.logspace(np.log10(probe_lo), np.log10(probe_hi * 1.05), 250)
    G_star = np.mean(
        [_G_of(n, w_, probe, B_STACK * N) for n, w_ in full_nw], axis=0)
    G_inf = float(G_star[-1])
    lam_star, frac_star = _lam_star_from_G(G_star, probe)
    frac_01 = _rms_frac_at(G_star, probe, LAM_SLOW * sc, G_inf)

    lam_grid = probe
    G_full = np.mean(
        [_G_of(n, w_, lam_grid, B_STACK * N) for n, w_ in full_nw], axis=0)
    G_bd = np.mean(
        [_G_of(n, w_, lam_grid, B_BLOCK * N) for n, w_ in bd_nw], axis=0)
    frac_bd_star = _rms_frac_at(G_bd, lam_grid, lam_star, G_inf)
    vis = (nodes > 0) & (share >= SHARE_FLOOR)

    # G_bd is a smaller-batch RMS and can sit a fraction above G(∞). Cap so
    # both curves meet the dotted E line rather than overshooting it.
    G_full = np.minimum(np.asarray(G_full, float), G_inf)
    G_bd = np.minimum(np.asarray(G_bd, float), G_inf)

    print(f"[lim] data range λ=({probe_lo:.3e}, {probe_hi:.3e})  "
          f"E={E_now:.3e}  G(lo)={float(G_full[0]):.3e}  "
          f"G(hi)={float(G_full[-1]):.3e}  G_bd(hi)={float(G_bd[-1]):.3e}")
    print(f"[λ*] lam_star={lam_star:.6e}  G(λ*)/E={100.0 * frac_star:.2f}%  "
          f"G_inf={G_inf:.3e}")

    return dict(
        label=label, slug=slug, forcing=forcing,
        lam_grid=lam_grid, G_full=G_full, G_bd=G_bd, E_now=G_inf,
        lam_star=lam_star, lam_max=probe_hi, nodes=nodes, share=share,
        vis=vis, frac_star=frac_star, frac_01=frac_01,
        frac_bd_star=frac_bd_star, G_inf=G_inf,
    )


def compute_tail_causes(results, *, model_name="GELU Dense", forcing="white",
                        use_cache=True):
    """Lanczos + G(λ) for one model. Returns a plot-ready dict, or None."""
    slug, label = _resolve_tail_spec(model_name)
    r = next((x for x in results.get(forcing, []) if x["slug"] == slug), None)
    if r is None:
        r = load_run(forcing, slug, label)
    if not (r["run_dir"] / "model.pt").exists():
        print(f"[warn] no model.pt for {forcing}_{slug}; skip tail figure")
        return None
    cache_fp = _tail_cache_fp(forcing, slug)
    if use_cache and cache_fp.exists():
        print(f"[cache] load {cache_fp}")
        raw = load_pt(cache_fp)
        d = _apply_tail_view(raw)
        d["r"] = r
        return d

    print(f"[provenance] late model.pt Jacobians, stacked K=J P Jᵀ "
          f"(Adam diagonal P) Lanczos m={TAIL_LANCZOS}, B={B_STACK}, "
          f"{N_BATCH} batches; block-diag B={B_BLOCK}; device={device}")
    print(f"  model={label}  forcing={forcing}  run={r['run_dir']}")
    if device == "cpu":
        print("[warn] no CUDA — tail-causes will be extremely slow on CPU")

    model = _load_model(r["run_dir"])
    P = _load_adam_P(r["run_dir"], model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  dim(K)=B·N={B_STACK * N};  n_params={n_params};  "
          f"rank(K)≤min(BN,n_params)={min(B_STACK * N, n_params)};  "
          f"Lanczos m={TAIL_LANCZOS}")
    E_obs, sE = final_abs_spectrum(r)
    if E_obs is not None:
        print(f"  observed E_k from modal_abs_err_history step={sE}")

    full_nw, bd_nw, E_now_runs = [], [], []
    keep = None
    for bi in range(N_BATCH):
        f_full = _forcing_batch(B_STACK, forcing, 1000 + bi)
        with torch.no_grad():
            e_full = model(f_full) - f_full @ A_inv.to(device).T
        E_now_runs.append(float(
            torch.linalg.norm(e_full) / math.sqrt(B_STACK * N)))
        K_op = _make_kernel_op(model, f_full, P=P)
        nodes_f, w_f, Q, V, beta0 = _lanczos(
            K_op, e_full, m=TAIL_LANCZOS, return_basis=True)
        nodes_b, w_b = _block_diag_measure(
            model, f_full[:B_BLOCK], e_full[:B_BLOCK], P=P)
        full_nw.append((nodes_f, w_f))
        bd_nw.append((nodes_b, w_b))
        if keep is None:
            keep = dict(f_full=f_full, e_full=e_full, nodes=nodes_f,
                        w=w_f, Q=Q, V=V, beta0=beta0, P=P)
        del K_op
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    batch {bi + 1}/{N_BATCH}")

    nodes = np.asarray(keep["nodes"], float)
    w = np.asarray(keep["w"], float)
    Q, V, beta0 = keep["Q"], keep["V"], keep["beta0"]
    e_full = keep["e_full"]

    e_full_rec = _reconstruct_error(np.ones(len(nodes)), V, Q, beta0)
    max_dev = float((e_full_rec - e_full).abs().max())
    print(f"[recon] max |e_full_rec - e_full| = {max_dev:.3e}")
    if not torch.allclose(e_full_rec, e_full, rtol=1e-2, atol=1e-5):
        print(f"[warn] full Ritz reconstruction not allclose "
              f"(max dev={max_dev:.3e})")

    raw = dict(
        version=TAIL_CACHE_VER, label=label, slug=slug, forcing=forcing,
        E_now=float(np.mean(E_now_runs)),
        full_nodes=[np.asarray(n, float) for n, _ in full_nw],
        full_w=[np.asarray(w_, float) for _, w_ in full_nw],
        bd_nodes=[np.asarray(n, float) for n, _ in bd_nw],
        bd_w=[np.asarray(w_, float) for _, w_ in bd_nw],
        nodes=nodes, w=w,
    )
    torch.save(raw, cache_fp)
    print(f"[cache] wrote {cache_fp}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    d = _apply_tail_view(raw)
    d["r"] = r
    d["keep"] = keep
    print(
        "[caption] panel (a) stats — paste into the figure caption:\n"
        f"  RMS below λ*: {100.0 * d['frac_star']:.0f}%;  "
        f"below 0.1: {100.0 * d['frac_01']:.0f}%;  "
        f"cross-terms removed: {d['frac_bd_star']:.2g}"
    )
    return d


def _resolve_lim_val(val, d):
    if val == "lam_max":
        return float(d["lam_max"])
    return val


def _apply_grid_lim(ax, spec, d, ax_a=None):
    """Apply any provided x0/x1/y0/y1; optionally copy xlim from panel (a)."""
    if not spec:
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if spec.get("x_from_a") and ax_a is not None:
        x0, x1 = ax_a.get_xlim()
    sc = stack_rate_scale()  # hand-set limits are in raw eigenvalue units
    if "x0" in spec:
        x0 = _resolve_lim_val(spec["x0"], d)
        x0 = x0 if spec["x0"] == "lam_max" else x0 * sc
    if "x1" in spec:
        x1 = _resolve_lim_val(spec["x1"], d)
        x1 = x1 if spec["x1"] == "lam_max" else x1 * sc
    if "y0" in spec:
        y0 = _resolve_lim_val(spec["y0"], d)
    if "y1" in spec:
        y1 = _resolve_lim_val(spec["y1"], d)
    if x0 > 0 and x1 > x0:
        ax.set_xlim(x0, x1)
    if y0 > 0 and y1 > y0:
        ax.set_ylim(y0, y1)


def _draw_tail_row(_fig, ax_a, ax_b, d, lab_a, lab_b, *,
                   in_axes_legend=True, model_in_legend=True,
                   model_text=False, grid_lim=None):
    label = d["label"]
    k_lab = (rf"Full $K_{{\mu\nu}}$ {label}" if model_in_legend
             else rf"Full $K_{{\mu\nu}}$")
    ax_a.loglog(d["lam_grid"], d["G_full"], color="tab:red", label=k_lab)
    ax_a.loglog(d["lam_grid"], d["G_bd"], color="0.45", ls="--",
                label="Cross Terms Removed")
    ax_a.axhline(d["E_now"], color="k", lw=0.8, ls=":",
                 label="Observed Error $E$")
    ax_a.axvline(d["lam_star"], color="tab:blue", lw=1.2, ls="-",
                 label=r"$\eta\lambda^*$")
    ax_a.set_xlabel(r"Per-Step Rate $\eta\lambda$")
    ax_a.set_ylabel(r"Error Mass Below $\eta\lambda$ (RMS)")
    ax_a.grid(True, alpha=0.3)
    if in_axes_legend:
        loc_a = TAIL_LEGEND_A.get(d["slug"], "lower left")
        ax_a.legend(loc=loc_a, framealpha=0.92, fontsize=FONT_SMALL,
                    handlelength=1.4, fancybox=False, borderaxespad=0.4)
    if model_text:
        ax_a.text(0.97, 0.05, label, transform=ax_a.transAxes,
                  ha="right", va="bottom", fontsize=FONT_SMALL,
                  bbox=dict(facecolor="white", edgecolor="none",
                            alpha=0.82, pad=0.15))
    panel_label(ax_a, f"({lab_a})")

    nodes, share = d["nodes"], d["share"]
    vis = d["vis"]
    # Ideal Diag puts ~all error in one near-zero Ritz value; the rest have
    # shares far below SHARE_FLOOR and live at large λ. Show every λ>0 so
    # those modes are visible.
    if d["slug"] == "ideal_diagonal_spectral":
        vis = nodes > 0
    if vis.any():
        ax_b.scatter(nodes[vis], share[vis],
                     color=MODEL_COLOR.get(label, "0.2"),
                     s=14, alpha=0.85, zorder=2, linewidths=0)
    ax_b.axvline(d["lam_star"], color="tab:blue", lw=1.2, ls="-",
                 label=r"$\eta\lambda^*$")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"Per-Step Rate $\eta\lambda_\alpha$")
    ax_b.set_ylabel(r"Error Share $a_\alpha^2/\|e\|^2$")
    ax_b.grid(True, alpha=0.3)
    if in_axes_legend:
        ax_b.legend(loc="lower left", framealpha=0.92, fontsize=FONT_SMALL,
                    handlelength=1.4, fancybox=False, borderaxespad=0.4)
    panel_label(ax_b, f"({lab_b})")
    if grid_lim:
        _apply_grid_lim(ax_a, grid_lim.get("a"), d)
        _apply_grid_lim(ax_b, grid_lim.get("b"), d, ax_a=ax_a)


def plot_tail_causes_single(d, stem):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.01),
                             constrained_layout=True)
    _draw_tail_row(fig, axes[0], axes[1], d, "a", "b")
    save_both(fig, stem)
    plt.close(fig)


def plot_tail_causes_gelu_combined(d_smooth, d_white):
    """GELU two-panel for each forcing: smooth left, white right."""
    stem = "fig_tail_causes_flatness"
    print(f"\n=== {stem} ===")
    fig = plt.figure(figsize=(FIG_WIDTH, 6.20))
    gs = fig.add_gridspec(
        2, 2, left=0.12, right=0.99, top=0.91, bottom=0.14,
        hspace=0.30, wspace=0.22)
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 1])

    for col, d in enumerate((d_smooth, d_white)):
        _draw_tail_row(
            fig, axes[0, col], axes[1, col], d,
            chr(ord("a") + col), chr(ord("c") + col),
            in_axes_legend=False, model_in_legend=True)
        axes[0, col].set_xlabel("")
        if col == 1:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")
    plt.setp(axes[0, 0].get_xticklabels(), visible=False)
    plt.setp(axes[0, 1].get_xticklabels(), visible=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    place_legend(fig, axes, handles, labels, ncol=4)
    save_both(fig, stem)
    plt.close(fig)


def plot_tail_causes_grid(rows, stem):
    n = len(rows)
    if n == 0:
        print("[warn] no tail rows; skip grid")
        return
    print(f"\n=== {stem}  ({n} models × 2) ===")
    fig, axes = plt.subplots(n, 2, figsize=(FIG_WIDTH, 2.35 * n + 0.42),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, d in enumerate(rows):
        _draw_tail_row(
            fig, axes[i, 0], axes[i, 1], d,
            chr(ord("a") + 2 * i), chr(ord("a") + 2 * i + 1),
            in_axes_legend=False, model_in_legend=False,
            model_text=True,
            grid_lim=(TAIL_GRID_LIM.get(d["slug"])
                      if d["forcing"] == "white" else None))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    place_legend(fig, axes, handles, labels, ncol=4, offset_in=0.48)
    save_both(fig, stem)
    plt.close(fig)


def _tail_single_stem(slug, forcing):
    if slug == "gelu_dense":
        return f"fig_tail_causes_flatness_{forcing}"
    return f"fig_tail_causes_flatness_{forcing}_{slug}"


def _tail_stability_check(d):
    keep, r = d.get("keep"), d.get("r")
    if keep is None or r is None:
        print("[stability] skipped (loaded from cache; no Lanczos basis)")
        return
    lam_grid = d["lam_grid"]
    print("\n[stability] rerunning panel-(c) at m=200,400,800")
    model = _load_model(r["run_dir"])
    K_op = _make_kernel_op(model, keep["f_full"], P=keep.get("P"))
    stab_rows = []
    for m_chk in (200, 400, 800):
        nodes_m, w_m, Q_m, V_m, beta0_m = _lanczos(
            K_op, keep["e_full"], m=m_chk, return_basis=True)
        G_m = _G_of(nodes_m, w_m, lam_grid, B_STACK * N)
        lam_m, frac_m = _lam_star_from_G(G_m, lam_grid)
        e_s = _reconstruct_error(nodes_m < lam_m, V_m, Q_m, beta0_m)
        e_f = _reconstruct_error(nodes_m >= lam_m, V_m, Q_m, beta0_m)
        med_s = float(np.median(_modal_profile(e_s)[K_GAP - 1:]))
        med_f = float(np.median(_modal_profile(e_f)[K_GAP - 1:]))
        stab_rows.append((m_chk, lam_m, frac_m, med_s, med_f))
        del Q_m, V_m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"{'m':>6}  {'λ*':>12}  {'G/E':>8}  {'slow gap':>12}  "
          f"{'fast gap':>12}")
    for m_chk, lam_m, frac_m, med_s, med_f in stab_rows:
        print(f"{m_chk:6d}  {lam_m:12.4e}  {100.0 * frac_m:7.1f}%  "
              f"{med_s:12.4e}  {med_f:12.4e}")


def fig_tail_causes(results, *, model_name="GELU Dense", forcing="white",
                    stability_check=False, stem=None, use_cache=True):
    slug, label = _resolve_tail_spec(model_name)
    if stem is None:
        stem = _tail_single_stem(slug, forcing)
    print(f"\n=== {stem} ===")
    d = compute_tail_causes(
        results, model_name=model_name, forcing=forcing, use_cache=use_cache)
    if d is None:
        return
    plot_tail_causes_single(d, stem)
    if stability_check:
        _tail_stability_check(d)


def _compute_gelu_tail_pair(results, use_cache=True):
    pair = {}
    for f in FORCINGS:
        d = compute_tail_causes(
            results, model_name="GELU Dense", forcing=f, use_cache=use_cache)
        if d is not None:
            pair[f] = d
    return pair


def _plot_gelu_tail_pair(pair, *, stability_check=False):
    """4-panel GELU figure when both forcings exist; else the 2-panel."""
    if "smooth" in pair and "white" in pair:
        plot_tail_causes_gelu_combined(pair["smooth"], pair["white"])
        if stability_check:
            for d in (pair["smooth"], pair["white"]):
                _tail_stability_check(d)
        return
    for f, d in pair.items():
        stem = _tail_single_stem("gelu_dense", f)
        print(f"\n=== {stem} ===")
        plot_tail_causes_single(d, stem)
        if stability_check:
            _tail_stability_check(d)


def fig_tail_causes_suite(results, *, forcing="white", use_cache=True,
                          plot_individuals=False, stability_check=False,
                          model_name="all"):
    """GELU 4-panel (smooth | white) plus the per-forcing four-model grid.

    The `_all` grids omit GELU Dense (it has its own figure): one file
    per requested forcing.
    """
    forcings = FORCINGS if forcing == "both" else [forcing]
    if model_name.strip().lower() != "all":
        slug, _ = _resolve_tail_spec(model_name)
        if slug == "gelu_dense":
            pair = (_compute_gelu_tail_pair(results, use_cache=use_cache)
                    if forcing == "both"
                    else {})
            if forcing != "both":
                d = compute_tail_causes(
                    results, model_name=model_name, forcing=forcings[0],
                    use_cache=use_cache)
                if d is not None:
                    pair[forcings[0]] = d
            if pair:
                _plot_gelu_tail_pair(pair, stability_check=stability_check)
            return
        fig_tail_causes(
            results, model_name=model_name, forcing=forcings[0],
            stability_check=stability_check, use_cache=use_cache)
        return

    print(f"\n=== tail suite  models="
          f"{[name for _, name, *_ in TAIL_MODELS]}  forcings={forcings} ===")
    gelu_pair = _compute_gelu_tail_pair(results, use_cache=use_cache)
    _plot_gelu_tail_pair(gelu_pair, stability_check=stability_check)
    for d in gelu_pair.values():
        d.pop("keep", None)
        d.pop("r", None)
    gelu_pair.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for f in forcings:
        rows = []
        for slug, name, *_ in TAIL_MODELS:
            print(f"\n========== {name} / {f} ==========")
            d = compute_tail_causes(
                results, model_name=name, forcing=f, use_cache=use_cache)
            if d is None:
                continue
            if plot_individuals:
                stem = _tail_single_stem(slug, f)
                print(f"\n=== {stem} ===")
                plot_tail_causes_single(d, stem)
            d.pop("keep", None)
            d.pop("r", None)
            rows.append(d)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        plot_tail_causes_grid(
            rows, stem=f"fig_tail_causes_flatness_{f}_all")


K_PHENOMENON_GAP = 16
PHENOMENON_FLOOR_MULT = 3.0


def _phenomenon_stem(forcing):
    return f"fig_phenomenon_operator_{forcing}"


def _phenomenon_shade(ax):
    ax.axvspan(K_PHENOMENON_GAP, N, color="0.92", zorder=0)


def _load_abs_hist(run):
    hist = run.get("modal_abs_err_history") or {}
    ik = sorted(k for k in hist if isinstance(k, (int, np.integer)))
    if not ik:
        return None
    E = np.stack([np.asarray(hist[s], float).reshape(-1)[:N] for s in ik])
    return np.array(ik, float), E


def _milestone_steps(steps, E):
    t0 = 1 if len(steps) > 1 else 0
    L = PHENOMENON_FLOOR_MULT * float(np.median(E[-1][N // 2:]))
    out = np.full(N, np.inf)
    for k in range(N):
        below = np.where(E[t0:, k] <= L)[0]
        if len(below) == 0:
            continue
        i = below[0] + t0
        if i == t0:
            out[k] = steps[t0]
            continue
        y1, y0 = np.log(E[i, k] + 1e-300), np.log(E[i - 1, k] + 1e-300)
        x1, x0 = steps[i], steps[i - 1]
        frac = (np.log(L + 1e-300) - y0) / (y1 - y0 + 1e-300)
        out[k] = x0 + np.clip(frac, 0, 1) * (x1 - x0)
    return out


def _gelu_error_curves(ax, steps, E, karr, norm):
    from matplotlib import cm

    _phenomenon_shade(ax)
    sel = np.unique(np.geomspace(2, len(steps) - 1, 12).astype(int))
    for si in sel:
        ax.loglog(karr, np.maximum(E[si], 1e-18),
                  color=cm.viridis(norm(max(steps[si], 1))), lw=1.0)
    ax.grid(True, alpha=0.3, which="both")
    return sel


def _milestone_scatter(ax, results, forcing, karr):
    _phenomenon_shade(ax)
    max_tc = 0.0
    handles = []
    for slug, name, short, color in MODELS:
        r = next((x for x in results[forcing] if x["name"] == name), None)
        if r is None:
            continue
        h_ = _load_abs_hist(r)
        if h_ is None:
            continue
        steps, E = h_
        tc = _milestone_steps(steps, E)
        finite = np.isfinite(tc)
        ax.plot(karr[finite], tc[finite], ".", ms=3.2, color=color,
                alpha=0.75, ls="none")
        if finite.any():
            max_tc = max(max_tc, float(np.nanmax(tc[finite])))
        handles.append(Line2D([], [], marker=".", ls="none", ms=6,
                              color=color, label=name))
        print(f"[provenance] {name}/{forcing} "
              f"modal_abs_err_history steps {int(steps[0])}..{int(steps[-1])}")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    return handles, max_tc


def fig_phenomenon_operator(results, forcing="both"):
    """GELU error histories and five-model milestone times.

    ``both`` writes the 4-panel paper figure (smooth left, white right).
    A single forcing writes the original 2-panel.
    """
    if forcing == "both":
        _fig_phenomenon_operator_combined(results)
        return
    _fig_phenomenon_operator_one(results, forcing)


def _fig_phenomenon_operator_combined(results):
    """(a,c) smooth; (b,d) white. Top: GELU E_k; bottom: milestone times."""
    stem = "fig_phenomenon_operator"
    print(f"\n=== {stem} ===")
    from matplotlib import cm, colors as mcolors

    karr = np.arange(1, N + 1)
    gelu = {}
    step_vals = []
    for forcing in FORCINGS:
        gd = next((x for x in results[forcing] if x["name"] == "GELU Dense"),
                  None)
        hist = None if gd is None else _load_abs_hist(gd)
        gelu[forcing] = hist
        if hist is None:
            print(f"[warn] GELU Dense {forcing} missing modal_abs_err_history")
            continue
        steps, _ = hist
        sel = np.unique(np.geomspace(2, len(steps) - 1, 12).astype(int))
        step_vals.extend(max(float(steps[si]), 1.0) for si in sel)
        print(f"[provenance] GELU Dense/{forcing} "
              f"modal_abs_err_history steps={list(map(int, steps))}")
    if not step_vals:
        print("[warn] no GELU Dense histories; skip combined phenomenon")
        return

    fig = plt.figure(figsize=(FIG_WIDTH, 6.20))
    gs = fig.add_gridspec(
        2, 2, left=0.11, right=0.88, top=0.91, bottom=0.14,
        hspace=0.30, wspace=0.22)
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 1],
                                sharey=axes[1, 0])

    norm = mcolors.LogNorm(vmin=min(step_vals), vmax=max(step_vals))
    handles, max_tc = [], 0.0
    for col, forcing in enumerate(FORCINGS):
        ax_e = axes[0, col]
        hist = gelu[forcing]
        if hist is not None:
            steps, E = hist
            _gelu_error_curves(ax_e, steps, E, karr, norm)
        panel_label(ax_e, f"({chr(ord('a') + col)})")

        ax_m = axes[1, col]
        hnds, tc = _milestone_scatter(ax_m, results, forcing, karr)
        max_tc = max(max_tc, tc)
        if hnds and not handles:
            handles = hnds
        ax_m.set_xlabel(r"Sine Mode $k$")
        panel_label(ax_m, f"({chr(ord('c') + col)})")

    axes[0, 0].set_ylabel(r"Absolute RMS Modal Error $E_k$")
    axes[1, 0].set_ylabel(r"Step to Reach Floor Level")
    plt.setp(axes[0, 0].get_xticklabels(), visible=False)
    plt.setp(axes[0, 1].get_xticklabels(), visible=False)
    plt.setp(axes[1, 1].get_yticklabels(), visible=False)
    if max_tc > 0:
        ylo, _ = axes[1, 0].get_ylim()
        axes[1, 0].set_ylim(ylo, max_tc * 4.0)

    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    cax = inset_axes(
        axes[0, 1], width="4.5%", height="100%", loc="lower left",
        bbox_to_anchor=(1.06, 0.0, 1.0, 1.0),
        bbox_transform=axes[0, 1].transAxes, borderpad=0)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("GELU Dense Training Step", fontsize=FONT_SMALL)
    cbar.ax.tick_params(labelsize=FONT_SMALL)

    if handles:
        place_legend(fig, axes[1], [h for h in handles],
                     [h.get_label() for h in handles], ncol=3)
    save_both(fig, stem)
    plt.close(fig)


def _fig_phenomenon_operator_one(results, forcing):
    stem = _phenomenon_stem(forcing)
    print(f"\n=== {stem} ===")
    from matplotlib import cm, colors as mcolors

    karr = np.arange(1, N + 1)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.05),
                             gridspec_kw=dict(left=0.10, right=0.98,
                                              top=0.90, bottom=0.16,
                                              wspace=0.32))

    gd = next((x for x in results[forcing] if x["name"] == "GELU Dense"), None)
    if gd is None:
        print(f"[warn] GELU Dense {forcing} missing; skip phenomenon")
        plt.close(fig)
        return
    hist = _load_abs_hist(gd)
    if hist is None:
        print(f"[warn] GELU Dense {forcing} has no modal_abs_err_history; "
              f"skip phenomenon")
        plt.close(fig)
        return
    steps_gd, E_gd = hist
    print(f"[provenance] panel (a): {forcing}_gelu_dense "
          f"modal_abs_err_history steps={list(map(int, steps_gd))}")
    sel = np.unique(np.geomspace(2, len(steps_gd) - 1, 12).astype(int))
    norm = mcolors.LogNorm(vmin=max(steps_gd[sel[0]], 1),
                           vmax=steps_gd[sel[-1]])
    _gelu_error_curves(axes[0], steps_gd, E_gd, karr, norm)
    axes[0].set_xlabel(r"Sine Mode $k$")
    axes[0].set_ylabel(r"Absolute RMS Modal Error $E_k$")
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    fig.colorbar(sm, ax=axes[0], pad=0.02, label="GELU Dense Training Step")
    panel_label(axes[0], "(a)")

    handles, max_tc = _milestone_scatter(axes[1], results, forcing, karr)
    ylo, _ = axes[1].get_ylim()
    if max_tc > 0:
        axes[1].set_ylim(ylo, max_tc * 4.0)
    axes[1].set_xlabel(r"Sine Mode $k$")
    axes[1].set_ylabel(r"Step to Reach Floor Level")
    axes[1].legend(handles=handles, loc="lower left", framealpha=0.92,
                   fontsize=FONT_SMALL, borderaxespad=0.4)
    panel_label(axes[1], "(b)")
    save_both(fig, stem)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
ALL_FIGS = [
    "setup", "gap", "spectra", "diag", "align", "misalign",
    "tail", "phenomenon", "error_loading", "loss",
]
DEFAULT_FIGS = [
    "gap", "spectra", "misalign", "tail", "phenomenon", "loss",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate spectral-bias paper figures.")
    p.add_argument(
        "--only", nargs="+", default=None, metavar="FIG",
        help="Subset of figures to generate. Default: "
             + ", ".join(DEFAULT_FIGS) + ". "
             "Keys: " + ", ".join(ALL_FIGS) + ". "
             "Pass --only all for every key including setup/diag/"
             "align/error_loading. Example: --only gap spectra")
    p.add_argument(
        "--tail-model", default="all",
        help="Tail-causes model (display name or slug), or 'all' "
             "(default): GELU 4-panel plus the full-model suite")
    p.add_argument(
        "--forcing", default=None, choices=["white", "smooth", "both"],
        help="Forcing for the single-column figures (misalign, tail, "
             "phenomenon). Overrides --tail-forcing / --misalign-forcing "
             "/ --phenomenon-forcing")
    p.add_argument(
        "--tail-forcing", default="white",
        choices=["white", "smooth", "both"],
        help="Forcing for the tail-causes figure")
    p.add_argument(
        "--tail-individuals", action="store_true",
        help="Also write a two-panel figure for every tail model, "
             "not only GELU")
    p.add_argument(
        "--recompute", action="store_true",
        help="Ignore cached Lanczos products and recompute the tail "
             "figures")
    p.add_argument(
        "--misalign-forcing", default="white",
        choices=["white", "smooth", "both"],
        help="Forcing for the merged misalignment figure")
    p.add_argument(
        "--phenomenon-forcing", default="both",
        choices=["white", "smooth", "both"],
        help="Forcing for the phenomenon-operator figure. "
             "'both' writes the 4-panel figure (smooth left, white right)")
    p.add_argument(
        "--stability-check", action="store_true",
        help="Extra Lanczos resolution check for the GELU tail figure")
    return p.parse_args()


def _which_figures(only):
    if not only:
        return set(DEFAULT_FIGS)
    if len(only) == 1 and only[0].strip().lower() == "all":
        return set(ALL_FIGS)
    keys = []
    for item in only:
        keys.extend(s.strip().lower() for s in item.split(",") if s.strip())
    which = set(keys)
    unknown = which - set(ALL_FIGS)
    if unknown:
        sys.exit(f"unknown figure keys: {unknown}. Choose from: "
                 + ", ".join(ALL_FIGS))
    return which


def main():
    args = parse_args()
    which = _which_figures(args.only)

    print(f"device={device}  out={OUT_DIR}")
    print(f"MODEL_DICTS_DIR={MODEL_DICTS_DIR}")
    print(f"figures={sorted(which)}")
    configure_geometry()
    results = load_all_runs()

    if "setup" in which:
        fig_setup()
    if "gap" in which:
        fig_spectral_gap(results)
    if "spectra" in which:
        fig_final_spectra(results)
    if "diag" in which:
        fig_diag_failure(results)
    if "align" in which:
        fig_alignment(results)
    if args.forcing is not None:
        args.tail_forcing = args.forcing
        args.misalign_forcing = args.forcing
        args.phenomenon_forcing = args.forcing

    if "misalign" in which:
        mis_forcings = (FORCINGS if args.misalign_forcing == "both"
                        else [args.misalign_forcing])
        for forcing in mis_forcings:
            fig_misalignment(results, forcing=forcing)
    if "tail" in which:
        fig_tail_causes_suite(
            results,
            model_name=args.tail_model,
            forcing=args.tail_forcing,
            use_cache=not args.recompute,
            plot_individuals=args.tail_individuals,
            stability_check=args.stability_check,
        )
    if "phenomenon" in which:
        fig_phenomenon_operator(results, forcing=args.phenomenon_forcing)
    if "error_loading" in which:
        fig_error_loading(results)
    if "loss" in which:
        fig_loss_curves(results)

    print("\n[done] figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
