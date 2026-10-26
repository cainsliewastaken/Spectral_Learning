"""Train the paper model suite (1D Poisson operator learning).

Artifacts are written to SAVE_ROOT (see paths.py) and read by
generate_finalized_figures.py.

    shifter python train_networks.py
"""

import gc
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call, jacrev

from models import (
    GELUDense,
    IdealDiagonalSpectralOperator,
    LinearDense,
    LinearFNO1D,
    NonlinearFNO1D,
)
from paths import SAVE_ROOT

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
torch.manual_seed(0)

# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------
N = 128
steps = 12000
N_FINAL_NTK = 100
BATCH_SIZE = 512
NTK_BATCH_SIZE = None  # defaults to min(32, BATCH_SIZE)
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
COSINE_ETA_MIN_RATIO = 0.01
GRAD_CLIP_NORM = 1.0
save_steps = [
    0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000,
    3000, 4000, 5000, 6000, 8000, 10000, 12000,
]

SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Discrete Poisson operator and sine basis
# ------------------------------------------------------------
h = 1.0 / (N + 1)
A = torch.zeros((N, N), dtype=dtype, device=device)
for i in range(N):
    A[i, i] = 2.0 / h ** 2
    if i > 0:
        A[i, i - 1] = -1.0 / h ** 2
    if i < N - 1:
        A[i, i + 1] = -1.0 / h ** 2
A_inv = torch.linalg.inv(A)

j = torch.arange(1, N + 1, dtype=dtype, device=device).view(-1, 1)
k = torch.arange(1, N + 1, dtype=dtype, device=device).view(1, -1)
S = torch.sqrt(torch.tensor(2.0 / (N + 1), dtype=dtype, device=device)) * torch.sin(
    math.pi * j * k / (N + 1)
)
print("device =", device)
print("SAVE_ROOT =", SAVE_ROOT)
print("||S^T S - I|| =", torch.norm(S.T @ S - torch.eye(N, dtype=dtype, device=device)).item())


def generate_forcing_batch(batch_size, forcing_type="white"):
    coeffs = torch.randn(batch_size, N, dtype=dtype, device=device)
    if forcing_type == "white":
        scales = torch.ones(N, dtype=dtype, device=device)
    elif forcing_type == "smooth":
        scales = 1.0 / (torch.arange(1, N + 1, dtype=dtype, device=device) ** 2)
    else:
        raise ValueError("forcing_type must be 'white' or 'smooth'")
    return (coeffs * scales.view(1, -1)) @ S.T


def modal_relative_error(u_pred, u_true):
    err_modal = (u_pred - u_true) @ S
    true_modal = u_true @ S
    num = torch.mean(err_modal ** 2, dim=0)
    den = torch.mean(true_modal ** 2, dim=0) + 1e-16
    return torch.sqrt(num / den).detach().cpu()


def modal_absolute_error(u_pred, u_true):
    err_modal = (u_pred - u_true) @ S
    return torch.sqrt(torch.mean(err_modal ** 2, dim=0)).detach().cpu()


def _params_and_buffers(model, params, buffers):
    if params is None:
        params = dict(model.named_parameters())
    if buffers is None:
        buffers = dict(model.named_buffers())
    return params, buffers


def _jacobian_matrix(jac_dict, params):
    parts = [jac_dict[name].reshape(jac_dict[name].shape[0], -1) for name in params]
    return torch.cat(parts, dim=1)


def batched_ntk_jjt(model, f_batch, *, params=None, buffers=None):
    """Per-input NTK Θ = J J^T, J = ∂u/∂θ. Shape (B, N_out, N_out)."""
    params, buffers = _params_and_buffers(model, params, buffers)
    grams = []
    for i in range(f_batch.shape[0]):
        f_i = f_batch[i]

        def u_of_params(p):
            return functional_call(
                model, {**p, **buffers}, (f_i.unsqueeze(0),)
            ).squeeze(0)

        jac = jacrev(u_of_params)(params)
        G = _jacobian_matrix(jac, params)
        grams.append(G @ G.T)
    return torch.stack(grams, dim=0)


def _adam_precond_diag(model, optimizer):
    """Diagonal Adam map P such that δθ ≈ −P g if m were replaced by g.

    P_ii = (lr / (1−β1^t)) / (sqrt(v_i)/sqrt(1−β2^t) + ε), in
    ``named_parameters()`` order. Momentum is not a diagonal operator on
    the current gradient; this is the second-moment / step-size metric.
    """
    pg = optimizer.param_groups[0]
    lr = float(pg["lr"])
    beta1, beta2 = pg["betas"]
    eps = float(pg["eps"])
    chunks = []
    for _, p in model.named_parameters():
        st = optimizer.state.get(p)
        n = p.numel()
        if st is None or "exp_avg_sq" not in st:
            chunks.append(torch.full((n,), lr, dtype=dtype, device="cpu"))
            continue
        step = st["step"]
        t = int(step.item() if torch.is_tensor(step) else step)
        t = max(t, 1)
        v = st["exp_avg_sq"].detach().to(dtype=dtype, device="cpu").reshape(-1)
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        denom = v.sqrt() / math.sqrt(bc2) + eps
        chunks.append((lr / bc1) / denom)
    P = torch.cat(chunks)
    print(f"  adam P: lr={lr:g}  n={P.numel()}  "
          f"min={float(P.min()):.3e}  median={float(P.median()):.3e}  "
          f"max={float(P.max()):.3e}")
    return P


def _empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model(
    model,
    forcing_type="smooth",
    steps=3000,
    batch_size=512,
    lr=5e-5,
    save_steps=None,
    name="model",
    ntk_batch_size=None,
    n_final_ntk=None,
):
    model = model.to(device=device, dtype=dtype)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=ADAM_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * COSINE_ETA_MIN_RATIO, last_epoch=-1
    )
    loss_fn = nn.MSELoss()

    if save_steps is None:
        save_steps = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, steps]
    if ntk_batch_size is None:
        ntk_batch_size = min(32, batch_size)
    n_final_ntk = N_FINAL_NTK if n_final_ntk is None else int(n_final_ntk)
    if n_final_ntk < 1 or n_final_ntk > steps:
        raise ValueError(f"n_final_ntk must be in [1, {steps}]")
    ntk_steps = set(range(steps - n_final_ntk + 1, steps + 1))

    f_val = generate_forcing_batch(4096, forcing_type=forcing_type)
    u_val_true = f_val @ A_inv.T
    f_ntk = generate_forcing_batch(ntk_batch_size, forcing_type=forcing_type)

    loss_history = []
    modal_err_history = {}
    modal_abs_err_history = {}
    output_history = {}
    truth_history = {}
    ntk_history = {}

    with torch.no_grad():
        u0 = model(f_val)
        modal_err_history[0] = modal_relative_error(u0, u_val_true)
        modal_abs_err_history[0] = modal_absolute_error(u0, u_val_true)
        ntk_history[0] = batched_ntk_jjt(model, f_ntk).detach().cpu()

    for step in range(1, steps + 1):
        f_batch = generate_forcing_batch(batch_size, forcing_type=forcing_type)
        u_true = f_batch @ A_inv.T

        optimizer.zero_grad()
        u_pred = model(f_batch)
        loss = loss_fn(u_pred, u_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        save_diag = step in save_steps
        save_ntk = step in ntk_steps
        if save_diag or save_ntk:
            with torch.no_grad():
                if save_diag:
                    u_val_pred = model(f_val)
                    modal_err_history[step] = modal_relative_error(u_val_pred, u_val_true)
                    modal_abs_err_history[step] = modal_absolute_error(u_val_pred, u_val_true)
                    output_history[step] = u_val_pred.detach().cpu()
                    truth_history[step] = u_val_true.detach().cpu()
                if save_ntk:
                    ntk_history[step] = batched_ntk_jjt(model, f_ntk).detach().cpu()
            _empty_cache()

    adam_P = _adam_precond_diag(model, optimizer)
    return {
        "model": model,
        "loss_history": loss_history,
        "modal_err_history": modal_err_history,
        "modal_abs_err_history": modal_abs_err_history,
        "save_steps": save_steps,
        "n_final_ntk": n_final_ntk,
        "forcing_type": forcing_type,
        "name": name,
        "ntk_history": ntk_history,
        "output_history": output_history,
        "truth_history": truth_history,
        "adam_P": adam_P,
        "adam_lr": float(optimizer.param_groups[0]["lr"]),
        "adam_step": steps,
    }


def _to_cpu_serializable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_serializable(v) for v in obj)
    return obj


def save_model_dictionaries(result, run_slug):
    run_dir = SAVE_ROOT / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    for key, value in result.items():
        if key == "model" and isinstance(value, nn.Module):
            torch.save(value.to("cpu"), run_dir / f"{key}.pt")
        elif isinstance(value, (dict, list, tuple)):
            torch.save(_to_cpu_serializable(value), run_dir / f"{key}.pt")
        elif key in ("n_final_ntk", "forcing_type", "name", "adam_lr", "adam_step"):
            torch.save(value, run_dir / f"{key}.pt")
        elif torch.is_tensor(value):
            torch.save(value.detach().cpu(), run_dir / f"{key}.pt")


def _max_saved_training_step(run_dir):
    for fname in ("save_steps.pt", "loss_history.pt", "modal_err_history.pt", "ntk_history.pt"):
        fp = run_dir / fname
        if not fp.exists():
            continue
        obj = torch.load(fp, map_location="cpu")
        if not obj:
            continue
        if fname == "loss_history.pt":
            return int(len(obj))
        if isinstance(obj, dict):
            return int(max(obj.keys()))
        return int(max(obj))
    return None


def train_save_and_discard(model, forcing_type, lr, name, run_slug, skip_if_exists=True):
    run_dir = SAVE_ROOT / run_slug
    if skip_if_exists and (run_dir / "ntk_history.pt").exists():
        max_saved = _max_saved_training_step(run_dir)
        have_P = (run_dir / "adam_P.pt").exists()
        if max_saved is not None and max_saved >= steps and have_P:
            print(f"Skipping {name}: artifacts already exist in {run_dir}")
            return
        if not have_P:
            print(f"Re-training {name}: missing adam_P.pt")
        elif max_saved is not None:
            print(f"Re-training {name}: existing artifacts stop at step {max_saved}")

    save_steps_run = sorted({int(s) for s in save_steps if 0 <= int(s) <= steps} | {0, steps})
    print(
        f"Training {name} (Adam, lr={lr}, batch={BATCH_SIZE}, "
        f"steps={steps}, N_FINAL_NTK={N_FINAL_NTK}) ..."
    )
    result = train_model(
        model,
        forcing_type=forcing_type,
        steps=steps,
        batch_size=BATCH_SIZE,
        lr=lr,
        save_steps=save_steps_run,
        name=name,
        ntk_batch_size=NTK_BATCH_SIZE,
        n_final_ntk=N_FINAL_NTK,
    )
    save_model_dictionaries(result, run_slug)
    print(
        f"Saved {name} in {SAVE_ROOT / run_slug} "
        f"({len(result['ntk_history'])} NTK snapshots)"
    )
    del result
    _empty_cache()
    gc.collect()


def main():
    for forcing in ("white", "smooth"):
        train_save_and_discard(
            LinearDense(N, width=512, depth=4), forcing, 5e-5,
            f"Linear Dense ({forcing})", f"{forcing}_linear_dense",
            skip_if_exists=False)
        train_save_and_discard(
            GELUDense(N, width=512, depth=4), forcing, 5e-5,
            f"GELU Dense ({forcing})", f"{forcing}_gelu_dense",
            skip_if_exists=False)
        train_save_and_discard(
            IdealDiagonalSpectralOperator(N, S=S, K_retained=32), forcing, 1e-2,
            f"Ideal Diagonal Spectral ({forcing})", f"{forcing}_ideal_diagonal_spectral")
        train_save_and_discard(
            LinearFNO1D(N, S=S, width=32, layers=3, K_retained=N), forcing, 5e-5,
            f"Linear FNO ({forcing})", f"{forcing}_linear_fno_k64",
            skip_if_exists=False)
        train_save_and_discard(
            NonlinearFNO1D(N, S=S, width=32, layers=3, K_retained=N), forcing, 5e-5,
            f"Nonlinear FNO ({forcing})", f"{forcing}_nonlinear_fno_k64",
            skip_if_exists=False)


if __name__ == "__main__":
    main()
