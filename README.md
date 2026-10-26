# The Dynamics of Spectral Bias in Operator Learning

Code for the paper *The Dynamics of Spectral Bias in Operator Learning*
(C. Ainslie, C.-Y. Lai, A. Chattopadhyay). It trains the five operator models on
the discrete Poisson problem and produces every figure in the paper.

## Files

| File | Purpose |
|---|---|
| `models.py` | The five models: ideal diagonal spectral operator, linear and nonlinear FNO, GELU dense and deep linear dense networks |
| `train_networks.py` | Trains all ten runs (five models, smooth and white forcing) with Adam and saves histories, final weights and the Adam preconditioner |
| `generate_finalized_figures.py` | Kernel diagnostics (stacked-kernel Lanczos, block-diagonal measure, batch-averaged kernel) and the operator-learning figures |
| `paths.py` | Location of the saved runs (`SPECTRAL_BIAS_ROOT` overrides the default) |
| `spectral_bias_scalar/section6_plots.py` | Scalar-regression figures of Appendix A |

## Environment

Python with PyTorch, NumPy and Matplotlib. All runs use double precision. On
NERSC Perlmutter we used the Shifter image `nersc/pytorch:26.01.01`, e.g.
`shifter --image=nersc/pytorch:26.01.01 python train_networks.py`. The Lanczos
diagnostics need a GPU.

## Reproducing the paper

```bash
export SPECTRAL_BIAS_ROOT=/path/to/output   # runs go to $SPECTRAL_BIAS_ROOT/saved_model_dicts
python train_networks.py
python generate_finalized_figures.py --misalign-forcing both --tail-forcing both
cd spectral_bias_scalar && python section6_plots.py   # writes to spectral_bias_scalar/figs/
```

`generate_finalized_figures.py` writes to `finalized_figures/` (set `FIG_OUT_DIR`
to change this) and caches kernels and Lanczos results in `_cache/`. Use
`--only <keys>` to make a subset (`gap spectra misalign tail phenomenon loss`).

| Paper figure | Output file |
|---|---|
| Fig. 1 | `fig_spectral_gap` |
| Fig. 2 | `paper_fig_final_spectra` |
| Fig. 3 | `fig_tail_causes_flatness` |
| Fig. 4 | `paper_fig_misalignment_smooth` |
| Fig. 5 | `fig_phenomenon_operator` |
| Fig. A1 | `spectral_bias_scalar/figs/fig3_depth_geometry` |
| Fig. A2 | `spectral_bias_scalar/figs/fig4_modal_rates` |
| Fig. A3 | `fig_tail_causes_flatness_white_all` |
| Fig. A4 | `fig_tail_causes_flatness_smooth_all` |
| Fig. A5 | `paper_fig_misalignment_white` |
| Fig. A6 | `fig_loss_curves` |
