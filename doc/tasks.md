## Tasks
- `beta_estimates_scatter_EM_balanced_case.pdf` is too sparse
  * Run `./scripts/EM_betas_estimates.py` again
  * Increase to `num_estimation_runs = 1000` independent estimations 
  * To get a denser scatter of joint probability

- Recreate `MLEs_scale_free.pdf`, `empirical_b_k.pdf`, `empirical_lambda_k.pdf`
  * Assuming global functional form is wrong
  * Generalized model fit `di_lauro_bk_model` to higher-order rates in `./src/estimate_total_rates.py` clearly doesn't work
  * Switch to non-parametric KDE-based, maybe kernel regression estimator?
  * Directly estimate `E[S_{K}^{2} | K = k]` using weighted averages of `\tilde{b_k}` values around `k` to get the `b_k_fit` curve
