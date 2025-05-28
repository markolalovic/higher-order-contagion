# Figures
```
    ./figures/combined
    ├── MF_decomposition.pdf
    ├── beta_estimates_scatter_EM_balanced_case.pdf
    ├── beta_estimates_scatter_MLE_balanced_case.pdf
    ├── beta_estimates_scatter_MLE_hard_case.pdf
    ├── estimates
    │   ├── MLEs_complete.pdf
    │   ├── MLEs_random_ER.pdf
    │   ├── MLEs_regular.pdf
    │   ├── MLEs_scale_free.pdf
    │   ├── empirical_a_k.pdf
    │   ├── empirical_b_k.pdf
    │   └── empirical_lambda_k.pdf
    ├── gillespie_runs_all_classes.pdf
    ├── k_star_contour.pdf
    ├── kolmogorov_EM_fits_comparison_1.pdf
    ├── kolmogorov_EM_fits_comparison_2.pdf
    ├── kolmogorov_solutions.pdf
    └── todo
        └── gillespie-sims-hard-pairs.pdf
```

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
