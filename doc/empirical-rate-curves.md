## Models used to fit empirical infection rate curves $\widetilde{a}_k$ and $\widetilde{b}_k$

Generalized model fits, inspired by Di Lauro et al. "Network inference from population-level observation of epidemics." Scientific Reports 10.1 (2020): 18779.

To get the approximate functional forms for state-dependent infection rates, let

$$
\tilde{a}_k = C_a \cdot k^{p_a} \cdot (N - k)^{p_a} 
\cdot \exp\left(\alpha_a \frac{2k - N}{N}\right) 
$$

$$
\tilde{b}_k = C_b \cdot k \cdot (k - 1)^{p_b} 
\cdot (N-k)^{p_b} \cdot \exp\left(\alpha_b \frac{2k - N}{N}\right)
$$

where:
- $C > 0$ is the scaling parameter
- $p > 0$ controls the shape, the rise and fall, how sharp the transitions are
- $\alpha$ controls the skewness

To get the $\tilde{a}$, $\tilde{b}$ curves, we find the values of $(C_a, p_a, \alpha_a)$, and $(C_b, p_b, \alpha_b)$ using non-linear least squares with the `curve_fit` function from the `scipy.optimize` library in Python.

With bounds for the parameter: $C_a, C_b \ge 0$, and $p_a, p_b > 0$.

The models are implemented in functions `di_lauro_ak_model`, `di_lauro_bk_model` in `./src/estimate_total_rates.py`.

Fitting is done in `./scripts/estimates_export.py`.