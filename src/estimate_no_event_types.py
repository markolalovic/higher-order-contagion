""" Estimation of betas from no event types continuously-observed process:
  * state-wise total rates a_k, b_k, i.e.: per-state k rates
  * underlying global rates beta_1, beta_2, i.e.: per-interaction rates  
"""

import numpy as np
import matplotlib.pylab as plt

from scipy.optimize import minimize
from scipy.optimize import least_squares

def sk1(k, N):
    r""" Calculates structural count $S_{k}^{(1)}$ for CSC. """
    if k <= 0 or k >= N:
        return 0
    return k * (N - k)

def sk2(k, N):
    r""" Calculates structural count $S_{k}^{(2)}$ for CSC. """
    if k < 2 or k >= N:
        return 0
    return 0.5 * k * (k - 1) * (N - k)

def lambda_k(k, N, beta1, beta2):
    r""" Calculates birth rate from state k for CSC. """
    return beta1 * sk1(k, N) + beta2 * sk2(k, N)

def get_sufficient_stats(trajectory, N):
    r"""
    Calculates the sufficient statistics: `T_k, B_k, D_k` from a single run (trajectory):
        - trajectory[0, :] are event times
        - trajectory[2, :] are states (total infected)
    """
    times = trajectory[0]
    states = trajectory[2].astype(int)

    T_k = np.zeros(N + 1)
    B_k = np.zeros(N + 1)
    D_k = np.zeros(N + 1)

    for i in range(len(times) - 1):
        current_k = states[i]
        
        time_spent = times[i + 1] - times[i]
        T_k[current_k] += time_spent

        # determine birth or death event
        next_k = states[i + 1]
        if next_k > current_k:
            # then it was a birth event from current_k state
            B_k[current_k] += 1
        elif next_k < current_k:
            # it was a death event from current_k state
            D_k[current_k] += 1
        # else: it is the final record at time_max not an event
        
    return {'T_k': T_k, 'B_k': B_k, 'D_k': D_k}

def calculate_log_likelihood(params, stats, N, mu):
    r"""
    Calculates the log-likelihood for continuously observed data without event types.
    """
    beta1, beta2 = params
    T_k, B_k, D_k = stats['T_k'], stats['B_k'], stats['D_k']
    
    logL = 0
    for k in range(N + 1):
        if T_k[k] > 0: # only states that were visited
            lambda_val = lambda_k(k, N, beta1, beta2)
            mu_val = mu * k
            # birth term
            if B_k[k] > 0:
                if lambda_val > 1e-12: # avoid log(0)
                    logL += B_k[k] * np.log(lambda_val)
                else: # zero rate: probability is zero
                    return -np.inf 
            
            # TODO: death term is not needed here
            if D_k[k] > 0:
                if mu_val > 1e-12:
                    logL += D_k[k] * np.log(mu_val)
                else:
                    return -np.inf
            
            # waiting time term
            logL -= (lambda_val + mu_val) * T_k[k]
            
    return logL

def compute_likelihood_surface(b1_range, b2_range, stats, N, mu):
    r"""
    Computes the log-likelihood over a grid of scaled beta parameters
    given:
      - b1_range: (min, max, steps) for scaled beta1
      - b2_range: (min, max, steps) for scaled beta2
      - stats: observed (sufficient) statistics T_k, B_k, D_k
    """
    beta1_vec = np.linspace(*b1_range)
    beta2_vec = np.linspace(*b2_range)
    
    logL_surface = np.zeros((b1_range[2], b2_range[2]))

    for i, b1 in enumerate(beta1_vec):
        for j, b2 in enumerate(beta2_vec):
            b1_orig = b1 / N
            b2_orig = b2 / (N**2)
            logL_surface[i, j] = calculate_log_likelihood(
                [b1_orig, b2_orig], stats, N, mu
            )
    # transpose for plotting (X = beta1, Y = beta2)
    logL_surface = logL_surface.T
    
    return {
        "beta1_scaled_vec": beta1_vec,
        "beta2_scaled_vec": beta2_vec,
        "logL_surface": logL_surface
    }

def plot_zoomed_likelihood(surface_data, true_params, threshold=-5.0, save_it=False):
    r"""
    Zoomed-in contour plot of the log-likelihood surface.
    threshold defines the distance from max for the zoom-in region.
    """
    b1_vec = surface_data["beta1_scaled_vec"]
    b2_vec = surface_data["beta2_scaled_vec"]
    logL = surface_data["logL_surface"]
    
    b1_true, b2_true = true_params
    
    # determine grid MLE and high-likelihood region
    logL_max = np.nanmax(logL)
    max_loc = np.unravel_index(np.nanargmax(logL), logL.shape)
    mle_b2 = b2_vec[max_loc[0]]
    mle_b1 = b1_vec[max_loc[1]]
    
    # indices within the threshold of the max
    high_likelihood_indices = np.argwhere(logL >= logL_max + threshold)
    
    # zoom-in boundaries from the indices
    min_b2_idx, min_b1_idx = high_likelihood_indices.min(axis=0)
    max_b2_idx, max_b1_idx = high_likelihood_indices.max(axis=0)
    
    # padding to the boundaries
    padding_b1 = (b1_vec[1] - b1_vec[0]) * 2
    padding_b2 = (b2_vec[1] - b2_vec[0]) * 2
    
    xlim = (b1_vec[min_b1_idx] - padding_b1, b1_vec[max_b1_idx] + padding_b1)
    ylim = (b2_vec[min_b2_idx] - padding_b2, b2_vec[max_b2_idx] + padding_b2)

    # -------------------
    # --- Plotting ------
    # -------------------
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    
    # relative scale for colors
    relative_logL = logL - logL_max
    
    contour = ax.contourf(b1_vec, b2_vec, relative_logL,
                          levels=np.linspace(threshold, 0, 21),
                          cmap='viridis')

    ax.contour(b1_vec, b2_vec, relative_logL,
               levels=np.linspace(threshold, 0, 5), 
               colors='white', linewidths=0.5, alpha=0.7)

    # true values and grid MLE as red X and blue X
    ax.plot(b1_true, b2_true, 'rx', markersize=12, markeredgewidth=2.5, label='True Value')
    ax.plot(mle_b1, mle_b2, 'bx', markersize=12, markeredgewidth=2, label='Grid MLE')
    
    ax.set_xlabel(r'Scaled Pairwise Rate ($\beta_1 N$)', fontsize=14)
    ax.set_ylabel(r'Scaled Higher-Order Rate ($\beta_2 N^2$)', fontsize=14)
    
    # apply zoom-in
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.legend(fontsize=12)
    fig.colorbar(contour, label=r'$\ell(\beta_1, \beta_2) - \ell_{max}$')
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()

    if save_it:
        output_filename = "../figures/inference/no_event_log_likelihood_contour.pdf"
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    else: 
        plt.show()

def grid_search(stats, N, mu):
    r"""
    Finds the MLEs for beta1 and beta2 using a grid search given:
    - stats: sufficient statistics T_k, B_k, D_k
    """
    # define the grid for the scaled parameters
    n_steps = 50 # TODO: increase to 100
    beta1_s_min, beta1_s_max, beta1_s_steps = 0, 8.0, n_steps
    beta2_s_min, beta2_s_max, beta2_s_steps = 0, 15.0, n_steps

    beta1_scaled_vec = np.linspace(beta1_s_min, beta1_s_max, beta1_s_steps)
    beta2_scaled_vec = np.linspace(beta2_s_min, beta2_s_max, beta2_s_steps)
    
    # compute log-likelihood over the grid
    logL_surface = np.zeros((beta1_s_steps, beta2_s_steps))
    
    for i, b1_s in enumerate(beta1_scaled_vec):
        for j, b2_s in enumerate(beta2_scaled_vec):
            # convert scaled params to original for likelihood calculation
            b1_orig = b1_s / N
            b2_orig = b2_s / (N**2) 
            logL_surface[i, j] = calculate_log_likelihood([b1_orig, b2_orig], stats, N, mu)

    # find MLE on the evaluated grid
    max_flat_idx = np.nanargmax(logL_surface)
    max_idx_b1, max_idx_b2 = np.unravel_index(max_flat_idx, logL_surface.shape)
    
    beta1_hat = beta1_scaled_vec[max_idx_b1]
    beta2_hat = beta2_scaled_vec[max_idx_b2]
    
    return beta1_hat, beta2_hat

def estimate_dlm(stats, N, initial_guess):
    """
    Finds the MLE for beta1 and beta2 by direct maximization of the log-likelihood.
    Done by minimizing the negative log-likelihood using an optimizer given:
      - suff stats 
      - initial guess TODO: could come from two-stage regression?
    """
    T_k = stats['T_k']
    B_k = stats['B_k']
    # D_k contribution to the likelihood wrt. beta is constant

    # objective function: 
    # the negative log-likelihood part that depends on beta1, beta2
    def neg_log_likelihood(params):
        beta1, beta2 = params
        # must be non-negative
        if beta1 < 0 or beta2 < 0:
            return np.inf

        logL = 0
        for k in range(N + 1):
            if T_k[k] > 0:
                lambda_val = lambda_k(k, N, beta1, beta2)
                
                # contribution from observed births
                if B_k[k] > 0:
                    if lambda_val > 1e-12:
                        logL += B_k[k] * np.log(lambda_val)
                    else:
                        print("Birth observed but rate is 0!")
                        return np.inf
                
                # contribution from time spent in state k (exposure time)
                logL -= lambda_val * T_k[k]
                
        return -logL

    # constrained optimization: non-negative rates
    # 'L-BFGS-B' supports bounds
    result = minimize(
        neg_log_likelihood,
        x0=initial_guess,
        method='L-BFGS-B',
        bounds=[(1e-12, None), (1e-12, None)]
    )

    if not result.success:
        print(f"DLM did not converge: {result.message}")
    
    beta1_scaled, beta2_scaled = result.x
    beta1_hat = beta1_scaled * N
    beta2_hat = beta2_scaled * (N**2) 

    return beta1_hat, beta2_hat

def estimate_tsr(stats, N, initial_guess):
    r"""
    Two-stage regression using weighted least squares:
        1. stage: estimate birth rates $\lambda_k$ for each state $k$ visited
        2. stage: fit $\beta_1 f_1(k) + \beta_2 f_2(k)$ to $(k, \lambda_k)$ points
    
    Approximate weights as $T_k$ since we do not know the true $\lambda_k$.

    Initial guess:
      * x0 = [0.01 / N, 0.01 / (N**2)]
      * x0 = [1 / N, 1 / (N**2)]
    """
    T_k = stats['T_k']
    B_k = stats['B_k']
    
    # Stage 1: estimate lambda_k for each k where T_k > 0
    # consider only states  k \geq 2 since S_k^{2} = 0 otherwise
    valid_k_indices = np.where((T_k > 1e-9) & (np.arange(N + 1) >= 2))[0]
            
    k_vals = valid_k_indices
    lambda_hat_k = B_k[k_vals] / T_k[k_vals]

    # Stage 2: weighted least squares regression
    # minimize sum of weighted squared residuals 
    # objective_fn = weights * (observed - predicted)^2
    
    # calculate predictors / features
    f1_vals = np.array([sk1(k, N) for k in k_vals])
    f2_vals = np.array([sk2(k, N) for k in k_vals])
    
    # define the weights 
    # use T_k as it is proportional to the inverse of variance
    weights = T_k[k_vals]

    # objective function
    def weighted_residuals(params):
        beta1, beta2 = params
        predicted_lambda = beta1 * f1_vals + beta2 * f2_vals
        residuals = lambda_hat_k - predicted_lambda
        return np.sqrt(weights) * residuals

    # use least_squares 
    # constrain parameters to non-negative values
    result = least_squares(
        weighted_residuals,
        initial_guess,
        bounds=([0, 0], [np.inf, np.inf])
    )

    if not result.success:
        print("Weighted least-squares failed.")
    
    beta1_scaled, beta2_scaled = result.x
    beta1_hat = beta1_scaled * N
    beta2_hat = beta2_scaled * (N**2) 

    return beta1_hat, beta2_hat

def estimate_em(stats, N, initial_guess, max_iter=2000, tol=1e-9):
    r"""
    Derived EM algorithm to estimate betas, given:
      - stats: sufficient statistics T_k, B_k, D_k
      - initial_guess
      - max_iter: maximum number of iterations
      - tol: tolerance for checking convergence
    """
    T_k = stats['T_k']
    B_k = stats['B_k']
    
    # direct translation
    # denominators for the M-step are constants
    # sum_k S_k^(1) * T_k
    denom1 = np.sum([sk1(k, N) * T_k[k] for k in range(N + 1)])
    # sum_k S_k^(2) * T_k
    denom2 = np.sum([sk2(k, N) * T_k[k] for k in range(N + 1)])

    # check they are not zero
    if denom1 <= 1e-12 or denom2 <= 1e-12:
        print("Can not estimate: zero exposure time in the denominator.")
        return np.nan, np.nan

    beta1_m, beta2_m = initial_guess 
    for i in range(max_iter):
        beta1_old, beta2_old = beta1_m, beta2_m
        
        # E-Step: calculate conditional expected counts
        exp_U_k = np.zeros(N + 1) # E[U_k]
        exp_V_k = np.zeros(N + 1) # E[V_k]
        for k in range(1, N):
            if B_k[k] > 0: # only states from which births occured
                a_k = beta1_m * sk1(k, N)
                b_k = beta2_m * sk2(k, N)
                lambda_total = lambda_k(k, N, beta1_m, beta2_m)
                
                if lambda_total > 1e-12:
                    # E[U_k] = B_k * P(event is PW)
                    exp_U_k[k] = B_k[k] * (a_k / lambda_total)

                    # E[V_k] = B_k * P(event is HO)
                    exp_V_k[k] = B_k[k] * (b_k / lambda_total)

        # M-Step: update parameters using the expected counts
        beta1_m = np.sum(exp_U_k) / denom1
        beta2_m = np.sum(exp_V_k) / denom2
        
        # check for convergence: relative change in parameters
        if abs(beta1_m - beta1_old) < tol and abs(beta2_m - beta2_old) < tol:
            # print(f"EM converged in {i + 1} iterations.")
            break
    
    beta1_hat = beta1_m * N
    beta2_hat = beta2_m * (N**2) 

    return beta1_hat, beta2_hat