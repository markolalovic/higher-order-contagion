import numpy as np
from scipy.integrate import solve_ivp
import pickle

from scipy.special import comb
from scipy.sparse import diags # for sparse diagonal matrices

def precompute_complete_graph_arrays(N):
    """ Precomputes arrays for vectorized ODE."""
    k_vals = np.arange(N + 1)
    
    s1_k = np.zeros(N + 1, dtype=np.float64)
    s2_k = np.zeros(N + 1, dtype=np.float64)

    # s1(k) = k * (N - k)
    s1_k = k_vals * (N - k_vals)

    # s2(k) = binom(k, 2) * (N - k)
    # if k < 2 comb(k, 2) is 0
    valid_k = k_vals >= 2
    s2_k[valid_k] = comb(k_vals[valid_k], 2, exact = False) * (N - k_vals[valid_k])

    return k_vals, s1_k, s2_k

def ode_system_vectorized(t, p, N, beta1, beta2, mu, k_vals, s1_k, s2_k):
    """ Vectorized ODE system for complete graph. """
    dpdt = np.zeros_like(p) # p = [p0, p1, ..., pk, ..., pN]

    # calculate rates using precomputed s1, s2 arrays
    rate_a = beta1 * s1_k
    rate_b = beta2 * s2_k
    infection_rate_k = rate_a + rate_b # total infection rate from state k

    recovery_rate_k = mu * k_vals # total recovery rate from state k

    # flows between states
    # from k - 1 (infection): for p[1] to p[N]
    inflow_inf = infection_rate_k[:-1] * p[:-1] # rates from k = 0, ..., N-1 times p[0], ...., p[N - 1]

    # from k + 1 (recovery): for p[0] to p[N - 1]
    inflow_rec = recovery_rate_k[1:] * p[1:] # rates from k = 1, ..., N times p[1], ..., p[N]

    # from k (infection and recovery) - for p[1] to p[N - 1] for both
    # and specific terms for p[0] and p[N]
    outflow_k = infection_rate_k + recovery_rate_k # Total outflow rate from state k

    # construct the dpdt vector
    # states k = 1 to N - 1
    dpdt[1:-1] = inflow_inf[1:] + inflow_rec[:-1] - outflow_k[1:-1] * p[1:-1]

    # boundary state k = 0
    # dp0/dt = c_1 * p_1 (only inflow from recovery)
    # outflow_k[0] is 0 because k = 0 (no recovery) and s1(0) = s2(0) = 0 (no infection)
    dpdt[0] = inflow_rec[0] # = recovery_rate_k[1] * p[1] = mu * 1 * p[1]

    # boundary state k = N
    # dpN/ dt = (a_{N - 1} + b_{N - 1}) * p_{N - 1} - c_N * p_N
    # outflow_k[N] is only recovery_rate_k[N] because s1(N) = s2(N) = 0 (no sus nodes)
    dpdt[N] = inflow_inf[-1] - outflow_k[N] * p[N] # inflow_inf[-1] is rate from N - 1 * p[N - 1]

    return dpdt

def jacobian_complete_sparse(t, p, N, beta1, beta2, mu, k_vals, s1_k, s2_k):
    """Constructs sparse tridiagonal Jacobian matrix. """

    # calculate rate components ~~ ode_system_vectorized
    rate_a = beta1 * s1_k
    rate_b = beta2 * s2_k
    infection_rate_k = rate_a + rate_b  # R(k) term coefficient in dp_{k+1}/dt (rate from k)
    recovery_rate_k = mu * k_vals       # D(k) term coefficient in dp_{k-1}/dt (rate from k)
    outflow_k = infection_rate_k + recovery_rate_k # O(k) term coefficient in dp_k/dt (rate from k)

    # main diagonal -O(k) = -(infection_rate_k + recovery_rate_k)
    diag_main = -outflow_k

    # upper diagonal: D(k + 1) = recovery_rate_k[k + 1] = mu * (k + 1) for k= 0, ..., N - 1
    diag_upper = recovery_rate_k[1:] # for k = 1 to N

    # lower diagonal: R(k - 1) = infection_rate_k[k - 1] for k = 1, ..., N
    diag_lower = infection_rate_k[:-1] # for k = 0 to N - 1

    # create sparse matrix using `scipy.sparse.diags`
    # offsets = [0, 1, -1] for main, upper, lower diagonals
    jacobian = diags(
      [diag_main, diag_upper, diag_lower],
      [0, 1, -1],
      shape=(N + 1, N + 1),
      format='csc' # efficient format for solvers
    )
    return jacobian


if __name__ == "__main__":
    # setup
    # TODO: no need for graph object
    g_type = "complete"
    N = 1000

    I0 = 5
    time_max = 20
    mu = 1.0

    beta1_steps, beta2_steps = 10, 30
    beta1_span = (1, 8)
    beta2_span = (1, 16)

    beta1_vec_unscaled = np.linspace(beta1_span[0], beta1_span[1], beta1_steps)
    beta2_vec_unscaled = np.linspace(beta2_span[0], beta2_span[1], beta2_steps)

    # scale parameters 
    beta1_vec = beta1_vec_unscaled / N
    beta2_vec = beta2_vec_unscaled / (N**2)

    M = N + 1 # number of states

    # precompute N dependent arrays once only
    print("Precomputing arrays ...")
    k_vals, s1_k, s2_k = precompute_complete_graph_arrays(N)
    print("Done.")

    # set initial condition
    p0 = np.zeros(M, dtype=np.float64) # float64 for solver
    p0[I0] = 1.0
    print(f"p0 = {p0[:20]} ...")

    # time range and eval times to evaluate solution
    nsteps = 101
    t_span = (0.0, time_max)
    t_eval = np.linspace(t_span[0], t_span[1], nsteps)

    solve_for_betas = True
    file_path = f"../results/KE_solutions_G={g_type}_N={N}_I0={I0}_optimized.pickle"

    if solve_for_betas:
        solutions = {}
        start_time = time.time()
        total_solves = len(beta1_vec) * len(beta2_vec)
        count = 0

        print(f"Starting ODE solves for {total_solves} parameter pairs...")
        for i, beta1 in enumerate(beta1_vec):
            for j, beta2 in enumerate(beta2_vec):
                count += 1
                if count % 10 == 0: # print progress update
                    elapsed = time.time() - start_time
                    est_total = (elapsed / count) * total_solves
                    print(f"\t Solving {count}/{total_solves}... elapsed = {elapsed:.1f}, estimated total = {est_total:.1f}")

                # solve using vectorized function and sparse Jacobian
                sol = solve_ivp(
                    ode_system_vectorized,
                    t_span,
                    p0,
                    method="LSODA",
                    t_eval=t_eval,
                    jac=jacobian_complete_sparse,
                    args=(N, beta1, beta2, mu, k_vals, s1_k, s2_k), # args to ODE and Jacobian
                    rtol=1e-6, # relative tolerance
                    atol=1e-9 # absolute tolerance
                )
                solutions[str((i, j))] = sol

        end_time = time.time()
        print(f"\n Finished {total_solves} ODE solves in {end_time - start_time:.2f} seconds")

        # save the solutions
        with open(file_path, "wb") as f:
            pickle.dump(solutions, f)
        print(f"Saved solutions to {file_path}")    
    else:
        # load solutions
        with open(file_path, "rb") as f:
            solutions = pickle.load(f)
    
    # TODO: draw expected curves and contour plot