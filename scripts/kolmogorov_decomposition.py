# kolmogorov_decomposition.py

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt
import os

if __name__ == "__main__":
    ## --- Setup ---
    # NOTE: using N = 100
    # N = 1000 would be very slow
    # and memory intensive as the Q matrix would be 500k x 500k
    # the qualitative behavior should be the same
    N = 100
    I0 = 20 # decreased from 50 for N = 1000

    time_max = 10.0
    beta1_scaled = 2.0
    beta2_scaled = 6.0
    mu = 1.0

    beta1 = beta1_scaled / N
    beta2 = beta2_scaled / (N**2)

    ##
    # state space mapping 
    # to map 2D state space to 1D state space
    print("")
    state_to_idx = {}
    idx_to_state = []
    idx = 0
    for k in range(N + 1):
        for k1 in range(k + 1):
            k2 = k - k1
            state = (k1, k2)
            state_to_idx[state] = idx
            idx_to_state.append(state)
            idx += 1
    M = len(idx_to_state)
    print(f"N = {N}, I0 = {I0}, Total states M = {M}")

    ##
    # construct the sparse transition rate matrix Q
    print("constructing Q ...")
    Q = lil_matrix((M, M), dtype=np.float64)

    # fill the i-th column of Q: transitions from state i
    for i in range(M):
        k1, k2 = idx_to_state[i]
        k = k1 + k2

        # rates of events from state i = (k1, k2)
        if k >= N:
            # off boundary
            a_rate, b_rate = 0, 0
        else:
            a_rate = beta1 * k * (N - k)
            b_rate = beta2 * (k * (k - 1) / 2) * (N - k) if k >= 2 else 0.0

        rec1_rate = mu * k1
        rec2_rate = mu * k2

        # total out rate for the diagonal elements
        Q[i, i] = -(a_rate + b_rate + rec1_rate + rec2_rate)

        # off diagonal elements Q[ dest, src ]
        if k < N:
            dest_idx_a = state_to_idx[(k1 + 1, k2)]
            Q[dest_idx_a, i] = a_rate

            dest_idx_b = state_to_idx[(k1, k2 + 1)]
            Q[dest_idx_b, i] = b_rate
        if k1 > 0:
            dest_idx_r1 = state_to_idx[(k1 - 1, k2)]
            Q[dest_idx_r1, i] = rec1_rate 
        if k2 > 0:
            dest_idx_r2 = state_to_idx[(k1, k2 - 1)]
            Q[dest_idx_r2, i] = rec2_rate
    Q_csc = csc_matrix(Q)
    print("constructed Q \n")

    ##
    # initial condition p(0)
    # y(0) = 50 of unspecified origin
    # let's put uniform prob dist over all (k1, k2) where k1 + k2 = 50
    p0 = np.zeros(M)
    initial_states_indices = []
    for k1_init in range(I0 + 1):
        k2_init = I0 - k1_init
        initial_states_indices.append(state_to_idx[(k1_init, k2_init)])

    # let it be uniform prob dist
    if initial_states_indices:
        p0[initial_states_indices] = 1.0 / len(initial_states_indices)

    ## 
    # solve the system E
    print("Solving system E ...")
    def ode_system(t, p):
        return Q_csc.dot(p)

    t_span = [0, time_max]
    t_eval = np.linspace(t_span[0], t_span[1], 201)

    sol = solve_ivp(ode_system, t_span, p0, t_eval=t_eval, method='BDF')
    print("Done. System E solved.\n")

    ## 
    # calculate expected values for plotting
    k1_vals = np.array([s[0] for s in idx_to_state])
    k2_vals = np.array([s[1] for s in idx_to_state])

    # E[X] = sum( x P(X = x) )
    E_k1 = sol.y.T @ k1_vals
    E_k2 = sol.y.T @ k2_vals
    E_k_total = E_k1 + E_k2

    # check that E[k_total] - ( E[k_pw] + E[k_ho] ) is close to 0
    max_diff = np.max(np.abs(E_k_total - (E_k1 + E_k2)))
    print(f"max abs (E[k_total] - (E[k_pw] + E[k_ho]) = {max_diff:.4e} \n")

    ## 
    # plot results
    plt_DPI = 200
    fig_w, fig_h = 8, 5
    plt_legend_fontsize = 16
    plt_labels_fontsize = 18
    plt_tick_fontsize = 14

    plt_linewidth_total = 3
    plt_linewidth_pw = 2.5
    plt_linewidth_ho = 3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=plt_DPI)

    # plotting expectations
    ax.plot(sol.t, E_k_total, color='black', linestyle='-', linewidth=plt_linewidth_total,
            label=r'$E[k_{total}(t)]$', zorder=3)
    ax.plot(sol.t, E_k1, color='red', linestyle='--', linewidth=plt_linewidth_pw,
            label=r'$E[k_{pw}(t)]$', zorder=2)
    ax.plot(sol.t, E_k2, color='blue', linestyle=':', linewidth=plt_linewidth_ho,
            label=r'$E[k_{ho}(t)]$', zorder=2)

    ax.set_xlabel("Time (t)", fontsize=plt_labels_fontsize)
    ax.set_ylabel("Expected Number of Infected", fontsize=plt_labels_fontsize)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.legend(fontsize=plt_legend_fontsize, loc='lower right',
            frameon=True, fancybox=True, shadow=False,
            framealpha=0.9, edgecolor='gray')

    ax.set_ylim(bottom=0, top=max(0.01, np.max(E_k_total) * 1.05))
    ax.set_xlim(left=0, right=time_max)

    ax.tick_params(axis='both', which='major', labelsize=plt_tick_fontsize)
    plt.tight_layout()
    # plt.show()

    # save the figure
    output_folder = "../figures/combined"
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.join(output_folder, "kolmogorov_decomposition.pdf")
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")


'''
$ python3 kolmogorov_decomposition_2.py

N = 100, I0 = 20, Total states M = 5151
constructing Q ...
constructed Q 

Solving system E ...
Done. System E solved.

max abs (E[k_total] - (E[k_pw] + E[k_ho]) = 0.0000e+00 

Plot saved to ../figures/combined/kolmogorov_decomposition.pdf
'''