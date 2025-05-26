# ./scripts/MF_limit_ODE_decomposition.py
# Saves the MF limit ODE decomposition figure to `./figures/combined/MF_decomposition.pdf`

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

def ode_system_D(t, state_vec, beta1_scaled, beta2_scaled, mu_recovery):
    r"""
    System of ODEs (D) for y, p, and h proportions
      * p(t) tracks newly generated PW infections since t = 0
      * h(t) tracks newly generated HO infections since t = 0 
    """
    y, p, h = state_vec
    y = min(max(y, 0.0), 1.0) # y must always be between 0 and 1
    p = max(p, 0.0)
    h = max(h, 0.0)

    y_term = y * (1.0 - y)
    y_sq_term = y**2 * (1.0 - y)

    dydt = beta1_scaled * y_term + (beta2_scaled / 2.0) * y_sq_term - mu_recovery * y
    dpdt = beta1_scaled * y_term - mu_recovery * p
    dhdt = (beta2_scaled / 2.0) * y_sq_term - mu_recovery * h

    return [dydt, dpdt, dhdt]

def solve_system_D(N, I0, beta1, beta2, mu, time_max, num_steps=201):
    r""" 
    Solves the system (D) of ODEs using solve_ivp
      * p(t) tracks newly generated PW infections since t = 0
      * h(t) tracks newly generated HO infections since t = 0
    """
    beta1_s = beta1 * N
    beta2_s = beta2 * (N**2)

    y0_prop = I0 / N
    p0_prop = 0.0
    h0_prop = 0.0 
    initial_state_vec = [y0_prop, p0_prop, h0_prop]

    t_span_val = (0.0, time_max)
    t_eval_points = np.linspace(t_span_val[0], t_span_val[1], num_steps)

    sol = solve_ivp(
        ode_system_D,
        t_span_val,
        initial_state_vec,
        method='LSODA',
        t_eval=t_eval_points,
        args=(beta1_s, beta2_s, mu),
        rtol=1e-6, atol=1e-9
    )
    return sol

if __name__ == "__main__":
    # --- Plot settings ---
    plt_linewidth_total = 2
    plt_linewidth_pw = 2
    plt_linewidth_ho = 4


    # --- Setup ---
    N = 1000
    I0 = 50
    I0_prop = I0 / N

    time_max = 10.0

    beta1_scaled = 2.0
    beta2_scaled = 6.0
    mu = 1.0

    beta1 = beta1_scaled / N
    beta2 = beta2_scaled / (N**2)

    solution_D = solve_system_D(N, I0, beta1, beta2, mu, time_max)

    t = solution_D.t
    y_total = solution_D.y[0]
    p_pairwise = solution_D.y[1]
    h_higher_order = solution_D.y[2]

    # Check that y = y0 + int(dp) + int(dh)
    p_plus_h_sum = p_pairwise + h_higher_order
    difference_y_vs_sum = y_total - (I0_prop + p_pairwise + h_higher_order)
    print("\nMax abs difference between y(t) - y(0) and (p(t) + h(t)):")
    print(f"{np.max(np.abs( (y_total - y_total[0]) - (p_pairwise + h_higher_order) )):.2e}\n")

    # -----------------
    # ---- Plot it ----
    # -----------------
    # TODO: adjusted the size slightly for slides
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    # Plot y(t) = Total Infected Fraction
    ax.plot(t, y_total, color='black', linestyle='-', linewidth=plt_linewidth_total,
            label=r'$y(t)$ (Total)', zorder=3)

    # Plot p(t) = Pairwise Contribution (Red Dashed)
    ax.plot(t, p_pairwise, color='red', linestyle='--', linewidth=plt_linewidth_pw,
            label=r'$p(t)$ (Pairwise)', zorder=2)

    # Plot h(t) = Higher-Order Contribution (Blue Dotted)
    ax.plot(t, h_higher_order, color='blue', linestyle=':', linewidth=plt_linewidth_ho,
            label=r'$h(t)$ (Higher-Order)', zorder=2)

    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Fraction Infected", fontsize=12)

    # TODO: careful with scaled parameters, addewd to the title for consistency
    # TODO: adjust smaller title font for slides
    title_beta1_str = f"{beta1_scaled:.1f}"
    title_beta2_str = f"{beta2_scaled:.1f}"
    ax.set_title(f"Decomposition of Infections (MF Model)\n" + \
                 r"$\beta_1 N = $" + title_beta1_str + r", $\beta_2 N^2 = $" + title_beta2_str + \
                 r", $y(0) = $" + f"{I0_prop:.2f}",
                 fontsize=11)
    
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.5) # TODO: grid to see the crossover at 4.1
    ax.set_ylim(bottom=0, top=max(0.01, np.max(y_total) * 1.05))
    ax.set_xlim(left=0, right=time_max)

    # TODO: ensure ticks are visible
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    # -----------------------
    # --- save the figure ---
    # -----------------------
    # plt.show()
    current_script_dir_fig = os.path.dirname(os.path.abspath(__file__))
    project_root_fig = os.path.dirname(current_script_dir_fig)
    output_figure_dir_fig = os.path.join(project_root_fig, "figures", "combined")
    os.makedirs(output_figure_dir_fig, exist_ok=True)
    output_filename = os.path.join(output_figure_dir_fig, "MF_decomposition.pdf")
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\nFigure saved to: {output_filename}")
