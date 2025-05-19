import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from higher_order_structures import ErdosRenyiSC
from plot_utils import plot_degree_distribution_ErdosRenyiSC

def run_er_sc_realization_test(N, d1, d2, nruns=100):
    r""" 
    Testing ErdosRenyiSC Generation Algorithm:
      - Check that means of realized p1, p2 equal targets p1, p2
      - Using precise p1_initial, p2_triangles, the properties simplify
    """
    realized_p1 = []
    realized_p2 = []
    
    realized_d1 = []
    realized_d2 = []

    actual_N = []

    target_p1 = d1 / (N - 1.0)
    target_p2 = (2.0 * d2) / ((N - 1.0) * (N - 2.0))

    for i in range(nruns):
        if (i + 1) % (nruns // 10 or 1) == 0:
            print(f"Run {i + 1}/{nruns} ...")
        g = ErdosRenyiSC(N, d1, d2)
        
        realized_p1.append(g.p1_realized) # p1_realized is obs_pw / max_pw
        realized_p2.append(g.p2_realized) # p2_realized is obs_ho / max_ho

        realized_d1.append(g.d1_realized)
        realized_d2.append(g.d2_realized)
        
        actual_N.append(g.N)

    # calculate means and StdDevs of the realized values
    mean_realized_p1 = np.mean(realized_p1)
    std_realized_p1 = np.std(realized_p1)

    mean_realized_p2 = np.mean(realized_p2)
    std_realized_p2 = np.std(realized_p2)

    mean_realized_d1 = np.mean(realized_d1)
    std_realized_d1 = np.std(realized_d1)

    mean_realized_d2 = np.mean(realized_d2)
    std_realized_d2 = np.std(realized_d2)

    print(f"Average N: {np.mean(actual_N):.2f}, min N: {np.min(actual_N)}, max N: {np.max(actual_N)}\n")
    
    print(f"Target p^(1) from d1: {target_p1:.8f}")
    print(f"Target p^(2) from d2: {target_p2:.8f}\n")
    
    print(f"Mean realized p1: {mean_realized_p1:.8f}, SD: {std_realized_p1:.2e}")
    print(f"Mean realized p2: {mean_realized_p2:.8f}, SD: {std_realized_p2:.2e}\n")

    print(f"Target d1: {d1:.2f}")
    print(f"Target d2: {d2:.2f}\n")

    print(f"Mean realized d1: {mean_realized_d1:.2f}, SD: {std_realized_d1:.2e}")
    print(f"Mean realized d2: {mean_realized_d2:.2f}, SD: {std_realized_d2:.2e}\n")

    # check for any significant deviations
    rel_diff_p1 = abs(mean_realized_p1 - target_p1) / target_p1
    rel_diff_p2 = abs(mean_realized_p2 - target_p2) / target_p2
    print(f"Relative diff for p1: {rel_diff_p1:.2%}")
    print(f"Relative diff for p2: {rel_diff_p2:.2%}\n")

if __name__ == "__main__":
    # Test ER SC
    N = 1000
    d1, d2 = (20, 6)
    g = ErdosRenyiSC(N, d1, d2)
    g.print()
    g.summary()
    plot_degree_distribution_ErdosRenyiSC(g)

    # check the 
    run_er_sc_realization_test(1000, 20, 6, nruns=5)
    # TODO: test with a few pairs of (d1, d2)
    # target_degrees_list = [(3, 1), (6, 2), (10, 3), (20, 6)]


"""
# Test 1
        Erdos-Renyi-SC on 1000 nodes with 12039 edges.

--------------------------------------
        Target d1: 20.00, Realized d1: 20.05
        Target d2: 6.00,  Realized d2: 6.04

        Initial p_G used for G(N, p_G): 0.00817743

        Expected p1: 0.02002002
        Expected p2: 0.00001204

        Target p1:  0.02002002
        Target p2:  0.00001204

--------------------------------------
        Realized p1: 0.02007007
        Realized p2: 0.00001212

        Realized number of pw edges:  10025/499500
        Realized number of ho edges:  2014/166167000

Run 1/5 ...
Run 2/5 ...
Run 3/5 ...
Run 4/5 ...
Run 5/5 ...
Average N: 1000.00, min N: 1000, max N: 1000

Target p^(1) from d1: 0.02002002
Target p^(2) from d2: 0.00001204

Mean realized p1: 0.02008208, SD: 3.56e-04
Mean realized p2: 0.00001205, SD: 1.78e-07

Target d1: 20.00
Target d2: 6.00

Mean realized d1: 20.06, SD: 3.56e-01
Mean realized d2: 6.01, SD: 8.88e-02

Relative diff for p1: 0.31%
Relative diff for p2: 0.15%


# Test 2
        Erdos-Renyi-SC on 1000 nodes with 11577 edges.

--------------------------------------
        Target d1: 20.00, Realized d1: 19.32
        Target d2: 6.00, Realized d2: 5.75

        Initial p_G used for G(N, p_G): 0.00817743

        Expected p1: 0.02002002
        Expected p2: 0.00001204

        Target p1:  0.02002002
        Target p2:  0.00001204

--------------------------------------
        Realized p1: 0.01933934
        Realized p2: 0.00001154

        Realized number of pw edges:  9660/499500
        Realized number of ho edges:  1917/166167000

Run 1/5 ...
Run 2/5 ...
Run 3/5 ...
Run 4/5 ...
Run 5/5 ...
Average N: 1000.00, min N: 1000, max N: 1000

Target p^(1) from d1: 0.02002002
Target p^(2) from d2: 0.00001204

Mean realized p1: 0.02018619, SD: 1.37e-04
Mean realized p2: 0.00001206, SD: 1.19e-07

Target d1: 20.00
Target d2: 6.00

Mean realized d1: 20.17, SD: 1.37e-01
Mean realized d2: 6.01, SD: 5.94e-02

Relative diff for p1: 0.83%
Relative diff for p2: 0.22%
"""