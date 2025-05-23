"""
Testing hypergraph generators:
  * Scale-free hypergraph using configuration model  
  * Erdos-Renyi like random hypergraph
  * Regular hypergraph TODO: ensure unique vertices and no repeated triangles

Commented out:
  * NegBinom hypergraph
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
# from scipy.stats import nbinom
from higher_order_structures import HigherOrderStructure
import os

from scipy.stats import zipf # for power-law Zeta distribution
# from itertools import combinations
from scipy.special import comb

def scale_free_hypergraph(N, gamma_pw, k_min_pw, gamma_ho, k_min_ho, attempts=1000):
    r"""
    Generates a hypergraph with degree sequence from a power-law distribution:

      P(k) \propto k^-gamma

    Using:
      * scipy.stats.zipf which gives P(k) proportional to k**(-gamma) for k = 1, 2, 3 ...
      * and clip degrees between k_min and k_max
    
    Example setup:
        * N = 2000        # number of nodes

        * gamma_pw = 2.5  # exponent for scale-free networks
        * k_min_pw = 2    # minimum pairwise degree

        * gamma_ho = 2.5  # can be different for HO interactions
        * k_min_ho = 1    # minimum number of triangles a node is in
    """
    all_edges = []
    k_max_pw = N - 1 # maximum PW degree
    k_max_ho = comb(N - 1, 2, exact=True) # maximum HO degree

    # generate PW edges
    for _ in range(attempts):        
        degrees_pw = zipf.rvs(gamma_pw, size=N)
        degrees_pw = np.clip(degrees_pw, k_min_pw, k_max_pw)
        if np.sum(degrees_pw) % 2 == 0:
            break
    edges_pw_list = configuration_model_edges(degrees_pw)
    all_edges.extend(edges_pw_list)

    # generate HO edges
    for _ in range(attempts):
        # degrees_ho = target number of triangles for each node
        degrees_ho = zipf.rvs(gamma_ho, size=N)
        degrees_ho = np.clip(degrees_ho, k_min_ho, k_max_ho)
        # TODO: convert this to stubs for the configuration model
        # NOTE: each node contributes 2 stubs for every triangle it is in
        # degrees_ho = degrees_ho * 2
        # NOTE: each triangle consumes 3 * 2 = 6 stubs in total
        if np.sum(degrees_ho) % 3 == 0:
            break
    triangles_ho_list = configuration_model_triangles(degrees_ho)
    all_edges.extend(triangles_ho_list)

    return all_edges, degrees_pw, degrees_ho

def test_scale_free_hypergraph(
        N=1000,  # number of nodes
        gamma_pw = 2.5,  # exponent for scale-free networks
        k_min_pw = 2,    # minimum pairwise degree
        gamma_ho = 2.5,  # can be different for HO interactions
        k_min_ho = 2    # minimum number of triangles a node is in
        ):
    all_edges, degrees_pw, degrees_ho = scale_free_hypergraph(
            N, gamma_pw, k_min_pw, gamma_ho, k_min_ho)

    g = HigherOrderStructure(N)
    g.name = "ScaleFree"
    g.set_edges(all_edges)
    g.print()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --------------------------------------------
    # PW degrees
    # --------------------------------------------
    # generated sequence of degrees
    counts_pw, bins_pw = np.histogram(degrees_pw, bins=np.logspace(np.log10(max(1, k_min_pw)), np.log10(np.max(degrees_pw) + 1), 30), density=True)
    bin_centers_pw = (bins_pw[:-1] + bins_pw[1:]) / 2
    axes[0].loglog(bin_centers_pw, counts_pw, 'o', color='blue', 
                   alpha=0.7, label='Sequence PW Degrees')
    # simulated degrees
    sim_degrees_pw = np.zeros(N, dtype=int)
    for node_idx in range(N):
        sim_degrees_pw[node_idx] = len(g.neighbors(node_idx, 1)) # NOTE: order = 1
    
    counts_pw, bins_pw = np.histogram(sim_degrees_pw, bins=np.logspace(np.log10(max(1, k_min_pw)), np.log10(np.max(sim_degrees_pw) + 1), 30), density=True)
    bin_centers_pw = (bins_pw[:-1] + bins_pw[1:]) / 2
    axes[0].loglog(bin_centers_pw, counts_pw, 'x', color='red', 
                   alpha=1, label='Simulated PW Degrees')
    # theoretical line
    k_plot = np.logspace(np.log10(k_min_pw), np.log10(np.max(degrees_pw)), 50)
    axes[0].loglog(k_plot, (k_plot**(-gamma_pw)) / (np.sum(k_plot**(-gamma_pw))),
                   'k', alpha=0.5, label=f'P(k) ~ k^{-gamma_pw}')
    
    axes[0].set_title(f'PW Degrees, Avg = {np.mean(degrees_pw):.2f}, Max = {np.max(degrees_pw):.0f}')
    axes[0].set_xlabel('Pairwise Degree k_pw')
    axes[0].set_ylabel('P(k_pw)')
    axes[0].legend()
    axes[0].grid(True, which="both", ls=":", alpha=0.7) # grid for loglog

    # --------------------------------------------
    # HO degrees
    # --------------------------------------------
    # generated sequence of degrees
    counts_ho, bins_ho = np.histogram(degrees_ho, bins=np.logspace(np.log10(max(1, k_min_ho)), np.log10(np.max(degrees_ho) + 1), 30), density=True)
    bin_centers_ho = (bins_ho[:-1] + bins_ho[1:]) / 2
    axes[1].loglog(bin_centers_ho, counts_ho, 'o', color='blue', 
                   alpha=0.7, label='Sequence HO Degrees')
    
    # simulated degrees
    # HO degrees: number of 3-node edges (triangles) a node is part of
    sim_degrees_ho = np.zeros(N, dtype=int)
    for node_idx in range(N):
        sim_degrees_ho[node_idx] = len(g.neighbors(node_idx, 2))
    
    counts_ho, bins_ho = np.histogram(sim_degrees_ho, bins=np.logspace(np.log10(max(1, k_min_ho)), np.log10(np.max(sim_degrees_ho) + 1), 30), density=True)
    bin_centers_ho = (bins_ho[:-1] + bins_ho[1:]) / 2
    axes[1].loglog(bin_centers_ho, counts_ho, 'x', color='red', 
                   alpha=1, label='Simulated HO Degrees')
    # theoretical line
    k_plot = np.logspace(np.log10(k_min_ho), np.log10(np.max(degrees_ho)), 50)
    axes[1].loglog(k_plot, (k_plot**(-gamma_ho)) / (np.sum(k_plot**(-gamma_ho))), 
                   'k', alpha=0.5, label=f'P(k) ~ k^{-gamma_ho}')
    
    axes[1].set_title(f'HO Degrees, Avg = {np.mean(degrees_ho):.2f}, Max = {np.max(degrees_ho):.0f}')
    axes[1].set_xlabel('HO Degree k_ho')
    axes[1].set_ylabel('P(k_ho)')
    axes[1].legend()
    axes[1].grid(True, which="both", ls=":", alpha=0.7) # grid for loglog

    title = f"Degree Distributions {g.name} with N = {N}, "
    title += f" gamma_pw = {gamma_pw}, k_min_pw = {k_min_pw}, "
    title += f" gamma_ho = {gamma_ho}, k_min_ho = {k_min_ho} "
    fig.suptitle(title, fontsize=16)
    
    name = f"scale_free_hypergraph_degree_distributions"
    save_dir = "../figures/higher_order_structures/"
    save_path = os.path.join(save_dir, f"{name}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    plt.show()


# from Federico:
def ER_like_random_hypergraph(N,p1,p2):
    """Our model"""
    #I first generate a standard ER graph with edges connected with probability p1
    G = nx.fast_gnp_random_graph(N, p1, seed=None)
    if not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        giant = G.subgraph(giant).copy()
        print('not connected, but GC has order ', giant.order(), 'and size', giant.size())
        G = giant
        N=G.order()
        mapping = {nn:0 for nn in list(G.nodes())}
        for iidx,ii in enumerate(list(G.nodes())):
            mapping[ii] = iidx
        G = nx.relabel_nodes(G,mapping)

    num_triangles = int(np.random.normal((N*(N-1)*(N-2))/6*p2,np.sqrt((N*(N-1)*(N-2)/6)*p2*(1-p2)),size=None))# np.random.binomial(N*(N-1)*(N-2)/6, p2, size=None)
    # NOTE: This stops working with N >~ 2000, as N*(N-1)*(N-2)/6 is interpreted as int32
    #       use instead np.random.normal(N*(N-1)*(N-2)/6*p2,np.sqrt(N*(N-1)*(N-2)/6*p2(1-p2)),size=None)
    #print(num_triangles)
    Triangles = np.zeros((num_triangles,3))
    for i in range(num_triangles):
        
        # we select randomly 3 distinct nodes of the network (we order the nodes in increasing index)
        tri = sorted(random.sample(range(N), 3))
        
        # we verify that the triplet of nodes was not already generated
        while np.sum(np.all(Triangles==tri,axis=1))!=0:
            tri = sorted(random.sample(range(N), 3))
        Triangles[i,:3]=tri
    
    # 2-body interactions
    m = G.number_of_edges() # number of edges in the network
    edges = np.zeros((m,2))
    edges[:,:2] = np.array([e for e in G.edges])
    edges = np.array(edges, dtype=int)
    
    # 3-body interactions
    Triangles = np.sort(Triangles)
    triangles = np.ones((len(Triangles),3), dtype=int)
        
    triangles[:,:3]=Triangles
    
    return G, edges, triangles

# from Federico:
def p1_p2_ER_like_uncorrelated_hypergraph(k1,k2,N):
    p2 = (2*k2)/((N-1)*(N-2))
    p1 = (k1)/((N-1))
    if (p1>=0) and (p2>=0):
        return p1, p2
    else:
        raise ValueError('Negative probability!')

def test_ER_hypergraph(test_p1_p2=False, test_k1_k2=False):
    # setup
    N = 1000

    k1_k2_list = [(3, 1), (6, 2), (10, 3), (20, 6)]
    (k1, k2) = k1_k2_list[2]
    print(f"Using (k1, k2) = {(k1, k2)}")

    p1, p2 = p1_p2_ER_like_uncorrelated_hypergraph(k1,k2,N)
    _, edges, triangles = ER_like_random_hypergraph(N,p1,p2)

    g_edges = []
    all_edges = edges.tolist() + triangles.tolist()
    for edge in all_edges:
        g_edges.append(tuple(edge))
    print(f"g_edges: {g_edges[:5]}, ..., {g_edges[-5:]}")
    # Using (k1, k2) = (10, 3)
    # g_edges: [(0, 158), (0, 301), (0, 327), (0, 406), (0, 436)], ..., [(320, 738, 903), (165, 390, 550), (503, 669, 968), (300, 493, 687), (406, 616, 672)]

    print(f"p1 = {p1:.4f}")
    print(f"p2 = {p2:.8f}")    
    # p1 = 0.0100
    # p2 = 0.00000602

    # $p_1 = 0.01 > ln(N)/N \approx 0.0069 \text{ where } E[\text{isolated nodes}] \approx 1$ 
    # $G$ is connected almost surely..

    if test_p1_p2:
        max_pw_edges = N * (N - 1) / 2
        max_ho_edges = N * (N - 1) * (N - 2) / 6

        p1_est = []
        p2_est = []
        nsims = 1000
        for _ in range(nsims):
            p1, p2 = p1_p2_ER_like_uncorrelated_hypergraph(k1,k2,N)
            _, edges, triangles = ER_like_random_hypergraph(N,p1,p2)
            p1_est.append(len(edges) / max_pw_edges)
            p2_est.append(len(triangles) / max_ho_edges)

        print(f"p1_est = {np.mean(p1_est):.4f}")
        print(f"p2_est = {np.mean(p2_est):.8f}")
        # p1 = 0.0100
        # p2 = 0.00000602        
        # p1_est = 0.0100
        # p2_est = 0.00000600
    
    if test_k1_k2:
        k1_est = []
        k2_est = []
        nsims = 1000
        for _ in range(nsims):
            p1, p2 = p1_p2_ER_like_uncorrelated_hypergraph(k1,k2,N)
            _, edges, triangles = ER_like_random_hypergraph(N,p1,p2)

            g_type = "random_ER"
            g = EmptyHypergraph(N)
            g.name = g_type
            g.set_edges(g_edges)

            k1_sim = np.mean([len(g.neighbors(i, 1)) for i in list(g.nodes.keys())])
            k2_sim = np.mean([len(g.neighbors(i, 2)) for i in list(g.nodes.keys())])

            k1_est.append(k1_sim)
            k2_est.append(k2_sim)
        
        print(f"k1_est = {np.mean(k1_est):.4f}")
        print(f"k2_est = {np.mean(k2_est):.4f}")
        # Using (k1, k2) = (10, 3)
        # k1_est = 10.0140
        # k2_est = 2.8680

# from Federico:
def configuration_model_edges(degrees):
    # Create stubs
    stubs = np.repeat(np.arange(len(degrees)), degrees)
    np.random.shuffle(stubs)
    
    # Pair stubs to create edges
    edges = []
    i = 0
    while i < len(stubs) - 1:
        u, v = stubs[i], stubs[i+1]
        
        # Avoid self-loops and multiple edges
        if u != v and [u, v] not in edges and [v, u] not in edges:
            edges.append([min(u, v), max(u, v)])
            i += 2
        else:
            # If invalid, continue searching
            i += 1
    
    return np.array(edges)

# from Federico:
def configuration_model_triangles(degrees):
    # Create stubs
    stubs = np.repeat(np.arange(len(degrees)), degrees) # "weighted list"
    np.random.shuffle(stubs)
    
    # Create triangles
    triangles = []
    i = 0
    while i < len(stubs) - 2:
        u, v, w = stubs[i], stubs[i+1], stubs[i+2]
        
        # Ensure unique vertices and no repeated triangles
        if len({u, v, w}) == 3 and u != v and u!= w and v!=w and\
           tuple(sorted([u,v,w])) not in [tuple(sorted(t)) for t in triangles]:
            
            triangles.append(np.sort([u, v, w]))
            i += 3
        else:
            # If invalid, continue searching
            i += 1
    
    return np.array(triangles)





# from Federico:
def random_regular_hypergraph(k_1,k_2, N, seed=None):
    """
    N*k_2/3 must be an int!
    
    Returns a k-regular random hypergraph, here each node has exactly
    - 1-degree = k_1
    - 2-degree = k_2
    The graph will be composed of k*N nodes.
    
    Parameters
    ----------
    k : int
      The degree of each node.
    N : int
      The number of nodes for the starting k-regular random graph. 
      The value of $N \times k$ must be even.
    seed : int, random_state, or None (default)
        Indicator of random number generation state.
        
    """
    if k_2*N % 3 != 0:
        raise ValueError('k_2 * N must be a multiple of 3!')

    else:
        
        if N*(N-1)*(N-2)/6 < k_2*N/3:
            raise ValueError('You cannot obtain a regular set of 2-hyperedges with this set of parameters!')

            #raise ValueError('Number of possible triangles is larger then the number of possible triples!')
        elif N*(N-1)*(N-2)/6 >= k_2*N/3:

            G = nx.random_regular_graph(k_1,N,seed=seed)

            triangles_list = []
            list_nodes = list(range(N))
            dict_k2 = {j:0 for j in range(N)}
            counter = 0
            while sum(list(dict_k2.values())) != k_2*N:    
                counter+=1
                #n1 = np.random.choice(dict_k2[j])
                a = np.array(list(dict_k2.values())) 
                b = np.array(list(dict_k2.keys()))
                n1,n2,n3 = sorted(np.random.choice(b[a!=k_2],3,replace=False))
               # if dict_k2[n1] != k_2 and dict_k2[n2] != k_2 and dict_k2[n3] != k_2:
                if (n1,n2,n3) not in triangles_list:
                    triangles_list.append((n1,n2,n3))
                    dict_k2[n1] += 1 
                    dict_k2[n2] +=1 
                    dict_k2[n3] += 1
                else:
                    pass
                if counter == k_2*N*10:
                    raise ValueError("Too many attempts, try again")

    # return G, np.array(triangles_list)
    # extract the edges 
    edges = np.array(G.edges())
    triangles = np.array(triangles_list)
    return edges, triangles

"""
### generate edges_list 

N=2000

var = 200#640#602
r = (k1**2)/(var - k1)
p = k1/var

for aaa in range(1000):
    degrees = nbinom.rvs(r, p, size=N)+1
    if sum(degrees)%2 == 0:
        break

print('Edges set generated')

edges_list = configuration_model_edges(degrees)


### generate triangles_list
var = 600#650#620
r = (k2**2)/(var - k1)
p = k2/var

for aaa in range(1000):
    degrees = nbinom.rvs(r, p, size=N)+1
    if sum(degrees)%3 == 0:
        break
triangles_list_variance_600 = configuration_model_triangles(degrees)
"""


'''
# Not flexible enough
def neg_binom_hypergraph(N, k_pw_avg, var_pw, k_ho_avg, var_ho, attempts=1000):
    r"""
    Generates a hypergraph with pairwise and higher-order degrees drawn
    from Negative Binomial distributions with parameters, example setup:

      * N = 2000      # number of nodes

      * k_pw_avg = 20 # average degree PW
      * var_pw = 200  # variance PW

      * k_ho_avg = 6  # average degree HO
      * var_ho = 100  # variance HO
    """
    all_edges = []

    # generate PW edges
    r_pw = (k_pw_avg**2) / (var_pw - k_pw_avg)
    p_pw = k_pw_avg / var_pw
    for _ in range(attempts):
        degrees_pw = nbinom.rvs(r_pw, p_pw, size=N) + 1
        if np.sum(degrees_pw) % 2 == 0:
            break
    edges_pw_list = configuration_model_edges(degrees_pw)
    all_edges.extend(edges_pw_list)

    # generate HO edges
    r_ho = (k_ho_avg**2) / (var_ho - k_ho_avg)
    p_ho = k_ho_avg / var_ho
    for _ in range(attempts):
        degrees_ho = nbinom.rvs(r_ho, p_ho, size=N) + 1
        # degrees_ho_stubs = node_triangle_counts * 2   # if these are stubs degreees
        if np.sum(degrees_ho) % 3 == 0:
            break
    edges_ho_list = configuration_model_triangles(degrees_ho)
    all_edges.extend(edges_ho_list)

    return all_edges, (r_pw, p_pw), (r_ho, p_ho)

def test_neg_binom_hypergraph():
    # setup
    N = 2000      # number of nodes

    k_pw_avg = 20 # average degree PW
    var_pw = 200  # variance PW

    k_ho_avg = 6  # average degree HO
    var_ho = 100  # variance HO

    all_edges, (r_pw, p_pw), (r_ho, p_ho) = neg_binom_hypergraph(N, k_pw_avg, var_pw, k_ho_avg, var_ho)

    g = EmptyHypergraph(N)
    g.name = "NegBinom"
    g.set_edges(all_edges)
    g.print()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PW degrees
    sim_degrees_pw = np.zeros(N, dtype=int)
    for node_idx in range(N):
        sim_degrees_pw[node_idx] = len(g.neighbors(node_idx, 1))

    # HO degrees: number of 3-node edges (triangles) a node is part of
    sim_degrees_ho = np.zeros(N, dtype=int)
    for node_idx in range(N):
        sim_degrees_ho[node_idx] = len(g.neighbors(node_idx, 2))

    max_degree_pw = np.max(sim_degrees_pw) if len(sim_degrees_pw) > 0 else 0
    var_degree_pw = np.var(sim_degrees_pw) if len(sim_degrees_pw) > 0 else 0
    avg_degree_pw = np.mean(sim_degrees_pw) if len(sim_degrees_pw) > 0 else 0
    bins_pw = np.arange(-0.5, max_degree_pw + 1.5, 1)

    # PW degrees
    axes[0].hist(sim_degrees_pw, bins=bins_pw, density=True,
                alpha=0.7, label='Simulated PW Degrees', color='blue', ec='black')

    k_values_pw = np.arange(0, max_degree_pw + 1)
    pmf_pw = nbinom.pmf(k_values_pw - 1, r_pw, p_pw) # -1 because +1 was added in generation
    axes[0].plot(k_values_pw, pmf_pw, ms=4, color='black',
                label=f'NegBinom PMF (r={r_pw:.2f}, p={p_pw:.2f}')

    axes[0].axvline(avg_degree_pw, color="red", linestyle='--', lw=2,
                    label=f"Avg. PW Degree = {avg_degree_pw:.2f}")

    axes[0].set_xlabel('PW Degree')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.7)
    axes[0].set_xlim(left=-1)

    # HO degrees
    max_degree_ho = np.max(sim_degrees_ho) if len(sim_degrees_ho) > 0 else 0
    var_degree_ho = np.var(sim_degrees_ho) if len(sim_degrees_ho) > 0 else 0
    avg_degree_ho = np.mean(sim_degrees_ho) if len(sim_degrees_ho) > 0 else 0
    bins_ho = np.arange(-0.5, max_degree_ho + 1.5, 1)

    axes[1].hist(sim_degrees_ho, bins=bins_ho, density=True,
                alpha=0.7, label='Simulated HO Degrees', color='blue', ec='black')

    k_values_ho = np.arange(0, max_degree_ho + 1)
    pmf_ho = nbinom.pmf(k_values_ho - 1, r_ho, p_ho) # -1 because +1 was added in generation
    axes[1].plot(k_values_ho, pmf_ho, ms=4, color='black',
                label=f'NegBinom PMF (r={r_ho:.2f}, p={p_ho:.2f})')

    axes[1].axvline(avg_degree_ho, color="red", linestyle='--', lw=2,
                    label=f"Avg. HO Degree = {avg_degree_ho:.2f}")

    axes[1].set_xlabel('HO Degree')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.7)
    axes[1].set_xlim(left=-1)

    # add also variances as text
    print(f"var_degree_pw = {np.mean(var_degree_pw):.4f}")
    print(f"var_degree_ho = {np.mean(var_degree_ho):.4f}")

    text_x, text_y = 0.62, 0.81
    text_var_degree_pw = f"Variance PW Degree = {var_degree_pw:.2f}"
    axes[0].text(text_x, text_y, text_var_degree_pw,
                transform=axes[0].transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(fc='white', alpha=1))

    text_var_degree_ho = f"Variance HO Degree = {var_degree_ho:.2f}"
    axes[1].text(text_x, text_y, text_var_degree_ho,
                transform=axes[1].transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(fc='white', alpha=1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig_title = f"Degree distributions for H = {g.name}: N = {N} with:" 
    fig_title += f" k_pw = {k_pw_avg}, var_pw = {var_pw}, k_ho = {k_ho_avg}, var_ho = {var_ho}"
    fig_title += " drawn from Negative binomial distribution shifted by +1"
    fig.suptitle(fig_title)

    name = f"neg_binom_hypergraph_degree_distributions"
    save_dir = "../figures/hypergraphs/"
    save_path = os.path.join(save_dir, f"{name}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    plt.show()

'''
