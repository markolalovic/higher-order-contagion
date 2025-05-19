#!/usr/bin/env sage -python
# -*- coding: utf-8 -*-

from higher_order_structures import HigherOrderStructure, Complete, RandomHypergraph
from higher_order_structures import ErdosRenyiSC

def example():
    r"""Example of a higher-order structure."""
    N = 4
    g = HigherOrderStructure(N)
    g.name = "Higher Order Example"
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 1), (1, 2, 3)]
    g.set_edges(edges)
    g.print()

    # set node positions, labels
    positions = {0: (0, 1), 1: (0, 0), 2: (1, 1), 3: (1, 0)}
    # labels = {0: "a", 1: "b", 2: "c", 3: "d"}

    # draw it and save the drawing
    file_name = "../figures/higher_order_structures/ho_example.svg"
    draw_hypergraph(g, pos=positions, fname=file_name)
    
    # set states of some nodes and print the summary:
    g.nodes[0]['state'] = 1
    g.nodes[0]['state'] = 2
    g.summary()

    # list the neighbors of a node i:
    i = 1
    pw_nbs = g.neighbors(i, order = 1)
    ho_nbs = g.neighbors(i, order = 2)
    print(f"\t Pairwise neighbors of node {i}: {pw_nbs}\n")
    print(f"\t Higher-order neighbors of node {i}: {ho_nbs}\n")

def cycle_hypergraph(N = 10):
    N = 10
    g = HigherOrderStructure(N)
    g.name = "Cycle Hypergraph"
    edges = []
    for i in range(N):
        # 2-node edges form a cycle
        j = (i + 1) % N
        edges.append((i, j))
        
        # 3-node edges form a cycle
        k = (i + 2) % N
        edges.append((i, j, k))

    g.set_edges(edges)
    file_name = "../figures/higher_order_structures/ho_cycle_hypergraph.svg"
    draw_hypergraph(g, fname=file_name)
    g.print()


if __name__ == "__main__":
    # example()
    # cycle_hypergraph()
    
    # # random hypergraph    
    # file_name = "../figures/higher_order_structures/ho_complete.svg"
    # draw_hypergraph(Complete(N = 10), fname=file_name)    

    # # random hypergraph    
    # N = 10
    # g = RandomHypergraph(N = N, p1 = 0.5, p2 = 0.5 / N)
    # file_name = "../figures/higher_order_structures/ho_random_hypergraph.svg"
    # draw_hypergraph(g, fname=file_name)

    # --- Erdos-Renyi-SC --- 
    N = 1000
    d1, d2 = (20, 6)
    g = ErdosRenyiSC(N, d1, d2)
    g.print()
    g.summary()
    # Erdos-Renyi-SC on 1000 nodes with 12123 edges.

    # Target d1: 20.00, Realized d1: 20.21
    # Target d2: 6.00, Realized d2: 6.06

    # Initial p1 used for G(N, p1): 0.0081
    # Expected p1 used for pw edges: 0.01994882
    # Expected p2 used for ho edges: 0.00001204

    # Realized p1_overall: 0.0202
    # Realized p2: 0.00001215

    # Realized pw edges:  10104/499500
    # Realized ho edges:  2019/166167000

