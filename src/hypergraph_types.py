#!/usr/bin/env sage -python
# -*- coding: utf-8 -*-

from Hypergraphs import EmptyHypergraph, CompleteHypergraph
from utils import draw_hypergraph

def example_hypergraph():
    r"""Example of a hypergraph on 4 nodes with 5 edges."""
    N = 4
    g = EmptyHypergraph(N)
    g.name = "Example Hypergraph"
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 2, 3)]
    g.set_edges(edges)
    g.print()

    # set node positions, labels
    positions = {0: (0, 1), 1: (0, 0), 2: (1, 1), 3: (1, 0)}
    labels = {0: "a", 1: "b", 2: "c", 3: "d"}

    # draw it and save the drawing
    file_name = "../figures/hypergraphs/example_hypergraph.svg"
    draw_hypergraph(g, pos=positions, lab=labels, fname=file_name, y_shift=-0.003)
    
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
    g = EmptyHypergraph(N)
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
    file_name = "../figures/hypergraphs/cycle_hypergraph.svg"
    draw_hypergraph(g, fname=file_name)
    g.print()


if __name__ == "__main__":
    # example_hypergraph()
    
    # cycle_hypergraph()

    # complete hypergraph
    file_name = "../figures/hypergraphs/complete_hypergraph.svg"
    draw_hypergraph(CompleteHypergraph(5), fname=file_name)