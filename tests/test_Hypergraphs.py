#!/usr/bin/env sage -python
# -*- coding: utf-8 -*-

from Hypergraphs import example_hypergraph

if __name__ == "__main__":
    file_name = "../figures/example_hypergraph.svg"
    g = example_hypergraph()
    g.print()
    g.summary()

    # check the neighbors of node 1:
    print(f"\tNeighbors of node = 1 of order = 1: {g.neighbors(i = 1, order = 1)}\n")
    print(f"\tNeighbors of node = 1 of order = 2: {g.neighbors(i = 1, order = 2)}\n")

    g.plot(file_name, y_shift=-0.008)
    