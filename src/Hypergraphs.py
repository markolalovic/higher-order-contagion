#!/usr/bin/env sage -python
# -*- coding: utf-8 -*-

from itertools import combinations
import random

class EmptyHypergraph:
    def __init__(self, N):
        r"""
        Generic superclass for hypergraphs, contains no edges:
          * Initializes an empty hypergraph
          * Implements all the required methods for Gillespie simulation
          * Other hypergraph classes are just hypergraph generators
        """
        self.name = "Empty hypergraph"
        self.N = N
        self.nodes = {i: {'state': 0, 'rate': 0.0} for i in range(N)}
        self.edges = []
        self.neighbors_memo = {1: {}, 2: {}} # precomputed neighbors
    
    def set_edges(self, edges):
        r"""
        Precompute and store all pairwise (edge) and 
        higher-order (hyperedge) neighbors. 
        """
        self.edges = edges
        for i in range(self.N):
            self.neighbors_memo[1][i] = set()
            self.neighbors_memo[2][i] = set()
        
        for edge in self.edges:
            if len(edge) == 2:
                u, v = edge
                self.neighbors_memo[1][u].add(v)
                self.neighbors_memo[1][v].add(u)
            
            elif len(edge) == 3:
                i, j, k = edge
                self.neighbors_memo[2][i].add((j, k))
                self.neighbors_memo[2][j].add((i, k))
                self.neighbors_memo[2][k].add((i, j))
    
    def neighbors(self, i, order):
        r"""
        Returns neighbors of node `i` for a given order:
          * g.neighbors(i, 1) is the list of all edge neighbors of i
          * g.neighbors(i, 2) is the list of all hyperedge neighbors of i
            
         Must run g.set_edges(edges) first!
        """
        return list(self.neighbors_memo[order].get(i, [])) # converts to a list

    def number_of_nodes(self):
        return self.N
    
    def get_node_attributes(self, attr):
        return {i: self.nodes[i][attr] for i in range(self.N)}
    
    def get_simplices(self, order=1):
        """ Return order-simplices in self. """
        return [edge for edge in self.edges if len(edge) == (order + 1)]

    def print(self):
        print(f"\t{self.name} on {self.N} nodes with {len(self.edges)} edges.\n")

    def summary(self):
        print(f"\tGraph name: {self.name}")
        print(f"\tNumber of nodes N = {self.N}")
        
        print("\tnodes: ")
        for node in self.nodes.keys():
            print(f"\t\t{node}: {self.nodes[node]}")

        print("\tedges: ")
        for edge in self.edges:
            if len(edge) == 2:
                print(f"\t\t{edge}")
        print("\thyperedges: ")
        for edge in self.edges:
            if len(edge) == 3:
                print(f"\t\t{edge}")
        print("\n")

class CompleteHypergraph(EmptyHypergraph):
    r"""
    Complete hypergraph on `N` nodes, in which each edge and hyperedge is present. 
    """
    def __init__(self, N):
        super().__init__(N) # initializes N, nodes and empty set of edges
        self.name = "Complete hypergraph"
        self.set_edges(self.generate_complete_hypergraph()) # precompute the neighbors

    def generate_complete_hypergraph(self):
        nodes = list(range(self.N))
        edges = list(combinations(nodes, 2)) # all possible edges
        edges.extend(combinations(nodes, 3)) # all possible hyperedges
        return edges

class RandomHypergraph(EmptyHypergraph):
    r"""
    (Binomial) Random hypergraph on `N` nodes, in which each edge and 
    hyperedge is present independently with probability `p`.
    """
    def __init__(self, N, p1, p2):
        super().__init__(N)
        self.p2 = p1 # probability of an edge O(N^2)
        self.p1 = p2 # prob of a hyperedge should be order N smaller since it scales as O(N^3)
        self.name = "(Binomial) random hypergraph"
        self.set_edges(self.generate_random_hypergraph())
    
    def generate_random_hypergraph(self):
        nodes = list(range(self.N))
        edges = []
        
        # append edges (pairwise edges) with probability p
        for edge in combinations(nodes, 2):
            if random.random() < self.p1:
                edges.append(edge)
        
        # append hyperedges (triangles) with probability p
        for hyperedge in combinations(nodes, 3):
            if random.random() < self.p2:
                edges.append(hyperedge)
        
        return edges

def example_hypergraph():
    r"""Example of a hypergraph:
      * edges forming a cycle on 5 nodes
      * hyperedges forming 2 overlaping triangles on 4 nodes
    """
    N = 5
    g = EmptyHypergraph(N)
    g.name = "Example Hypergraph"
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), 
               (0, 1, 2), (1, 2, 3)]
    g.set_edges(edges)
    g.nodes[0]['state'] = 1
    return g

def example45():
    r"""Example of a hypergraph (Subsection 4.5) on 4 nodes with 5 edges."""
    N = 4
    g = EmptyHypergraph(N)
    g.name = "Example Hypergraph 4.5"
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 2, 3)]
    g.set_edges(edges)
    return g

"""
Old plot:

    def plot(self, file_name=''):
        sage_g = Graph()
        
        # only draws pair-wise edges for now
        sage_g.add_edges(self.get_simplices(order = 1))

        # no position specified for plot
        p = sage_g.graphplot()

        if file_name:
            p.plot().save(file_name)
        else:
            file_name = "../figures/test.png"
            p.plot().save(file_name)
            p.plot().show() # interactive only 
            # gives: `Graphics object consisting of 11 graphics primitives`
            # TODO: how to force display?
            img = mpimg.imread(file_name)
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
"""