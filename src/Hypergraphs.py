""" Hypergraph classes:
  * Empty -> General
  * Complete
  * (Binomial) Random
"""

from itertools import combinations
from math import comb
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
          * g.neighbors(i, 1) is the list of all 2-node neighbors of i
          * g.neighbors(i, 2) is the list of all 3-node neighbors of i
        
        Must run g.set_edges(edges) first!
        """
        return list(self.neighbors_memo[order].get(i, [])) # converts to a list

    def number_of_nodes(self):
        return self.N
    
    def get_node_attributes(self, attr):
        return {i: self.nodes[i][attr] for i in range(self.N)}
    
    def get_edges(self, order=1):
        r"""Returns (order + 1)-node edges in self,
            e.g., order=1 edges are 2-node edges are pairwise edges. """
        return [edge for edge in self.edges if len(edge) == (order + 1)]

    def print(self):
        print(f"\t{self.name} on {self.N} nodes with {len(self.edges)} edges.\n")

    def summary(self):
        print(f"\t Graph name: {self.name}")
        print(f"\t Number of nodes N = {self.N}")
        
        print("\t nodes: ")
        for node in self.nodes.keys():
            print(f"\t\t{node}: {self.nodes[node]}")
        
        print("\t edges: ")
        for edge in self.edges:
            if len(edge) == 2:
                print(f"\t\t{edge}")
        print("\t hyperedges: ")
        for edge in self.edges:
            if len(edge) == 3:
                print(f"\t\t{edge}")
        print("\n")

class CompleteHypergraph(EmptyHypergraph):
    r"""
    Complete hypergraph on `N` nodes, in which each edge is present.
    Overrides most methods of EmptyHypergraph to not store edges explicitly.
    """
    def __init__(self, N):
        super().__init__(N) # initializes N, nodes
        self.name = "Complete hypergraph"
        # self.edges and self.neighbors_memo are empty, generates nbs and edges on demand
    
    def neighbors(self, i, order):
        other_nodes = [j for j in range(self.N) if j != i]
        if order == 1:
            return other_nodes
        elif order == 2:
            return list(combinations(other_nodes, 2))
    
    def get_edges(self, order=1):
        nodes = list(range(self.N))
        if order == 1:
            return list(combinations(nodes, 2))
        elif order == 2:
            return list(combinations(nodes, 3))

    def print(self):
        print(f"\t{self.name} on {self.N} nodes with {comb(self.N, 2) + comb(self.N, 3)} edges.\n")

    def summary(self):
        nodes = list(range(self.N))
        pw_edges = list(combinations(nodes, 2))
        ho_edges = list(combinations(nodes, 3))
        print(f"\t Graph name: {self.name}")
        print(f"\t Number of nodes N = {self.N}")
        
        print("\t nodes: ")
        for node in self.nodes.keys():
            print(f"\t\t{node}: {self.nodes[node]}")
        
        print("\t 2-node edges: ")
        for edge in pw_edges:
            print(f"\t\t{edge}")
        print("\t 3-node edges: ")
        for edge in ho_edges:
            print(f"\t\t{edge}")
        print("\n")

class RandomHypergraph(EmptyHypergraph):
    r"""
    (Binomial) Random hypergraph on `N` nodes, in which each 2-node edge and 
    3-node edge is present independently with probability p1 and p2 respectively.
    """
    def __init__(self, N, p1, p2):
        super().__init__(N)
        self.p1 = p1 # probability of a 2-node edge O(N^2)
        self.p2 = p2 # probability of a 3-node edge should be order N smaller since it scales as O(N^3)
        self.name = "(Binomial) random hypergraph"
        self.set_edges(self.generate_random_hypergraph())
    
    def generate_random_hypergraph(self):
        nodes = list(range(self.N))
        pw_edges = list(combinations(nodes, 2))
        ho_edges = list(combinations(nodes, 3))
        edges = []
        
        # append 2-node (pairwise) edges with probability p1
        for edge in pw_edges:
            if random.random() < self.p1:
                edges.append(edge)
        
        # append 3-node (higher-order) edges with probability p2
        for hyperedge in ho_edges:
            if random.random() < self.p2:
                edges.append(hyperedge)
        
        return edges

