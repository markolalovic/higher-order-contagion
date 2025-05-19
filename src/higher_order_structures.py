"""./src/higher_order_structures.py
Module for defining and generating higher-oder network structures, 
including general hypergraphs and classes of simplicial complexes.

HigherOrderStructure: implements all common methods and attributes needed
for up to 2-simplices or 3-node hyperedges, but can be extendable to higher orders.
"""

from itertools import combinations
import numpy as np
from math import comb
import random
from higher_order_generators import generate_er_sc_components, get_p1_p2_for_target_degrees

class HigherOrderStructure:
    def __init__(self, N):
        r"""
        Generic superclass, contains no edges:
          - Initializes an empty higher order structure
          - Implements all the required methods for Gillespie simulation
          - Other classes are just generators
        """
        self.name = "Empty"
        self.N = N
        self.nodes = {i: {'state': 0, 'rate': 0.0} for i in range(N)}
        self.edges = []
        self.neighbors_memo = {1: {}, 2: {}} # precomputed neighbors
    
    def set_edges(self, edges):
        r"""
        Helper to precompute and store all pairwise (edge) and 
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
                v1, v2, v3 = edge
                self.neighbors_memo[2][v1].add( tuple( sorted((v2, v3)) ) )
                self.neighbors_memo[2][v2].add( tuple( sorted((v1, v3)) ) )
                self.neighbors_memo[2][v3].add( tuple( sorted((v1, v2)) ) )
    
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
            e.g., order=1 edges are 2-node or pairwise edges. """
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

class Complete(HigherOrderStructure):
    r"""
    Complete simplicial complex on `N` nodes, in which every edge is present.
    Overrides most methods of HigherOrderStructure to not store edges explicitly.
    """
    def __init__(self, N):
        super().__init__(N) # initializes N, nodes
        self.name = "Complete"
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

class RandomHypergraph(HigherOrderStructure):
    r"""
    (Binomial) Random hypergraph on `N` nodes, in which every 2-node edge and 
    3-node edge is present independently with probability p1 and p2 respectively.
    """
    def __init__(self, N, p1, p2):
        super().__init__(N)
        self.p1 = p1 # probability of a 2-node edge O(N^2)
        self.p2 = p2 # probability of a 3-node edge should be order N smaller since it scales as O(N^3)
        self.name = "(Binomial) random hypergraph"
        self.set_edges(self.generate_random_hypergraph())
    
    def generate_random_hypergraph(self):
        nodes_list = list(range(self.N))
        edges_list = []
        # append 2-node (pairwise) edges with probability p1
        for pw_edge in list(combinations(nodes_list, 2)):
            if random.random() < self.p1:
                edges_list.append(pw_edge)
        # append 3-node (higher-order) edges with probability p2
        for ho_edge in list(combinations(nodes_list, 3)):
            if random.random() < self.p2:
                edges_list.append(ho_edge)
        return edges_list


class ErdosRenyiSC(HigherOrderStructure):
    r"""
    Erdos-Renyi like simplicial complex on `N` nodes.
      - Aims to achieve target average degrees d1 and d2
      - Generated by creating an initial G(N, p1_initial) graph, 
      - then adding 2-simplices with probability p2_triangles, 
      - and enforcing downward closure by adding all the faces.
    """
    def __init__(self, N, d1, d2, attempts=100):
        r"""
        Returns an instance of the Erdos-Renyi like Simplicial Complex given: 
        - d1: Target average pairwise degree
        - d2: Target average number of 2-simplices (triangles) a node is part of 
        - N: Target number of nodes (NOTE: N must be > 3)
        - NOTE:
            * d1 should be high enough that p1_initial > np.log(N) / N 
            * so that G(n, p1_initial) is connected almost surely
        """
        self.p1_initial, self.p2_triangles = get_p1_p2_for_target_degrees(d1, d2, N)

        # generate the SC components
        for _ in range(attempts): # max attempts to get a connected G(n, p_init) graph of target size
            N_realized, edges, triangles = generate_er_sc_components(N, self.p1_initial, self.p2_triangles)
            if N_realized == N:
                break
        else:
            print(f"Could not generate connected G(n, p1) on {N} nodes after 1000 attempts, increase d1.")
        # TODO: Could consider taking giant component of G(n, p1) instead

        # initialize HO structure with N_realized
        super().__init__(N_realized)
        self.d1_target = d1
        self.d2_target = d2
        self.name = f"Erdos-Renyi-SC"

        # combine all simplices
        all_simplices = []
        all_simplices.extend(edges)
        all_simplices.extend(triangles)
        self.set_edges(all_simplices) # calculates memo_nbs

        self.calculate_realized_properties()

    def calculate_realized_properties(self):
        r""" Helper to calculate realized degrees and probabilities. """
        if self.N == 0:
            self.d1_realized = 0.0
            self.d2_realized = 0.0
            self.p1_realized = 0.0
            self.p2_realized = 0.0
            self.realized_pw_edges = 0
            self.realized_ho_edges = 0
            return
        
        self.realized_pw_edges = len(self.get_edges(order=1))
        self.realized_ho_edges = len(self.get_edges(order=2))

        pw_degrees = [len(self.neighbors(i, 1)) for i in range(self.N)]
        ho_degrees = [len(self.neighbors(i, 2)) for i in range(self.N)]
        self.d1_realized = np.mean(pw_degrees)
        self.d2_realized = np.mean(ho_degrees)

        self.max_pw_edges = comb(self.N, 2)
        self.max_ho_edges = comb(self.N, 3)
        self.p1_realized = self.realized_pw_edges / self.max_pw_edges
        self.p2_realized = self.realized_ho_edges / self.max_ho_edges

        self.calculate_expected_p1_overall()
        self.p1_edges = self.calculate_expected_p1_overall()

    def calculate_expected_p1_overall(self):
        r"""
        Calculates expected overall probability of a pairwise edge existing in ErdosRenyiSC.
        """
        # prob that (u, v) is not formed by any of the N - 2 triangles involving u,v
        prob_no_triangle_forms_uv = (1.0 - self.p2_triangles)**(self.N - 2)
        # prob that (u, v) is formed by at least one triangle
        prob_triangle_forms_uv = 1.0 - prob_no_triangle_forms_uv
        expected_p1 = self.p1_initial + (1.0 - self.p1_initial) * prob_triangle_forms_uv
        return expected_p1
    
    def is_sc_valid(self):
        r""" Checks for downward closure property for all 2-simplices. """
        edge_set = set()
        for edge in self.get_edges(order=1):
            edge_set.add(tuple(sorted(edge)))

        for triangle in self.get_edges(order=2):
            v1, v2, v3 = triangle
            if not (tuple(sorted((v1, v2))) in edge_set and
                    tuple(sorted((v1, v3))) in edge_set and
                    tuple(sorted((v2, v3))) in edge_set):
                print(f"Downward closure violated!")
                return False
        return True

    def summary(self):
        # super().summary()
        print(f"\tTarget d1: {self.d1_target:.2f}, Realized d1: {self.d1_realized:.2f}")
        print(f"\tTarget d2: {self.d2_target:.2f}, Realized d2: {self.d2_realized:.2f}\n")
        
        print(f"\tInitial p1 used for G(N, p1): {self.p1_initial:.4f}")
        print(f"\tExpected p1 used for pw edges: {self.p1_edges:.8f}")
        print(f"\tExpected p2 used for ho edges: {self.p2_triangles:.8f}\n")

        print(f"\tRealized p1_overall: {self.p1_realized:.4f}")
        print(f"\tRealized p2: {self.p2_realized:.8f}\n")

        print(f"\tRealized pw edges:  {self.realized_pw_edges}/{self.max_pw_edges}")
        print(f"\tRealized ho edges:  {self.realized_ho_edges}/{self.max_ho_edges}\n")

        # print(f"\tIs valid SC: {self.is_sc_valid()}") # TODO: comment out to speed-up
        print(f"")
        print("\n")

