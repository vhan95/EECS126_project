# graphs.py
# A script for generating various types of graphs

# Victor Han, Josh Sanz, Robert Wang
# 2019-04-20

import networkx as nx
import numpy as np 

"""
Erdos Renyi Algorithm:

The following algorithm will generate a random graph, G, with n nodes and edges that form with probability p:
1. Generate a complete graph with n nodes.
2. For each edge in the graph:
	a. Delete the edge with probability 1-p.

"""

def erdos_renyi(numNodes, prob):
	graph = nx.complete_graph(numNodes)
	final_graph = graph.copy()

	for u,v in graph.edges():
		if np.random.binomial(1,1-prob,1).tolist()[0] == 1:
			final_graph.remove_edge(u,v)

	return final_graph

"""
Complete Graph:

Returns a complete graph, G, on n nodes
"""

def complete_graph(numNodes):
	return nx.complete_graph(numNodes)

"""
Cycle:

Returns a cycle with n nodes
"""

def cycle(numNodes):
	return nx.cycle_graph(numNodes)

"""
2D Torus:

Returns a graph with 2D torus topology.

"""
def torus_2d(dim1, dim2):
	graph = nx.grid_2d_graph(dim1,dim2,periodic=True)
	final_graph = nx.convert_node_labels_to_integers(graph)

	return final_graph