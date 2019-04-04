# ErdosRenyi.py
# A script for generating random graphs using the Erdos Renyi Algorithm

# Victor Han, Josh Sanz, Robert Wang
# 2019-04-04

"""
Erdos Renyi Algorithm:

The following algorithm will generate a random graph, G, with n nodes and edges that form with probability p:
1. Generate a complete graph with n nodes.
2. For each edge in the graph:
	a. Delete the edge with probability 1-p.

"""
import networkx as nx
import numpy as np 

def random_graph(numNodes, prob):
	graph = nx.complete_graph(numNodes)
	final_graph = graph.copy()

	for u,v in graph.edges():
		if np.random.binomial(1,1-prob,1).tolist()[0] == 1:
			final_graph.remove_edge(u,v)

	return final_graph