# VoterModel.py
# A class for building, running, and analyzing voter models

# Victor Han, Josh Sanz, Robert Wang
# 2019-04-03

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Voter:
    """Representation of a single voter in a larger model"""
    voting_methods = ('simple',)

    def __init__(self, belief=0, paccept=1.0):
        """
        Construct a Voter

        Parameters:
          belief: initial belief in the interval [-1, 1]
          paccept: probability of accepting a belief update
        """
        self.belief = belief
        self.paccept = paccept

        self._votes = []

    def exchange_votes(self, other):
        """Exchange votes across an edge"""
        self._votes.append(other.belief)
        other._votes.append(self.belief)

    def update(self, method):
        assert method in self.voting_methods, "unknown voting method"
        if method == "simple":
            # Majority non-neutral vote wins
            vsum = sum(self._votes)
            # Probabilistically accept update
            accept = np.random.rand() < self.paccept
            if vsum == 0:
                pass  # Own belief is unchanged
            elif vsum < 0 and accept:
                self.belief = -1
            elif vsum > 0 and accept:
                self.belief = 1


class VoterModel:
    """A class for building, running, and analyzing voter models"""
    init_methods = ('rand_pair', 'all_rand')

    def __init__(self, graph=None, voting='simple', nbeliefs=2):
        """
        Construct a VoterModel.

        Parameters:
          graph: a networkx graph representing the model connectivity
                 automatically generates an E-R(50, 0.125) graph if None
          voting: string representing the voting and belief update method
                  valid options are: {simple}
          nbeliefs: the number of possible beliefs
                    must be 2 for now
        """
        if graph is None:
            self.graph = nx.erdos_renyi_graph(50, 0.125)
        else:
            self.graph = graph

        assert voting in Voter.voting_methods, "voting method must be in {}".format(Voter.voting_methods)
        self.voting = voting

        assert nbeliefs == 2, "only 2 beliefs allowed for now"
        self.nbeliefs = nbeliefs

        self._voters = []

    def initialize(self, init_method):
        """Initialize nodes based on a model"""
        assert init_method in self.init_methods, "initialization method must be in {}".format(self.init_methods)
        if init_method == "rand_pair":
            self._voters = [Voter(0, 1.0) for i in range(self.graph.order())]
            vupdate = np.random.choice(self.graph.order(), 2)
            self._voters[vupdate[0]].belief = 1
            self._voters[vupdate[1]].belief = -1
        elif init_method == "all_rand":
            self._voters = [Voter(np.random.choice([-1, 0, 1]), 1.0) for i in range(self.graph.order())]

    def draw(self):
        """Plot the current state with matplotlib"""
        colors = [v.belief for v in self._voters]
        options = {
            'node_color': colors,
            'node_size': 100,
            'width': 3,
            'cmap': 'bwr'
        }
        plt.subplot()
        nx.draw(self.graph, **options)
        plt.draw()
