# VoterModel.py
# A class for building, running, and analyzing voter models

# Victor Han, Josh Sanz, Robert Wang
# 2019-04-03

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Voter:
    """Representation of a single voter in a larger model"""
    voting_methods = ('simple', 'probability')
    visualization_methods = ('shell','random','kamada_kawai')

    def __init__(self, degree, belief=0, paccept=1.0):
        """
        Construct a Voter

        Parameters:
          degree: the degree of the node representing this Voter
          belief: initial belief in the interval [-1, 1]
          paccept: probability of accepting a belief update
        """
        self.degree = degree
        self.belief = belief
        self.paccept = paccept

        self._votes = []

    def exchange_votes(self, other):
        """Exchange votes across an edge"""
        self._votes.append(other.belief)
        other._votes.append(self.belief)

    def update(self, method):
        assert method in self.voting_methods, "unknown voting method"
        self._votes = np.array(self._votes)
        ##### SIMPLE #####
        if method == "simple":
            # Majority non-neutral vote wins
            sum_b1 = sum(self._votes == 1)
            sum_b2 = sum(self._votes == 2)
            # Probabilistically accept update
            accept = np.random.rand() < self.paccept
            if (sum_b1 + sum_b2) == 0:
                pass  # Own belief is unchanged
            elif sum_b1 > sum_b2 and accept:
                self.belief = 1
            elif sum_b2 > sum_b1 and accept:
                self.belief = 2
        ##### PROBABILITY #####
        elif method == "probability":
            sum_b1 = sum(self._votes == 1)
            sum_b2 = sum(self._votes == 2)
            p_b1 = sum_b1 / self.degree
            p_b2 = sum_b2 / self.degree
            draw = np.random.rand()
            if draw < p_b1:
                self.belief = 1
            elif draw > (1 - p_b2):
                self.belief = 2
        # Reset the votes for the next update
        self._votes = []


class VoterModel:
    """A class for building, running, and analyzing voter models"""
    init_methods = ('rand_pair', 'all_rand')

    def __init__(self, graph=None, voting='simple', nbeliefs=2, visualization='shell'):
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
        
        assert visualization in Voter.visualization_methods, "voting method must be in {}".format(Voter.visualization_methods)
        self.visualization = visualization

        self._voters = []

    def initialize(self, init_method):
        """Initialize nodes based on a model"""
        assert init_method in self.init_methods, "initialization method must be in {}".format(self.init_methods)
        degrees = [(n, nx.degree(self.graph, n)) for n in self.graph.nodes]
        if init_method == "rand_pair":
            self._voters = [Voter(d, 0, 1.0) for _, d in degrees]
            vupdate = np.random.choice(self.graph.order(), 2)
            self._voters[vupdate[0]].belief = 1
            self._voters[vupdate[1]].belief = 2
        elif init_method == "all_rand":
            self._voters = [Voter(d, np.random.choice([0, 1, 2]), 1.0) for _, d in degrees] 

    @staticmethod
    def belief_to_cmap(belief):
        """Convert {neutral=0, b1=1, b2=2} to the bwr colormap"""
        table = (0, 1, -1)
        return table[belief]
        
    def draw(self):
        """Plot the current state with matplotlib"""
        colors = [self.belief_to_cmap(v.belief) for v in self._voters]
        options = {
            'node_color': colors,
            'node_size': 100,
            'width': 3,
            'cmap': 'bwr'
        }
        plt.subplot()
        if self.visualization == 'shell':
            nonneutral = [i for i, b in enumerate(colors) if b != 0]
            neutral = [i for i, b in enumerate(colors) if b == 0]
            if len(nonneutral)==0 or len(neutral)==0:
                nx.draw_shell(self.graph, **options)
            else:
                nx.draw_shell(self.graph, nlist=[neutral, nonneutral], **options)
        elif self.visualization == 'random':
            nx.draw(self.graph, **options)
        elif self.visualization == 'kamada_kawai':
            nx.draw_kamada_kawai(self.graph, **options)   
        plt.draw()

    def update(self):
        """Vote and update beliefs for all nodes"""
        # Exchange votes across all edges
        for e in self.graph.edges:
            self._voters[e[0]].exchange_votes(self._voters[e[1]])
        # Update based on votes
        for v in self._voters:
            v.update(self.voting)
