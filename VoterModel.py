# VoterModel.py
# A class for building, running, and analyzing voter models

# Victor Han, Josh Sanz, Robert Wang
# 2019-04-03

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import imageio


class Voter:
    """Representation of a single voter in a larger model"""
    voting_methods = ('simple', 'probability', 'weighted_prob')
    visualization_methods = ('shell','random','kamada_kawai')

    def __init__(self, degree, belief=0, paccept=1.0, handicap_b1=1.0, handicap_b2=1.0):
        """
        Construct a Voter

        Parameters:
          degree: the degree of the node representing this Voter
          belief: initial belief, tuple of value {0, 1, 2} and weight [0, 1]
          paccept: probability of accepting a belief update
          handicap_b1: the propagation handicap for belief 1, [0, 1]
                       1.0 is no handicap
          handicap_b2: the propagation handicap for belief 2, [0, 1]
                       1.0 is no handicap
        """
        self.degree = degree
        self.belief = belief
        self.paccept = paccept
        self.handicap_b1 = handicap_b1
        self.handicap_b2 = handicap_b2

        self._votes = []

    def exchange_votes(self, other):
        """Exchange votes across an edge"""
        self._votes.append(other.belief)
        other._votes.append(self.belief)

    def update(self, method):
        assert method in self.voting_methods, "unknown voting method"
        cnt_b1 = cnt_b2 = wgt_b1 = wgt_b2 = 0
        # Voter's pre-existing belief carries weight
        if self.belief[0] == 1:
            cnt_b1 += 1
            wgt_b1 += self.belief[1]
        if self.belief[0] == 2:
            cnt_b2 += 1
            wgt_b2 += self.belief[1]
        for v in self._votes:
            if v[0] == 1:
                cnt_b1 += 1
                wgt_b1 += v[1]
            elif v[0] == 2:
                cnt_b2 += 1
                wgt_b2 += v[1]
        # Apply handicaps
        cnt_b1 *= self.handicap_b1
        wgt_b1 *= self.handicap_b1
        cnt_b2 *= self.handicap_b2
        wgt_b2 *= self.handicap_b2
        ##### SIMPLE #####
        if method == "simple":
            # Majority non-neutral vote wins
            # Probabilistically accept update
            accept = np.random.rand() < self.paccept
            if (cnt_b1 + cnt_b2) == 0:
                pass  # Own belief is unchanged
            elif cnt_b1 > cnt_b2 and accept:
                self.belief = (1, 1.)
            elif cnt_b2 > cnt_b1 and accept:
                self.belief = (2, 1.)
        ##### PROBABILITY #####
        elif method == "probability":
            p_b1 = cnt_b1 / self.degree
            p_b2 = cnt_b2 / self.degree
            draw = np.random.rand()
            if draw < p_b1:
                self.belief = (1, 1.)
            elif draw > (1 - p_b2):
                self.belief = (2, 1.)
        ##### WEIGHTED PROBABILITY #####
        elif method == "weighted_prob":
            p_b1 = wgt_b1 / self.degree
            p_b2 = wgt_b2 / self.degree
            draw = np.random.rand()
            if draw < p_b1:
                if self.belief[0] == 1:
                    p_b1 = max(p_b1, self.belief[1])  # Beliefs don't decrease in strength
                self.belief = (1, p_b1)
            if draw > (1 - p_b2):
                if self.belief[0] == 2:
                    p_b2 = max(p_b2, self.belief[1])  # Beliefs don't decrease in strength
                self.belief = (2, p_b2)
        # Reset the votes for the next update
        self._votes = []


class VoterModel:
    """A class for building, running, and a, nalyzing voter models"""
    init_methods = ('rand_pair', 'all_rand', 'all_rand_two', 'all_rand_n')

    def __init__(self, graph=None, voting='simple', handicap_b1=1., handicap_b2=1., nbeliefs=2, visualization='shell', redraw=True):
        """
        Construct a VoterModel.

        Parameters:
          graph: a networkx graph representing the model connectivity
                 automatically generates an E-R(50, 0.125) graph if None
          voting: string representing the voting and belief update method
                  valid options are: {simple, probability, weighted_prob}
          handicap_b1: the propagation handicap for belief 1, [0, 1]
                       1.0 is no handicap
          handicap_b2: the propagation handicap for belief 2, [0, 1]
                       1.0 is no handicap
          nbeliefs: the number of possible beliefs
                    must be 2 for now
          visualization: string representing the visualization method
          redraw: boolean which is true if visualization plots should be 
                  redrawn on the same axes and false otherwise
        """
        if graph is None:
            self.graph = nx.erdos_renyi_graph(50, 0.125)
        else:
            self.graph = graph

        self.handicap_b1 = handicap_b1
        self.handicap_b2 = handicap_b2

        assert voting in Voter.voting_methods, "voting method must be in {}".format(Voter.voting_methods)
        self.voting = voting

        assert nbeliefs == 2, "only 2 beliefs allowed for now"
        self.nbeliefs = nbeliefs
        
        assert visualization in Voter.visualization_methods, "visualization method must be in {}".format(Voter.visualization_methods)
        self.visualization = visualization

        self._voters = []
        
        self.redraw = redraw
        

    def initialize(self, init_method):
        """Initialize nodes based on a model"""
        assert init_method in self.init_methods, "initialization method must be in {}".format(self.init_methods)
        degrees = [(n, nx.degree(self.graph, n)) for n in self.graph.nodes]
        if init_method == "rand_pair":
            self._voters = [Voter(d, (0, 1.), 1.0,
                                  handicap_b1=self.handicap_b1, handicap_b2=self.handicap_b2) for _, d in degrees]
            vupdate = np.random.choice(self.graph.order(), 2)
            self._voters[vupdate[0]].belief = (1, 1.)
            self._voters[vupdate[1]].belief = (2, 1.)
        elif init_method == "all_rand":
            self._voters = [Voter(d, (np.random.choice([0, 1, 2]), 1.), 1.0, 
                                  handicap_b1=self.handicap_b1, handicap_b2=self.handicap_b2) for _, d in degrees] 
            
        elif init_method == "all_rand_two":
            self._voters = [Voter(d, (np.random.choice([1, 2]), 1.), 1.0, 
                                  handicap_b1=self.handicap_b1, handicap_b2=self.handicap_b2) for _, d in degrees] 
            
        elif init_method == "all_rand_n":
            n = self.graph.number_of_nodes()
            self._voters = [Voter(d, (np.random.randint(0,high=n), 1.), 1.0, 
                                  handicap_b1=self.handicap_b1, handicap_b2=self.handicap_b2) for _, d in degrees] 
        
        # setup drawing and saving the resulting gif   
        if self.redraw:
            plt.ion()
            #self.fig, self.ax = plt.subplots(figsize=(10,5))
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
        else:
            plt.ioff()
        self._images = []
            

    @staticmethod
    def belief_to_cmap(belief):
        """Convert {neutral=0, b1=1, b2=2} to the bwr colormap"""
        table = (0, 1, -1)
        return table[belief[0]] * belief[1]
        
    def draw(self):
        """Plot the current state with matplotlib"""
        colors = [self.belief_to_cmap(v.belief) for v in self._voters]
        labels = {}
        for n in self.graph.nodes:
            labels[n] = str(self._voters[n].belief[0])
        options = {
            'node_color': colors,
            'vmin': -1,
            'vmax': 1,
            'labels': labels,
            'font_weight': 'bold',
            'node_size': 200,
            'width': 3,
            'cmap': 'bwr'
        }
        
        if self.redraw:
            self.ax.clear()
        else:
            #self.fig, self.ax = plt.subplots(figsize=(10,5))
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
        if self.visualization == 'shell':
            nonneutral = [i for i, b in enumerate(colors) if b != 0]
            neutral = [i for i, b in enumerate(colors) if b == 0]
            if len(nonneutral)==0 or len(neutral)==0:
                nx.draw_shell(self.graph, ax=self.ax, **options)
            else:
                nx.draw_shell(self.graph, ax=self.ax, nlist=[neutral, nonneutral], **options)
        elif self.visualization == 'random':
            nx.draw(self.graph, ax=self.ax, **options)
        elif self.visualization == 'kamada_kawai':
            nx.draw_kamada_kawai(self.graph, ax=self.ax, **options)   
            

        # save the resulting figure so that we can make a gif later if wanted    
        self.fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self._images.append(image)
        
    def save_gif(self, fps=1, fname='sim.gif'):
        """Save all the images in the simulation into a gif"""
        imageio.mimsave('./'+fname, self._images, fps=fps)

    def update(self):
        """Vote and update beliefs for all nodes"""
        current_belief_arr = []
        updated_belief_arr = []
        
        # Exchange votes across all edges
        for e in self.graph.edges:
            self._voters[e[0]].exchange_votes(self._voters[e[1]])
        # Update based on votes
        for v in self._voters:
            current_belief_arr.append(v.belief[0])
            v.update(self.voting)
            updated_belief_arr.append(v.belief[0])

        return current_belief_arr, updated_belief_arr
