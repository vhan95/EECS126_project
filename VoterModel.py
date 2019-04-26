# VoterModel.py
# A class for building, running, and analyzing voter models

# Victor Han, Josh Sanz, Robert Wang
# 2019-04-03

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import imageio
from collections import defaultdict


class Voter:
    """Representation of a single voter in a larger model"""
    voting_methods = ('simple', 'probability', 'weighted_prob', 'single_neighbor')

    def __init__(self, degree, belief=0, paccept=1.0):
        """
        Construct a Voter

        Parameters:
          degree: the degree of the node representing this Voter
          belief: initial belief, tuple of value {0, 1, 2} and weight [0, 1]
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

    def push_vote(self, other):
        """Push self's belief on the other without receiving the other's belief"""
        other._votes.append(self.belief)

    def update(self, method):
        assert method in self.voting_methods, "unknown voting method"
        if not self._votes:
            return
        cnts = defaultdict(lambda : 0)
        wgts = defaultdict(lambda : 0)
        # Voter's pre-existing belief carries weight
        if self.belief[0] != 0:
            cnts[self.belief[0]] += 1
            wgts[self.belief[0]] += self.belief[1]
        for v in self._votes:
            if v[0] == 0:
                continue  # Neutral votes don't cause belief changes
            cnts[v[0]] += 1
            wgts[v[0]] += v[1]
        ##### SIMPLE #####
        if method in ["simple", "single_neighbor"]:
            # Majority non-neutral vote wins
            # Probabilistically accept update
            accept = np.random.rand() < self.paccept
            b_new = self.belief[0]
            cnt_max = 0
            for b in cnts:
                # Allow other beliefs to override internal if the count is equal
                # Necessary for the all-unique and single-voter cases to run
                if cnts[b] > cnt_max or (cnts[b] == cnt_max and b != self.belief[0]):
                    b_new = b
                    cnt_max = cnts[b]
            self.belief = (b_new, 1.)
        ##### PROBABILITY #####
        elif method == "probability":
            # Get the probabilities for each belief
            belief_list = list(cnts.keys())
            belief_probs = [cnts[b] / (self.degree + 1) for b in belief_list] # self-vote adds a degree
            # Add the probability of no change
            belief_list += [-1]
            belief_probs += [1 - sum(belief_probs)]
            draw = np.random.choice(belief_list, p=np.array(belief_probs))
            if draw == -1:
                draw = self.belief[0]
            self.belief = (draw, 1.)
        ##### WEIGHTED PROBABILITY #####
        elif method == "weighted_prob":
            belief_list = list(wgts.keys())
            belief_probs = [wgts[b] / (self.degree + 1) for b in belief_list]
            # Add the probability of no change
            belief_list += [-1]
            belief_probs += [1 - sum(belief_probs)]
            belief_probs = [x*(x>=0) for x in belief_probs]
            draw = np.random.choice(belief_list, p=np.array(belief_probs))
            if draw == -1:
                pass
            else:
                new_wgt = wgts[draw] / self.degree
                if draw == self.belief[0]:
                    # Beliefs don't decrease in strength
                    new_wgt = max(wgts[draw] / (self.degree + 1), self.belief[1])
                self.belief = (draw, new_wgt)
        # Reset the votes for the next update
        self._votes = []


class VoterModel:
    """A class for building, running, and a, nalyzing voter models"""
    init_methods = ('rand_pair', 'all_rand', 'all_rand_two', 'all_rand_n', 'all_unique')
    visualization_methods = ('shell', 'random', 'kamada_kawai', 'spring', 'spectral', 'circular')

    def __init__(self, graph=None, voting='simple', clock='discrete', nbeliefs=2, visualization='shell', redraw=False):
        """
        Construct a VoterModel.

        Parameters:
          graph: a networkx graph representing the model connectivity
                 automatically generates an E-R(50, 0.125) graph if None
          voting: string representing the voting and belief update method
                  valid options are: {simple, probability, weighted_prob}
          clock: string representing the time scale at which voters decide
                 to change their belief (discrete or exponential)
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
        
        self.clock = clock

        assert voting in Voter.voting_methods, "voting method must be in {}".format(Voter.voting_methods)
        self.voting = voting

        assert nbeliefs == 2, "only 2 beliefs allowed for now"
        self.nbeliefs = nbeliefs
        
        assert visualization in self.visualization_methods, "visualization method must be in {}".format(self.visualization_methods)
        self.visualization = visualization

        self._voters = []
        
        self.redraw = redraw
        
        self.node_pos = None
        
        if self.visualization == 'shell':
            pass
        elif self.visualization == 'random':
            self.node_pos=nx.random_layout(self.graph) 
        elif self.visualization == 'kamada_kawai':
            pass
        elif self.visualization == 'spring':
            self.node_pos=nx.spring_layout(self.graph) 
        elif self.visualization == 'spectral':
            self.node_pos=nx.spectral_layout(self.graph) 
        elif self.visualization == 'circular':
            self.node_pos=nx.circular_layout(self.graph) 

        self.init_method = None
        
        
        

    def initialize(self, init_method, k=0):
        """Initialize nodes based on a model"""
        assert init_method in self.init_methods, "initialization method must be in {}".format(self.init_methods)
        degrees = [(n, nx.degree(self.graph, n)) for n in self.graph.nodes]
        if init_method == "rand_pair":
            self._voters = [Voter(d, (0, 1.), 1.0) for _, d in degrees]
            vupdate = np.random.choice(self.graph.order(), 2)
            self._voters[vupdate[0]].belief = (1, 1.)
            self._voters[vupdate[1]].belief = (2, 1.)
        elif init_method == "all_rand":
            self._voters = [Voter(d, (np.random.choice([0, 1, 2]), 1.), 1.0) for _, d in degrees] 
            
        elif init_method == "all_rand_two":
            # Sets k voters to Belief 1, the remaining n-k voters to Belief 2
            self._voters = []
            n = self.graph.number_of_nodes()
            for i in range(1,k+1):
                _, d = degrees[i-1]
                self._voters.append(Voter(d, (1, 1.), 1.0))
            for i in range(k+1,n+1):
                _, d = degrees[i-1]
                self._voters.append(Voter(d, (2, 1.), 1.0))
            
        elif init_method == "all_rand_n":
            n = self.graph.number_of_nodes()
            self._voters = [Voter(d, (np.random.randint(1,high=(n+1)), 1.), 1.0) for _, d in degrees]
            
        elif init_method == "all_unique":
            self._voters = []
            n = self.graph.number_of_nodes()
            for i in range(1,n+1):
                _, d = degrees[i-1]
                self._voters.append(Voter(d, (i, 1.), 1.0))

        self.init_method = init_method
        
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
    def belief_to_bwr(belief):
        """Convert {neutral=0, b1=1, b2=2} to the bwr colormap"""
        table = (0, 1, -1)
        return table[belief[0]] * belief[1]
    
    @staticmethod
    def belief_to_tab10(belief):
        """Convert to the tab10 colormap"""
        return belief[0] % 10
        
    def draw(self):
        """Plot the current state with matplotlib"""
        if self.init_method == "all_rand_n" or self.init_method == "all_unique":
            colors = [self.belief_to_tab10(v.belief) for v in self._voters]
            cmap = 'tab10'
        else:
            colors = [self.belief_to_bwr(v.belief) for v in self._voters]
            cmap = 'bwr'
        labels = {}
        for n in self.graph.nodes:
            labels[n] = str(self._voters[n].belief[0])
        options = {
            'node_color': colors,
            'labels': labels,
            'font_weight': 'bold',
            'node_size': 200,
            'width': 3,
            'cmap': cmap
        }
        if self.init_method != "all_rand_n" and self.init_method != "all_unique":
            options['vmin'] = -1
            options['vmax'] = 1

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
        elif self.visualization == 'kamada_kawai':
            nx.draw_kamada_kawai(self.graph, ax=self.ax, **options)   
        else:
            nx.draw(self.graph, pos=self.node_pos, ax=self.ax, **options) 
            

        # save the resulting figure so that we can make a gif later if wanted    
        # the following three lines were taken from https://ndres.me/post/matplotlib-animated-gifs-easily/
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
        time_arr = []
        
        ## Exchange votes across edges ##
        # Continuous time, single voter version
        if self.clock == "exponential":
            # Get current beliefs
            for v in self._voters:
                current_belief_arr.append(v.belief[0])
            updated_belief_arr = current_belief_arr.copy()
            # Every voter has an exponential clock with rate 1/n
            # The minimum of all these clocks is exponential with rate 1
            time_arr.append(np.random.exponential(1))
            # Which voter woke up?
            node = np.random.choice(len(self._voters))
            edges = list(self.graph.edges(node))
            if self.voting == 'single_neighbor':
                # Convert a single neighbor at random
                edge = edges[np.random.choice(len(edges))]
                neighbor = edge[1] if edge[0] == node else edge[0]
                self._voters[node].push_vote(self._voters[neighbor])
                self._voters[neighbor].update(self.voting)
                updated_belief_arr[neighbor] = self._voters[neighbor].belief[0]
            else:
                # Convert all neighbors
                for e in edges:
                    neighbor = e[1] if e[0] == node else e[0]
                    self._voters[node].push_vote(self._voters[neighbor])
                    self._voters[neighbor].update(self.voting)
                    updated_belief_arr[neighbor] = self._voters[neighbor].belief[0]
        # Discrete time simultaneous voting version
        else:
            time_arr.append(1)
            for e in self.graph.edges:
                self._voters[e[0]].exchange_votes(self._voters[e[1]])
            # Update based on votes
            for v in self._voters:
                current_belief_arr.append(v.belief[0])
                v.update(self.voting)
                updated_belief_arr.append(v.belief[0])

        return current_belief_arr, updated_belief_arr, time_arr
