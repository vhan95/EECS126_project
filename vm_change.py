# vm_change.py
# A script for tracking and visualizing changes in our voting model

# Victor Han, Josh Sanz, Robert Wang
# 2019-04-21

import matplotlib.pyplot as plt

"""
Track Changes:
Everytime the beliefs of individuals in the voter population get updated, this function will keep track of:
	
	1. The average number of vote changes per update
	2. The composition of beliefs in the population per update
"""

def track_changes(current_beliefs, updated_beliefs, flux_arr, belief_arr):
	flux = sum(i != j for i, j in zip(current_beliefs, updated_beliefs))
	flux_arr.append(flux/len(updated_beliefs))
	belief_arr.append([updated_beliefs.count(0)/len(updated_beliefs), updated_beliefs.count(1)/len(updated_beliefs), updated_beliefs.count(2)/len(updated_beliefs)])

	return flux_arr, belief_arr

"""
Plot Flux:
Plots the average number of vote changes per update across all updates
"""
def plot_flux(flux_arr):
    plt.plot(flux_arr)
    plt.xlabel('Iterations')
    plt.ylabel('Average Number of Vote Changes per Iteration')
    plt.show()

def plot_comparisons(belief_arr):
    b0 = [item[0] for item in belief_arr]
    b1 = [item[1] for item in belief_arr]
    b2 = [item[2] for item in belief_arr]
    plt.plot(b0)
    plt.plot(b1)
    plt.plot(b2)
    plt.xlabel('Iterations')
    plt.ylabel('Percentage of Votes')
    plt.legend(['No belief', 'Belief 1', 'Belief 2'] , loc='upper left')
    plt.show()