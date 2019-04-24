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

def track_changes(current_beliefs, updated_beliefs, times, flux_arr, belief_arr, time_arr, beliefs):
	flux = sum(i != j for i, j in zip(current_beliefs, updated_beliefs))
	flux_arr.append(flux)

	b_arr = []
	for b in beliefs:
		b_arr.append(updated_beliefs.count(b)/len(updated_beliefs))
	belief_arr.append(b_arr)
	time_arr.append(times)

	return flux_arr, belief_arr, time_arr

"""
Plot Flux:
Plots the average number of vote changes per update across all updates
"""
def plot_flux(flux_arr):
    plt.plot(range(1,len(flux_arr)+1), flux_arr)
    plt.xlabel('Iterations')
    plt.ylabel('Number of Vote Changes per Iteration')
    plt.show()

"""
Plot Comparisons:
Plots the distribution of beliefs per iteration
"""

def plot_comparisons(belief_arr, beliefs):
    for i in range(len(beliefs)):
        dist_i = [item[i] for item in belief_arr]
        plt.plot(range(1,len(dist_i)+1), dist_i)
    plt.xlabel('Iterations')
    plt.ylabel('Percentage of Votes')
    plt.ylim(0,1)
    
    legend_arr = []
    for i in range(len(beliefs)):
        label = 'Belief ' + str(beliefs[i])
        legend_arr.append(label)
        
    plt.legend(legend_arr, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

"""
Convergence Time:
Reports the time that it takes for one belief to dominate a voter population.
"""

def convergence_time(time_arr, belief_arr):
    conv_arr = [i for i in range(len(belief_arr)) if set(belief_arr[i]) == {0,1}]
    conv_time = 0
    if len(conv_arr) == 0:
        # Implies that a belief has not completely dominated a voter population yet
        for i in range(len(time_arr)):
            # Return the total time elapsed for iterations (true time for convergence is greater than this value)
            conv_time += sum(time_arr[i])
            
    else:
        conv_idx = conv_arr[0]
        for i in range(conv_idx + 1):
            conv_time += sum(time_arr[i])
    
    return conv_time
        
        