# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:07:42 2017

@author: Jeff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Relationship between Binomial and Poisson distributions

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10,size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n=[20,100,1000]
p = [.5, .1, .01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n=n[i], p=p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))
    
 #   The means are all about the same, which can be shown to be true by 
 # doing some pen-and-paper work. The standard deviation of the 
 # Binomial distribution gets closer and closer to that of the Poisson 
 # distribution as the probability p gets lower and lower.
 

 
 # Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115,size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large/10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)