# ###
# 1. calculate_serial_weights.py

# Purpose: this will compute discrete-time serial interval weights w_k from a continuous gamma distribution g(u) 
# Functions: 
# - compute_serial_weights()
  
#   - Input: mean, std of disease (this is the continuous serial interval). Maximum number of days.
#   - Output: weights, `w`, an array of integers of size k_max; array of discrete-week weights. Its important that this sums to one, so use `scipy.signal.normalize` or otherwise.
# ###

import numpy as np
from scipy.stats import gamma

def compute_serial_weights(mean, std, k_max):
    
    # Raise some errors
    if k_max < 1:
        raise ValueError("k_max must be > 1")
    
    # A few more errors to consider: k_max has to be an integer, mean & std data type

    # Instantiate array of weights
    w = np.zeros(k_max)

    # Calculate the probability density function g(x) using given params
    var = std ** 2
    alpha = mean ** 2
    theta = var / mean

    g = gamma(a = alpha, scale = theta)

    # k_1 is always 1
    if k_max == 1:
        return np.array([1.0], dtype=float)

    # Compute for k = 2,...,k_max

    # Define k's. An 2d array of floats
    k = np.arrange(2, k_max+1)

    # Normalize w so that they sum to 1
    w = w.sum()
    return w

print(compute_serial_weights(1,2,3))