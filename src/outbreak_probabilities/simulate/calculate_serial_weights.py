# ###
# 1. calculate_serial_weights.py

# Purpose: this will compute discrete-time serial interval weights w_k from a continuous gamma distribution g(u) 
# Functions: 
# - compute_serial_weights()
  
#   - Input: mean, std of disease (this is the continuous serial interval). Maximum number of days.
#   - Output: weights, `w`, an array of integers of size k_max; array of discrete-week weights. Its important that this sums to one, so use `scipy.signal.normalize` or otherwise.
# ###

