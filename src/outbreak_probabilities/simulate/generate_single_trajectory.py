###
**2. generate_single_trajectory.py**

Purpose: this file produces a sequence of daily incident counts $I_t$ for t=1...k_max, where k_max is a user-specified upper limit, e.g. k_max for a 50-day simulation of cases. Either take a user-specifies first few weeks of cases or just randomly generate the second case (the first case is always 1?).

Functions:  
- simulate_trajectory()

  - Input: weights `w` from `compute_serial_weights()`. Max number of days. Initial cases (if necessary, could be `=None`). R value for this trajectory. 
  - Output: $I_{t}$, an array of integers of size k_max; array of simulated case numbers for each.
  
- calculate_R()

  - Input: user-specified range, e.g. `[0,5]` for `[R_min, R_max]`.
  - Output: `R`, an integer; the basic reproduction number.
###
