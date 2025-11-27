**The purpose of this file is to simulate trajectories for disease cases.**

There should be three files: one to calculate the serial weights (with the triangular kernel), one to generate single-trajectory paths, and another file that does the batch processing 

There should be at least these public APIs, i.e. functions that can be called outside of their own .py files. This is so that we can run the lines of code from elsewhere, for example from `cli.py` like this:

```
batch_process(N, w, max_days, R)

return (array shape (N, max_days), csv_path)
```

**1. calculate_serial_weights.py**

Purpose: this will compute discrete-time serial interval weights w_k from a continuous gamma distribution g(u) 
Functions: 
- `compute_serial_weights()`
  
  - Input: mean, std of disease (this is the continuous serial interval). Maximum number of days.
  - Output: weights, `w`, an array of integers of size k_max; array of discrete-week weights. Its important that this sums to one, so use `scipy.signal.normalize` or otherwise.

**2. generate_single_trajectory.py**

Purpose: this file produces a sequence of daily incident counts $I_t$ for t=1...k_max, where k_max is a user-specified upper limit, e.g. k_max for a 50-day simulation of cases. Either take a user-specifies first few weeks of cases or just randomly generate the second case (the first case is always 1?).

Functions:  
- `simulate_trajectory()`

  - Input: weights `w` from `compute_serial_weights()`. Max number of days. Initial cases (if necessary, could be `=None`). R value for this trajectory. 
  - Output: $I_{t}$, an array of integers of size k_max; array of simulated case numbers for each.
  
- `calculate_R()`

  - Input: user-specified range, e.g. `[0,5]` for `[R_min, R_max]`.
  - Output: `R`, an integer; the basic reproduction number.
 
**3. batch_processing.py**

Purpose: this file will call `generate_single_trajectory` as many times as needed to simulate a user-defined number of tracks. It should also write these results to a .csv or a Python `tempfile`. It should write to a CSV with headers day_1,...,day_max; may also include R.


Functions:
- `batch_process()`
  
  - Input: the number of tracks `N`, and the maximum number of days.
  - Output: a 2D array, printed to a csv.
