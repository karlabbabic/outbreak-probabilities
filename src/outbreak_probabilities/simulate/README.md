The purpose of this file is to simulate trajectories for disease cases. 

There should be three files: one to calculate the serial weights (with the triangular kernel), one to generate single-trajectory paths, and another file that does the batch processing 

There should be at least these public APIs, i.e. functions that can be called outside of their own .py files. 

1. calculate_serial_weights.py

Purpose: this will compute discrete-time serial interval weights w_k from a continuous gamma distribution g(u) 
Functions: 
- compute_serial_weight

2. generate_single_trajectory.py

Purpose: this file produces a sequence of daily incident counts $I_t$ for t=1...$day_{max}$, where $day_{max}$ is a user-specified upper limit, e.g. $day_{max}=50$ for a 50-day simulation of cases.

Functions:  

4. batch_processing.py
