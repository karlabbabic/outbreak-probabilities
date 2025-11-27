**The purpose of files in this folder is to calculate the PMO of synthetic data based on observed initial cases.**

This is the analytic method to calculate disease outbreak probability, based on the number of trajectories that fit the initial cases and each the outbreak threshold of 100 cases.

There should be two .py files. One that calculates the various probabilities and the overall PMO given a trajectory, and another that runs that file.

**1. calculate_probabilities.py**
Purpose: this file will calculate probablities of a major outbreak given an array of initial cases `I` e.g. `I=[1,2,3]`: the posterior `P(R|I)`, the branching conditional `P(major outbreak | I,R)` to give the PMO `P(major outbreak | I_k)`.

Functions:

- `CSV_to_array`
- `calculate_posterior()`
- `calculate_conditional()`
- `calculate_q`
- `calculate_overall_PMO()`
  
  - Input: filepath to CSV with the weights `w`. The `R` value (should be in the CSV).
  - Output: arrays containing the posterior, conditional, and overall PMO. 

**2. runner.py**

Purpose: this file should be the one calling `calculate_probabilities.py`. Could have plots running here. 
