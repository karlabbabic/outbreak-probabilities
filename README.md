# outbreak-probabilities
DTC Sandpit Challenge: methods for estimating the probability of a major outbreak

## To-Do
### Continuous Integration
- Create CI workflow in github/workflows/ci.yml
  - GitHub Actions
  - Code Coverage

### Testing
- Create tests in tests/test_name.py
  -  Test all .py files; probably means testing each Method separately, as well as IO
  -  PyTest
 
- Add readthedocs [documentation](https://docs.readthedocs.com/platform/stable/intro/add-project.html#manually-import-your-docs)

### Simulation of Trajectories
Input: first `k` weeks of infectious cases, e.g. `k[0:3]` of `k=[1,2,6,8,...]`.

Output: a CSV file `simulated_cases.csv` with case number entries, columns are days, e.g. '`day_1'`,'`day_2'`,'`day_3'`,....
  - Consider using the `tempfile` method in Python rather than saving to the user's directory every time?

### Method 1: Analytic Solution
Input: 
- the first `k` days worth of simulated infection data from `simulated_cases.csv`.
- estimated range for the reproduction number

Output: 
- The conditional probability `P([I1,I2,I3]|R)`
- Outbreak probability given first three cases `P(PMO | [I1,I2,I3)]`
- Outbreak probability given reproduction number `P(PMO | R]`
- Overall outbreak probability: `(conditional probability) x (outbreak probability given reproduction number)`
  

What to do:
1. Numerically compute the integral for the serial interval distribution $w_{k}=\int_{k-1}{k+1}(1-|u-k|)g(u)du$
  - Use SciPy.integrate or other faster methods
  - k: number of weeks where k=2,3,4,...
    - By definition $\sum{k=0}^{\infty}=1$, i.e. it is a val id probability distribution because the sum of all entries is 1$
  - Continuous serial interval takes the value $u$ weeks (continuous rather than the discrete `k`.)

2. Compute the expected number of new cases
  - Possibly use the @lru_cache decorator to speed up calculation
    
### Method 2: Trajectory Matching
Input: 
- the first `k` days worth of simulated infection data from `simulated_cases.csv`.
- an array containing an observed input sequence of early cases count e.g. `data = [1,2,6] = k[0:3]`
  
Output: 
- any trajectory of cases where the first `k` days of simulated data match the observed
- outbreak probability: the number of these cases where the last entry is greater than 100.

### Method 3: Machine Learning

Input: 
- an array containing an observed input sequence of early cases count e.g. `data = [1,2,6] = k[0:3]`
- ML method to be trained beforehand?
Output:
- ML-predicted random forest likely trees based on `data`.
- outbreak probability: average of outbreaks across all trees

