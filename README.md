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

Input: first `k` weeks of infectious cases, e.g. `k[0:3]' of 'k=[1,2,6,8,...]`.\n
Output: a CSV file with case number entries, columns are days, e.g. '`day_1'`,'`day_2'`,'`day_3'`,....
  - Consider using the `tempfile` method in Python rather than saving to the user's directory every time?
  - 
### Method 1: Analytic Solution

What to do:
1. Numerically compute the integral for the serial interval distribution $w_{k}=\int_{k-1}{k+1}(1-|u-k|)g(u)du$
  - Use SciPy.integrate or other faster methods
  - k: number of weeks, so 1/k is length of timestep, and k=2,3,4,...
    - By definition $\sum{k=0}^{\infty}=1$, i.e. it is a val id probability distribution because the sum of all entries is 1$
  - Continuous serial interval takes the value $u$ weeks 

2. Compute the expected number of new cases
  - Possibly use the @lru_cache decorator to speed up calculation
  - 
### Method 2: Trajectory Matching
### Method 3: Machine Learning
