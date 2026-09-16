# outbreak-probabilities
DTC Sandpit Challenge: methods for estimating the probability of a major outbreak

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>

  <ol>
    <li><a href="#project-site">Project Site</a></li>
    <li><a href="#installation">Installation</a></li>
    <li>
      <a href="#set-up">Set-up</a>
      <ul>
        <li><a href="#continuous-integration">Continuous Integration</a></li>
        <li><a href="#testing">Testing</a></li>
        <li><a href="#simulation-of-trajectories">Simulation of Trajectories</a></li>
      </ul>
    </li>
    <li>
      <a href="#methods">Methods</a>
      <ul>
        <li><a href="#method-1-analytic-solution">Method 1: Analytic Solution</a></li>
        <li><a href="#method-2-trajectory-matching">Method 2: Trajectory Matching</a></li>
        <li><a href="#method-3-machine-learning">Method 3: Machine Learning</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- explicit headings with ids to ensure anchors always work -->
<h2 id="project-site">Project Site</h2>

For an overview of the project and an interactive calculator, visit the GitHub Pages site: https://karlabbabic.github.io/outbreak-probabilities/

<h2 id="installation">Installation</h2>

Install the package from TestPyPI:

```
pip install -i https://test.pypi.org/simple/ outbreak-probabilities==1.0.1
```

For full documentation of every CLI command, its options, and worked examples see [`src/outbreak_probabilities/README.md`](src/outbreak_probabilities/README.md). The README in each sub-package folder (`simulate`, `analytic`, `trajectory_matching`, `machine_learning`) documents that method in more detail:

- [`src/outbreak_probabilities/simulate/README.md`](src/outbreak_probabilities/simulate/README.md) — simulating trajectories
- [`src/outbreak_probabilities/analytic/README.md`](src/outbreak_probabilities/analytic/README.md) — the analytic solution
- [`src/outbreak_probabilities/trajectory_matching/README.md`](src/outbreak_probabilities/trajectory_matching/README.md) — trajectory matching
- [`src/outbreak_probabilities/machine_learning/README.md`](src/outbreak_probabilities/machine_learning/README.md) — the machine-learning models

<h2 id="set-up">Set-up</h2>

<h3 id="continuous-integration">Continuous Integration</h3>

- Create CI workflow in `.github/workflows/ci.yml`  
  - GitHub Actions  
  - Code Coverage

<h3 id="testing">Testing</h3>

- Create tests in `tests/test_name.py`  
  - Test all `.py` files (test each Method separately + IO)  
  - Use PyTest

- Add Read the Docs documentation: https://docs.readthedocs.com/platform/stable/intro/add-project.html#manually-import-your-docs

<h3 id="simulation-of-trajectories">Simulation of Trajectories</h3>

**Input:** first `k` weeks of infectious cases, e.g. `k[0:3]` of `k = [1,2,6,8,...]`.  
**Output:** a CSV file `simulated_cases.csv` with case number entries; columns are days, e.g. `day_1`, `day_2`, `day_3`, ...  
- Consider using the `tempfile` module rather than saving to the user directory every time.

<h2 id="methods">Methods</h2>

<h3 id="method-1-analytic-solution">Method 1: Analytic Solution</h3>

**Input:**  
- the first `k` days worth of simulated infection data from `simulated_cases.csv`  
- estimated range for the reproduction number

**Output:**  
- The conditional probability `P([I1,I2,I3] | R)`  
- Outbreak probability given first three cases `P(PMO | [I1,I2,I3])`  
- Outbreak probability given reproduction number `P(PMO | R)`  
- Overall outbreak probability: `(conditional probability) × (outbreak probability given reproduction number)`

**What to do:**  
1. Numerically compute the integral for the serial interval distribution  
2. Compute the expected number of new cases

<h3 id="method-2-trajectory-matching">Method 2: Trajectory Matching</h3>

**Input:**  
-  Sequence of case counts

**Output:**  
- All trajectories of cases where the first `k` days of simulated data match the observed sequence  
- Outbreak probability: fraction of those trajectories classified as major outbreaks

<h3 id="method-3-machine-learning">Method 3: Machine Learning</h3>

**Input:**  
- an observed input sequence of early case counts, e.g. `data = [1,2,6] = k[0:3]`  
- ML model(s) trained on simulated trajectories

**Output:**  
- predicted outbreak probability (and model metrics); saved model files (TBD)
