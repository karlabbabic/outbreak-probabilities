# outbreak-probabilities
DTC Sandpit Challenge: methods for estimating the probability of a major outbreak

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>

  <ol>
    <li><a href="#to-do">To-Do</a></li>
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
<h2 id="to-do">To-Do</h2>

- Simulate – done  
- Analytic – upload the cell in rough-work collab to GitHub without the sliders (input params)  
- ML – write code that uses the simulated data and include plots (train 4 separate classifiers and save the models)  
- Write unit tests (ask Matthew) to cover as many lines as you can

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
