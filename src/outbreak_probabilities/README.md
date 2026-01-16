This directory is the source of the package `outbreak_probabilities`.

It holds the following folders:

`simulate`: simulate synthetic data using epidemiological model

`analytic`: computes the analytic solution for the outbreak probability based on initial cases 

`trajectory_matching`: selects trajectories based on identical matching to given observed data

`machine_learning`: trains a ML model to classify trajectories on their outbreak probabilities

So far, there is CLI commands to simulate, plot, trajectory match, and to compare the PMO from trajectory matching and the analytic solution. Run these commands in the terminal. 

In general, they are in the form `PYTHONPATH=src python -m outbreak_probabilities.runner [simulate,plot,match_trajectory,pmo_vs_r]`. This is so that when we turn this project into a package, it will be easy to transfer the commands, where we should be able to run it like `outbreak-probabilities [simulate,plot,match_trajectory,pmo_vs_r]`.

**For all commands, check any options with the --h (or --help) flag.**

**1. Simulate**
`PYTHONPATH=src python -m outbreak_probabilities.runner simulate`
This generates synthetic outbreak data. The default settings are 1000 outbreaks with R drawn uniformly from 0, 10. 

Examples of extra options:
`PYTHONPATH=src python -m outbreak_probabilities.runner simulate --N 100000 --seed 42 --out data/test_simulations_1M.csv`

**2. Plot**

`PYTHONPATH=src python -m outbreak_probabilities.runner plot`
This will plot the trajectories in some outbreak data. This is more for testing and checking to see that the paths are generatd.

Examples of extra options:
PYTHONPATH=src python -m outbreak_probabilities.runner plot --csv data/test_simulations.csv --sample-strategy random

**3. Match trajectories**

`PYTHONPATH=src python -m outbreak_probabilities.runner match`
This will find all outbreaks with a specified initial condition (by default 1,2,3; where week 1 has 1 case, week 2 has 2 cases, week 3 has 3 cases) and extracts them.

Examples of extra options:
`PYTHONPATH=src python -m outbreak_probabilities.runner match --initial-cases 1,2,0 --sample-strategy random --sample-size 10000`

**4. Plot the analytic PMO and running cumulative PMO of matched trajectories (and ML model, coming soon)**
`PYTHONPATH=src python -m outbreak_probabilities.runner pmo_vs_r` 

Use the `--full-index` flag for the plot to run over the full simulation index 
`PYTHONPATH=src python -m outbreak_probabilities.runner pmo_vs_r --full-index` 

Examples of extra options:
PYTHONPATH=src python -m outbreak_probabilities.runner pmo_vs_r --full-index --initial-cases 1,2,0 --sim-csv data/test_simulations_1M.csv --sample-size 200

PYTHONPATH=src python -m outbreak_probabilities.runner pmo_vs_r --initial-cases 1,0 

**5. List all available Machinge Leaning models**

`PYTHONPATH=src python -m outbreak_probabilities.predict --list`


**6. Predict PMO and probability using number of cases and model type (3 weeks of data)**

   
`PYTHONPATH=src python -m outbreak_probabilities.predict --weeks 3 --model RF --week 2.1 --week 1.4`

**7. Predicting with Input CSV File:**

`PYTHONPATH=src python -m outbreak_probabilities.predict --weeks 3 --model RF --batch inputs.csv --out preds.csv`
