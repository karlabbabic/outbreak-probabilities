This directory is the source of the package `outbreak_probabilities`.

It holds the following folders:

`simulate`: simulate synthetic data using epidemiological model

`analytic`: analytic solution for the outbreak probability based on simulated data

`trajectory_matching`: selects trajectories based on identical matching to given observed data

`machine_learning`: trains a ML model to classify trajectories on their outbreak probabilities

`compare`: compares outbreak probability resuls from `analycis`,`trajectory_matching`,`machine_learning`

`gui`: a simple graphical user interface 

1. Simulate
PYTHONPATH=src python -m outbreak_probabilities.runner simulate --N 100000 --seed 42 --out data/test_simulations_2.csv

2. Plot
PYTHONPATH=src python -m outbreak_probabilities.runner plot --csv data/test_simulations_2.csv --sample-strategy random

3. Match trajectories
PYTHONPATH=src python -m outbreak_probabilities.runner match --initial-cases 1,2,3,4
