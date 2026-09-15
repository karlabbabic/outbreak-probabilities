# outbreak-probabilities

This branch (`package`) is used to maintain the PyPI package hosted [here](https://test.pypi.org/project/outbreak-probabilities/1.0.1/). 

Simulation and analysis tools for estimating the probability of a major outbreak (PMO) from early case counts, using three independent methods: an analytic branching-process solution, trajectory matching against simulated data, and trained machine-learning classifiers.

## Installation

```
pip install outbreak-probabilities
```

## Quick start

Generate simulated outbreak data, then estimate PMO for an observed sequence of weekly case counts using each method:

```
outbreak simulate --N 10000 --out data/test_simulations.csv
outbreak analytic --initial-cases 1,2,3
outbreak match --sim-csv data/test_simulations.csv --initial-cases 1,2,3
outbreak-predict --weeks 3 --model RF --week 1 --week 2 --week 3
```

## Methods

- **Analytic** — computes PMO directly from a branching-process formula, integrated over a posterior on the reproduction number R given the observed initial weekly counts.
- **Trajectory matching** — finds simulated trajectories whose first *k* weeks match an observed sequence, and estimates PMO as the empirical fraction that became major outbreaks.
- **Machine learning** — RF/GB classifiers trained on simulated trajectories to predict PMO directly from early case counts, shipped with the package and served via `outbreak-predict`.

## CLI commands

The package installs these commands: `outbreak` (subcommands `simulate`, `plot`, `match`, `pmo_vs_r`, `analytic`), `outbreak-predict`, `outbreak-pmo-error-grid`, and `outbreak-benchmark`.

See [`src/outbreak_probabilities/README.md`](src/outbreak_probabilities/README.md) for full documentation of every command, its options, and examples.

## Development

```
git clone <repo-url>
cd outbreak-probabilities
pip install -e .[dev]
pytest src/outbreak_probabilities/tests
```
