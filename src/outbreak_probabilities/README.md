This directory is the source of the package `outbreak_probabilities`.

It holds the following folders:

`simulate`: simulate synthetic outbreak data using an epidemiological branching-process model

`analytic`: computes the analytic solution for the probability of a major outbreak (PMO) from initial case counts

`trajectory_matching`: selects simulated trajectories that match a given observed sequence, and estimates PMO empirically from them

`machine_learning`: trains RF/GB classifiers to predict PMO directly from early weekly case counts, and serves predictions from the packaged models

`compare`: combines the analytic, trajectory-matching and ML estimates onto one plot

## Installation

```
pip install outbreak-probabilities
```

or, for local development from a checkout of this repo:

```
pip install -e .
```

Installing the package puts the commands below on your `PATH`.

**For every command, check the available options with `--help`.**

## CLI commands

### `outbreak` — simulate, plot, match, and estimate PMO

`outbreak` is the main entry point. It has five subcommands: `simulate`, `plot`, `match`, `pmo_vs_r`, `analytic`.

**1. `outbreak simulate`** — generate synthetic outbreak data

```
outbreak simulate
```

Generates 1000 simulated outbreaks with R drawn uniformly from [0, 10] and writes them to `data/test_simulations.csv` (relative to your current directory). This is the input data every other command downstream reads from.

```
outbreak simulate --N 100000 --seed 42 --out data/test_simulations_1M.csv
```

**2. `outbreak plot`** — plot simulated trajectories

```
outbreak plot
```

Plots a sample of trajectories from a simulation CSV, plus a PMO histogram. Mainly useful for sanity-checking that simulate produced sensible paths.

```
outbreak plot --csv data/test_simulations.csv --sample-strategy random
```

**3. `outbreak match`** — find trajectories matching an observed sequence

```
outbreak match
```

Finds every simulated outbreak whose first weeks match a given initial condition (default `1,2,3`: week 1 has 1 case, week 2 has 2, week 3 has 3) and plots them.

```
outbreak match --initial-cases 1,2,0 --sample-strategy random --sample-size 10000
```

**4. `outbreak pmo_vs_r`** — empirical PMO from trajectory matching

```
outbreak pmo_vs_r
```

Plots the running PMO estimate (fraction of matched trajectories that became a major outbreak) as more matches are sampled, i.e. as a function of `r` = number of matches used.

```
outbreak pmo_vs_r --full-index
```
Runs the estimate over the full simulation index rather than a sample.

```
outbreak pmo_vs_r --full-index --initial-cases 1,2,0 --sim-csv data/test_simulations_1M.csv --sample-size 200
outbreak pmo_vs_r --initial-cases 1,0
```

**5. `outbreak analytic`** — closed-form PMO estimate

```
outbreak analytic --initial-cases 1,2,3
```

Computes PMO directly from a branching-process formula integrated over a posterior on R, given the observed initial weekly counts. Doesn't need any simulation data.

```
outbreak analytic --initial-cases 1,2,3 --r-min 0 --r-max 15 --print-grid
```

### `outbreak-predict` — ML-based PMO prediction

Serves predictions from the RF/GB classifiers shipped with the package (trained on 2, 3, 4, or 5 weeks of data; 5-week only has an RF model).

**List available models:**

```
outbreak-predict --list
```

**Predict from a single observation** (one value per `--week`, in order, matching the model's week count):

```
outbreak-predict --weeks 2 --model RF --week 2.1 --week 1.4
```

**Predict for a whole CSV at once:**

```
outbreak-predict --weeks 3 --model GB --batch cases.csv --out predictions.csv
```
`cases.csv` must have columns `week_1`, `week_2`, `week_3` (matching whatever `--weeks` you chose). If `--out` is omitted, results print to the terminal instead.

Add `--threshold 0.7` to change the probability cutoff used to classify "major" vs "minor" (default `0.5`).

### `outbreak-pmo-error-grid` — analytic vs. empirical PMO diagnostic

```
outbreak-pmo-error-grid --sim-csv data/test_simulations.csv --out-dir figs/pmo_error_grid
```

Sweeps a grid of initial conditions `[1, w2, w3]` (both `w2` and `w3` ranging `0..--max-digit`, default 9) and, for each, compares the analytic PMO estimate against the empirical trajectory-matching estimate. Writes a CSV of results plus two heatmaps: relative error, and how many matched samples were needed before the empirical estimate settled within `--rel-error-threshold` (default 1%) of the analytic one.

### `outbreak-benchmark` — simulate performance benchmark

```
outbreak-benchmark
```

Runs a fixed suite of simulate scenarios (varying N and R range) and reports runtime and PMO estimates for each, writing CSVs to `data/`. Scenarios are currently hardcoded in the script rather than configurable via flags.

## Advanced / maintainer tools (not installed as commands)

These regenerate the packaged ML models and the example comparison plots. They're not exposed as top-level commands, since normal users only need `outbreak-predict` to consume the already-trained models — but they can still be run as modules:

```
python -m outbreak_probabilities.machine_learning.ML_2weeks --data data/test_simulations.csv --model-dir path/to/write/models
```
Same pattern for `ML_3weeks`, `ML_4weeks`, `ML_5weeks`, and `ML_different_data_sizes` (trains across increasing training-set sizes, 500 to 35000 samples, for the convergence plots below). Both flags default to sensible values if omitted (`--data` to `data/test_simulations.csv`, `--model-dir` to the package's own model folder for that script).

```
python -m outbreak_probabilities.machine_learning.PLot_GBRF_different_data_sizes
```
Plots how ML predictions converge to the analytic solution as training data size grows (requires models already trained via `ML_different_data_sizes`).

```
python -m outbreak_probabilities.compare.compare_combined
python -m outbreak_probabilities.compare.comparison_r_vs_ml
```
Each combines the analytic, trajectory-matching, and ML curves onto one plot for a hardcoded initial condition (edit the `INITIALS`/`INITIAL` constant near the bottom of the file to change it).
