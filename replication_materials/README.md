# PyDPF run_experiments script documentation


# Running the validation script

The replication script `run_experiments.py` requires both `matplotlib` and `tqdm` in addition to PyDPF so install these before proceeding. This can be done from our provided requirements file.

PyDPF is a package designed for the implementation and comparison of deep learning algorithms, consequently many of the experiments take
a long time to run. Therefore, we provide command line arguments that allow a user to run each experiment separately or a smoke
test which runs through almost all lines of code but at lower particle counts, training iterations and experiment repeats.


Data is read from a single central folder (default: `data/`) and every result is written to a
single central folder (default: `results/`). Both are created and populated on demand. The Unsupervised learning of a single parameter
experiment, only, requires that the data folder contain the file `test_trajectory.csv` that is pre-placed in the data folder in our github.

**Important note:** by default the raw maze data is downloaded from a third party source and then pre-processed. To ensure that the data remains available we additionally 
provide the pre-processed data on zenodo at https://zenodo.org/records/23039951. The downloaded csv can be placed in the single central data folder; this will skip fetching the raw data from a third party source.

Running the full script takes a lot of time, roughly two-three days on our hardware. A large portion of which is due to testing 
the `Optimal Transport` DPF. Especially the two complete train and test runs required to time it in both deterministic and non-deterministic mode in the `maze` environment. As the total runtime is dominated by a small number of experiments, we recomend a researcher seeking to  focus

By default, methods that are run to completion on any experiment will not be rerun if the script is called again, this is to make it easy to run the script across multiple sessions. If you wish to recompute
results you can pass the `--overwrite-results` command line flag, or if you want the stored results to be entirely wiped you can pass the `--clear-results` flag.

---

## Quick start

```bash
# Everything, with paper settings
python run_experiments.py

# Validate the pipeline quickly (tiny epochs / particle counts)
python run_experiments.py --smoke

# One experiment, two methods
python run_experiments.py --experiment "sv_filtering" --models "Soft" "Stop-Gradient"

# Full argument list
python run_experiments.py --help
```


---

## Experiments

Selected with `-e` / `--experiment`. Each maps to a section of the paper.

| Name | Paper section                                                                                                   |
|---|-----------------------------------------------------------------------------------------------------------------|
| `kalman` | Linear Gaussian / Comparison with the Kalman filter                                                             |
| `proposal` | Linear Gaussian / Learning proposal parameters                                                                  |
| `sv_filtering` | Stochastic Volatility / Filtering given a fully specified model                                                 |
| `sv_single` | Stochastic Volatility / Unsupervised learning of a single parameter                                             |
| `sv_multiple` | Stochastic Volatility / Unsupervised learning of multiple parameters                                            |
| `maze` | Deep mind maze / Deep Learning                                                                                  |
| `example_usage` | Stochastic Volatility / Example usage (workflow demonstration)                                                  |
| `advanced_usage` | Stochastic Volatility / Advanced usage (Tests of the advanced-usage snippets: custom resamplers, custom filter) |

Experiments always run in the canonical order listed above, regardless of the order
given on the command line, and duplicates are removed.

## Methods / models

Selected with `-m` / `--models`. Names are matched against the methods each experiment
supports; a requested name that does not apply to the current experiment is reported
and skipped rather than being an error. If none of the requested names apply, that
experiment is skipped entirely.

| Experiment | Available `--models` values |
|---|---|
| `kalman` | `25`, `100`, `1000`, `10000` (particle counts) |
| `proposal` | `Bootstrap`, `Optimal`, plus the six DPF methods below |
| `sv_filtering` | the six DPF methods |
| `sv_single` | the six DPF methods |
| `sv_multiple` | the six DPF methods |
| `maze` | the six DPF methods |
| `example_usage` | *(no method selection)* |
| `advanced_usage` | *(no method selection)* |

The six DPF methods are:

```
DPF, Soft, Stop-Gradient, Marginal Stop-Gradient, Optimal Transport, Kernel
```

Names containing spaces or hyphens must be quoted, e.g.: `--models "Stop-Gradient" "Optimal Transport"`.

---

## Command line arguments

### Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `-e`, `--experiment` | one or more names | *all defaults* | Which experiment(s) to run. Choices are the eight experiment names plus `all`. |
| `-m`, `--models` | one or more names | *every method* | Restrict to these methods/models. |

### Paths and execution

| Argument              | Type | Default | Description                                                                                                                                       |
|-----------------------|---|---|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `--device`            | string | `auto` | `auto`, `cpu`, `cuda:0`, … With `auto`, CUDA is used if available, otherwise CPU with a warning.                                                  |
| `--data-dir`          | path | `./data` | Central data folder. Created if missing.                                                                                                          |
| `--results-dir`       | path | `./results` | Central results folder. Created if missing.                                                                                                       |
| `--smoke`             | flag | off | Tiny epochs / repeats / particle counts to validate the pipeline quickly. **Results are not paper-accurate.** Output files get a `_smoke` suffix. |
| `--setup-only`        | flag | off | Only prepare the data needed by the selected experiments, then exit without running anything.                                                     |
| `--overwrite-results` | flag | off | By default experiments that have complete results already stored are skipped, this flag turns off this behaviour..                                |
 | `--clear-results`     | flag | off | Clear all results files for the given experiments before re-populating them. |

### Data-generation parameters

All data-generation specific arguments default to the values used in the paper's experiments.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dx` | int | `25` | Linear-Gaussian state dimension. |
| `--dy` | int | `1` | Linear-Gaussian observation dimension. |
| `--alpha` | float | `0.91` | Stochastic-volatility generation α. |
| `--beta` | float | `0.5` | Stochastic-volatility generation β. |
| `--sigma` | float | `1.0` | Stochastic-volatility generation σ. |
| `--batch-size` | int | `128` | Generation / evaluation batch size. |

### Maze-specific

All maze specific arguments default to the values used in the paper's experiments.

| Argument               | Type                                        | Default | Description                                                                                                                                            |
|------------------------|---------------------------------------------|---|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--maze-deterministic` | `deterministic`, `nondeterministic`, `both` | `both` | Which maze run(s) to perform. `both` runs each in turn and writes two result files.                                                                    |
| `--maze-repeats`       | int                                         | `5` | Number of repeats, averaged. Forced to `1` under `--smoke`.                                                                                            |
| `--keep-raw`           | flag                                        | off | Keep the raw maze `.npz` archives after building `maze_data.csv`, only effective if the processed maze data is not already present in the data folder. |

---

## Data preparation

For each required file the script checks, in order: the central `data/` folder; then a
legacy copy in the original per-experiment sub-folder (copied across if found); then
regeneration or download.

| File | Used by | How it is obtained                                                                                                                                                                       |
|---|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dx={dx}-dy={dy}.csv` | `kalman`, `proposal` | Simulated each run (2000 trajectories; 50 under `--smoke`), 1000 time steps                                                                                                              |
| `alpha={a}-beta={b}-sigma={s}.csv` | `sv_filtering` | Simulated each run (500 trajectories; 50 under `--smoke`), 1000 time steps                                                                                                                       |
| `test_trajectory.csv` | `sv_single` | **Cannot be generated.** Must be present.                                                                                                                                                |
| `maze_data.csv` | `maze` | Built from `maze_data_raw.npz` / `maze_data_raw_2.npz`, which are downloaded automatically if absent, alternatively if can be obtained directly from https://zenodo.org/records/23039951 |
| `example_usage.csv` | `example_usage`, `advanced_usage` | Regenerated on every run                                                                                                                                                                 |

> **`test_trajectory.csv` is the one manual step.** If it is missing, `sv_single` fails
> with a `FileNotFoundError`. Download it from the PyDPF repository 
> (<https://github.com/John-JoB/pydpf>) if it is not present and place it in the data folder.

The maze raw data is fetched from `depositonce.tu-berlin.de` and unpacked if
the processed data is not already present; this
requires network access and a fair amount of disk space. Use `--setup-only` to do all
downloading and preprocessing ahead of time:

```bash
python run_experiments.py --setup-only
```

---

## Output

Written to `--results-dir`. Each CSV is indexed by `method`. Under `--smoke` every
filename gains a `_smoke` suffix.

| File | Experiment | Columns |
|---|---|---|
| `Kalman_comparison_results.csv` | `kalman` | `Time CPU (s)`, `Time GPU (s)`, `epsilon x`, `epsilon y` |
| `proposal_learning_results.csv` | `proposal` | `e_x`, `e_l`, `mean W2`, `ELBO` |
| `fully_specified_results.csv` | `sv_filtering` | `e_x`, `e_l`, `time` |
| `single_parameter_results.csv` | `sv_single` | `Forward Time (s)`, `Backward Time (s)`, `Gradient standard deviation`, `alpha error` |
| `multiple_parameter_results.csv` | `sv_multiple` | `ELBO`, `alpha error`, `beta error`, `sigma error` |
| `deep_mind_maze_results.csv` | `maze` (deterministic) | `Total time (hrs:min:s)`, `Test MSE` |
| `nondeterministic_deep_mind_maze_results.csv` | `maze` (non-deterministic) | `Total time (hrs:min:s)`, `Test MSE` |
| `example_usage_parameter_errors_.pdf` | `example_usage` | convergence plot |
| `<filter name>_parameter_errors_.pdf` | `advanced_usage` | one plot per snippet |

If the `kalman` experiment is run with `--device cpu` the 

Results CSVs are filled in place, so an interrupted run can be resumed by re-running
with the same arguments: completed rows are preserved and the corresponding experiment
not re-run unless `--overwrite-results` or `--clear-results` is passed. 

`--overwrite-results` only overwrites the results corresponding to models that the user asks for,
`--clear-results` wipes all rows of the results table for every experiment asked for.

---

## Examples

```bash
# Prepare all data and exit
python run_experiments.py --setup-only

# Fast end-to-end check of everything on CPU
python run_experiments.py --smoke --device cpu

# Run a single algorithm on a single experiment
python run_experiments.py -e sv_filtering  -m "Soft"

# Maze, deterministic only, single repeat, on the second GPU
python run_experiments.py -e maze --maze-deterministic deterministic \
    --maze-repeats 1 --device cuda:1

# Regenerate the linear-Gaussian results from scratch at with 10 dimensional state
python run_experiments.py -e kalman proposal --dx 10 --overwrite-results
```