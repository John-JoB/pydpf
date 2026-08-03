# PyDPF run_experiments script documentation



# Running the validation script

The replication script `run_experiments.py` requires both matplotlib and tqdm so install these before proceeding.

Pydpf is a package designed for the implementation and comparison of deep learning algorithms, consequently many of the experiments take
a long time to run. Therefore, we provide command line arguments that allow a user to run each experiment separately or a smoke
test which runs through almost all lines of code but at lower particle counts, training iterations and experiment repeats.


Data is read from a single central folder (default: `data/`) and every result is written to a
single central folder (default: `results/`). Both are created and populated on demand. The Unsupervised learning of a single parameter
experiment, only, requires that the data folder contain the file `test_trajectory.csv` that is provided in the supplementary materials.


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

Names containing spaces or hyphens must be quoted: `--models "Stop-Gradient" "Optimal Transport"`.

---

## Command-line arguments

### Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `-e`, `--experiment` | one or more names | *all defaults* | Which experiment(s) to run. Choices are the eight experiment names plus `all`. |
| `-m`, `--models` | one or more names | *every method* | Restrict to these methods/models. |

### Paths and execution

| Argument | Type | Default | Description |
|---|---|---|---|
| `--device` | string | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, … With `auto`, CUDA is used if available, otherwise CPU with a warning. |
| `--data-dir` | path | `./data` | Central data folder. Created if missing. |
| `--results-dir` | path | `./results` | Central results folder. Created if missing. |
| `--smoke` | flag | off | Tiny epochs / repeats / particle counts to validate the pipeline quickly. **Results are not paper-accurate.** Output files get a `_smoke` suffix. |
| `--setup-only` | flag | off | Only prepare the data needed by the selected experiments, then exit without running anything. |
| `--overwrite-results` | flag | off | Recreate (blank out) the results CSVs of the selected experiments even if they already exist. Without it, existing CSVs are kept and filled in place. |

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

| Argument | Type | Default | Description |
|---|---|---|---|
| `--maze-deterministic` | `deterministic` \| `nondeterministic` \| `both` | `both` | Which maze run(s) to perform. `both` runs each in turn and writes two result files. |
| `--maze-repeats` | int | `5` | Number of repeats, averaged. Forced to `1` under `--smoke`. |
| `--delete-raw` | flag | off | Delete the raw maze `.npz` archives after building `maze_data.csv`. |

---

## Data preparation

For each required file the script checks, in order: the central `data/` folder; then a
legacy copy in the original per-experiment sub-folder (copied across if found); then
regeneration or download.

| File | Used by | How it is obtained |
|---|---|---|
| `dx={dx}-dy={dy}.csv` | `kalman`, `proposal` | Simulated (2000 trajectories; 50 under `--smoke`), 1000 time steps |
| `alpha={a}-beta={b}-sigma={s}.csv` | `sv_filtering` | Simulated (500 trajectories; 50 under `--smoke`), 1000 time steps |
| `test_trajectory.csv` | `sv_single` | **Cannot be generated.** Must be present. |
| `maze_data.csv` | `maze` | Built from `maze_data_raw.npz` / `maze_data_raw_2.npz`, which are downloaded automatically if absent |
| `example_usage.csv` | `example_usage`, `advanced_usage` | Regenerated on every run |

> **`test_trajectory.csv` is the one manual step.** If it is missing, `sv_single` fails
> with a `FileNotFoundError`. Download it from the pydpf repository
> (<https://github.com/John-JoB/pydpf>) and place it in the data folder.

The maze raw data is ~fetched from `depositonce.tu-berlin.de` and unpacked; this
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

On a CUDA device the `kalman` experiment runs twice — once on GPU and once on CPU — so
that both timing columns are filled from a single invocation.

Results CSVs are filled in place, so an interrupted run can be resumed by re-running
with the same arguments: completed rows are preserved unless `--overwrite-results` is
passed.

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