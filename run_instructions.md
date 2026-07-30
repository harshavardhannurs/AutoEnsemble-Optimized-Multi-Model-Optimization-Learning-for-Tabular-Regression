# Running the Project

## 1. Create the Conda Environment

From the project root, create the required Conda environment:

```bash
conda env create -f environment.yml
```

---

## 2. Generate Predictions Using the Bundled Optimization Results

To generate predictions for a dataset using the optimization results included with the project, run:

```bash
python run.py --task <dataset_name>
```

### Required Argument

| Argument | Description |
|----------|-------------|
| `--task` | Name of the dataset to generate predictions for. **Required.** |

The project already includes previously generated optimization results located in one or more of the following directories:

- `outputs/hpo`
- `outputs/multi-fidelity`

When the command above is executed, the code automatically searches these directories for existing optimization results and uses them to generate predictions.

---

## 3. Generate Predictions Using Your Own Optimization Results

If you would like to perform hyperparameter optimization yourself before generating predictions, run:

```bash
python run.py \
    --task <dataset_name> \
    --seed <seed> \
    --output-path <path_to_save_predictions> \
    --hpo <facade_of_choice> \
    --hpo-trials <number_of_trials> \
    --hpo-workers <number_of_workers> \
    --hpo-timeout <timeout_in_seconds> \
    --delta <delta> \
    --temperature <temperature>
```

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|:--------:|:-------:|-------------|
| `--task` | ✅ | — | Dataset name for which predictions should be generated. |
| `--seed` | ❌ | `42` | Random seed used to ensure reproducibility. |
| `--output-path` | ❌ | `./predictions.npy` | Path where the generated predictions will be saved. |
| `--hpo` | ❌ | `off` | Hyperparameter optimization method (facade) to use. |
| `--hpo-trials` | ❌ | `200` | Number of optimization trials. |
| `--hpo-workers` | ❌ | `1` | Number of parallel workers used by the optimizer. |
| `--hpo-timeout` | ❌ | `86400` | Maximum optimization time (in seconds). |
| `--delta` | ❌ | `0.003` | Acceptable performance difference used during model selection. |
| `--temperature` | ❌ | `0.01` | Controls how strongly better-performing models are weighted in the final ensemble. Lower values give more weight to the best models. |

---

## Behavior of `--hpo`

The `--hpo` argument determines how the project obtains the model used for prediction.

- **`off` (default)**
  1. Attempts to load previously generated optimization results.
  2. If no optimization results are found, trains a baseline model using default hyperparameters.

- **Any supported optimizer**
  - Runs the specified hyperparameter optimization method.
  - Uses the resulting optimized model(s) to generate predictions.

---

## Minimal Example

To run the project using the bundled optimization results:

```bash
python run.py --task <dataset_name>
```

To run hyperparameter optimization before generating predictions:

```bash
python run.py --task <dataset_name> --hpo <facade_of_choice>
```