# Running the Project

## 1. Create a Local Python Environment

Choose either Conda or Python's built-in `venv`. Run all commands from the
project root.

### Option A: Conda

Create and activate the environment defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate automl_final
```

### Option B: Python `venv`

Python 3.12 is recommended because it matches the version used by the Conda
environment.

Create the virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install the project dependencies from `requirements.txt`:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Download the Local TabPFN Model

The project loads TabPFN from a local checkpoint instead of downloading the
model while a run is in progress. From the project root, use the command for
your operating system to download the official TabPFN v3 regressor checkpoint.

#### Windows PowerShell

```powershell
curl.exe -L --fail `
  "https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-regressor-v3_default.ckpt" `
  --output "tabpfn-v3-regressor-v3_default.ckpt"
```

#### Windows Command Prompt (CMD)

```cmd
curl.exe -L --fail ^
  "https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-regressor-v3_default.ckpt" ^
  --output "tabpfn-v3-regressor-v3_default.ckpt"
```

#### Linux

```bash
curl -L --fail \
  "https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-regressor-v3_default.ckpt" \
  --output tabpfn-v3-regressor-v3_default.ckpt
```

#### macOS

```bash
curl -L --fail \
  "https://huggingface.co/Prior-Labs/tabpfn_3/resolve/main/tabpfn-v3-regressor-v3_default.ckpt" \
  --output tabpfn-v3-regressor-v3_default.ckpt
```

Confirm that the checkpoint exists on Linux or macOS:

```bash
ls -lh tabpfn-v3-regressor-v3_default.ckpt
```

On Windows PowerShell, verify it with:

```powershell
Get-Item "tabpfn-v3-regressor-v3_default.ckpt"
```

Keep this file in the project root. To store it elsewhere, set its absolute
path before running the project:

```bash
export AUTOML_TABPFN_MODEL_PATH="/absolute/path/to/tabpfn-v3-regressor-v3_default.ckpt"
```

The checkpoint is intentionally excluded from Git because it is a large binary
file. Once it is downloaded, TabPFN can load it without an internet connection.
The project configures TabPFN to run on the CPU; CUDA is not required.


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
    --hpo-timeout <timeout_in_seconds> 
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
