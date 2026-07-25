## Running individual model files

To be able to run the files in **models** directory, change the directory to project root and run the following in the terminal:

`python -m models.<file_name>`

When you run the above command from your **automl_final** root directory, Python adds **automl_final** to *sys.path*. Because **automl_final** is now in *sys.path*, Python can easily resolve top-level imports like import utils.data_utils or import utils.model_utils.

## Exporting SMAC results

`smac_optim.py` automatically creates `smac_trials.csv` in the dataset's SMAC
output directory after optimization and validation finish.

You can also export a completed or still-running SMAC output manually:

```bash
python output_visual.py outputs/smac_optim_bike_sharing_demand
```

The input can be an individual run directory, a `runhistory.json` file, or a
parent directory containing multiple SMAC runs. Use `--output PATH` to choose a
different CSV destination.
