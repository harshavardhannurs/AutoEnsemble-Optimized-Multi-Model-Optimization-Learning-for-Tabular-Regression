import os
from pathlib import Path
from functools import partial

from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, EqualsCondition

from smac import HyperparameterOptimizationFacade as HPOFacade, Scenario

import models.ffn as ffn
import models.randomforest as rf
import models.xgb as xgb
import models.pfn as pfn

import utils.model_utils as mu
from output_visual import export_smac_results

MODELS = {'ffn': 'Feed Forward Neural Network', 'xgb': 'XGBoost', 'randomforest': 'Random Forest', 'pfn': 'TabPFN'}

DATASET_DICT = {1: 'bike_sharing_demand', 2: 'brazilian_houses', 3: 'superconductivity', 4: 'wine_quality', 5: 'yprop_4_1'}

# Hyperparameters

model = Categorical('model', ['ffn', 'xgb', 'randomforest', 'pfn'], default='ffn')

batch_size = Categorical('batch_size', [32, 64, 128, 256],  default=64)
no_of_layers = Integer('no_of_layers', (1, 4), default=2)
no_of_units = Categorical('no_of_units', [32, 64, 128, 256, 512], default=32)
learning_rate_ffn = Float('learning_rate_ffn', (1e-5, 1e-1), default=1e-3, log=True)
weight_decay = Float('weight_decay', (1e-6, 1e-2), default=1e-4, log=False)
batch_norm = Categorical('batch_norm', [True, False], default=False)
num_epochs = Categorical('num_epochs', [50, 100, 200, 300, 500], default=50)

n_estimators_xgb = Categorical('n_estimators_xgb', [50, 100, 200, 500], default=100)
learning_rate_xgb = Float('learning_rate_xgb', (1e-3, 0.3), default=0.1, log=True)
max_depth_xgb = Integer('max_depth_xgb', (2, 12), default=3)
subsample = Float('subsample', (0.5, 1), default=0.5)
colsample_bytree = Float('colsample_bytree',(0.5,1.0),default=0.8)
reg_lambda = Float('reg_lambda', (1e-3, 100), default=1e-3, log=True)

n_estimators_rf = Categorical('n_estimators_rf', [50, 100, 200, 500], default=100)
max_depth_rf = Categorical('max_depth_rf', [5, 10, 20, 30, 50], default=5)
min_samples_leaf = Integer('min_samples_leaf', (1, 20), default=1)
max_features = Float('max_features', (0.5, 1.0), default=1.0)

n_estimators_pfn = Categorical('n_estimators_pfn', [8, 16, 32, 64, 128, 256], default=32)

# Conditions
cond_ffn_1 = EqualsCondition(batch_size, model, 'ffn')
cond_ffn_2 = EqualsCondition(no_of_layers, model, 'ffn')
cond_ffn_3 = EqualsCondition(no_of_units, model, 'ffn')
cond_ffn_4 = EqualsCondition(learning_rate_ffn, model, 'ffn')
cond_ffn_5 = EqualsCondition(weight_decay, model, 'ffn')
cond_ffn_6 = EqualsCondition(batch_norm, model, 'ffn')
cond_ffn_7 = EqualsCondition(num_epochs, model, 'ffn')

cond_xgb_1 = EqualsCondition(learning_rate_xgb, model, 'xgb')
cond_xgb_2 = EqualsCondition(max_depth_xgb, model, 'xgb')
cond_xgb_3 = EqualsCondition(subsample, model, 'xgb')
cond_xgb_4 = EqualsCondition(colsample_bytree, model, 'xgb')
cond_xgb_5 = EqualsCondition(reg_lambda, model, 'xgb')
cond_xgb_6 = EqualsCondition(n_estimators_xgb, model, 'xgb')

cond_rf_1 = EqualsCondition(max_depth_rf, model, 'randomforest')
cond_rf_2 = EqualsCondition(min_samples_leaf, model, 'randomforest')
cond_rf_3 = EqualsCondition(max_features, model, 'randomforest')
cond_rf_4 = EqualsCondition(n_estimators_rf, model, 'randomforest')

cond_pfn_1 = EqualsCondition(n_estimators_pfn, model, 'pfn')

# Create Configuration Space
configspace = ConfigurationSpace()

configspace.add([
    model,
    batch_size, no_of_layers, no_of_units, learning_rate_ffn, weight_decay, batch_norm, num_epochs,
    n_estimators_xgb, learning_rate_xgb, max_depth_xgb, subsample, colsample_bytree, reg_lambda,
    n_estimators_rf, max_depth_rf, min_samples_leaf, max_features,
    n_estimators_pfn
])

configspace.add([
    cond_ffn_1, cond_ffn_2, cond_ffn_3, cond_ffn_4, cond_ffn_5, cond_ffn_6, cond_ffn_7,
    cond_xgb_1, cond_xgb_2, cond_xgb_3, cond_xgb_4, cond_xgb_5, cond_xgb_6,
    cond_rf_1, cond_rf_2, cond_rf_3, cond_rf_4,
    cond_pfn_1
])

def train(config, seed, dataset_name=None):

    model = config['model']

    print(
        f"[START][PID {os.getpid()}] Evaluating model={MODELS[model]}",
        flush=True
    )

    score = 0

    if model == 'ffn':
        _, __, r2_score_across_folds = ffn.pipelined_model(
            dataset_name=dataset_name,
            batch_size=int(config['batch_size']),
            no_of_layers=config['no_of_layers'],
            no_of_units=int(config['no_of_units']),
            learning_rate=config['learning_rate_ffn'],
            weight_decay=config['weight_decay'],
            batch_norm=config['batch_norm'],
            num_epochs=int(config['num_epochs']),
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'xgb':
        r2_score_across_folds = xgb.pipelined_model(
            dataset_name=dataset_name,
            n_estimators=int(config['n_estimators_xgb']),
            learning_rate=config['learning_rate_xgb'],
            max_depth=config['max_depth_xgb'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            reg_lambda=config['reg_lambda'],
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'randomforest':
        r2_score_across_folds = rf.pipelined_model(
            dataset_name=dataset_name,
            n_estimators=int(config['n_estimators_rf']),
            max_depth=config['max_depth_rf'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'],
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'pfn':
        r2_score_across_folds = pfn.pipelined_model(
            dataset_name=dataset_name,
            n_estimators=int(config['n_estimators_pfn']),
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    print(
        f"[RESULT][PID {os.getpid()}] "
        f"Model={MODELS[model]}, "
        f"R2={score:.4f}\n"
        f"Configuration: {config}",
        flush=True
    )

    return 1 - score


def main(dataset_name, n_trials=20, n_workers=1):

    train_function = partial(
        train,
        dataset_name=dataset_name
    )

    # Create Scenario
    scenario = Scenario(
        configspace=configspace,
        name=f'smac_optim_{dataset_name}',
        output_directory=Path(__file__).resolve().parent / 'outputs',
        # Each target-function call already averages performance over 10
        # cross-validation folds. Treat that aggregate as deterministic so
        # every candidate needs only one completed evaluation to compete for
        # incumbent status, regardless of model runtime.
        deterministic=True,
        walltime_limit=86400,
        n_trials=n_trials,
        n_workers=n_workers
    )

    smac = HPOFacade(scenario=scenario, target_function=train_function)

    incumbent = smac.optimize()

    print("Calculating the cost of the default configuration...", flush=True)
    default_cost = smac.validate(configspace.get_default_configuration())
    # smac.validate() evaluates the config, by default, across three random seeds and averages the cost across those three runs.

    print("Calculating the cost of the incumbent configuration...", flush=True)
    incumbent_cost = smac.validate(incumbent)

    try:
        csv_path = export_smac_results(
            Path(__file__).resolve().parent / 'outputs' / f'smac_optim_{dataset_name}'
        )
        print(f"SMAC trial results exported to: {csv_path}", flush=True)
    except (FileNotFoundError, OSError, ValueError) as exc:
        # The optimization results remain valid even if the convenience export
        # cannot be written (for example, because the destination is read-only).
        print(f"[WARNING] Could not export SMAC trial results: {exc}", flush=True)

    return incumbent, default_cost, incumbent_cost

def get_user_dataset_selection():

    print("The datasets available:", flush=True)
    for key, name in DATASET_DICT.items():
        print(f"  {key}. {name}", flush=True)

    while True:
        user_input = input("\nEnter the number corresponding to the dataset or \"exit\" to exit: ").strip()

        if user_input.isdigit():
            choice = int(user_input)
            if choice in DATASET_DICT:
                print("Dataset selected: ", DATASET_DICT[choice], flush=True)
                return DATASET_DICT[choice]
        if user_input.lower() == 'exit':
            print("\nExiting...", flush=True)
            return None

        print(f"Invalid input. Please enter a number between 1 and {len(DATASET_DICT)}.", flush=True)

def get_user_no_trials_selection():

    while True:
        user_input = input("\nEnter the number of trials you want SMAC to run or \"exit\" to exit: ").strip()

        if user_input.isdigit():
            choice = int(user_input)
            if choice > 0:
                print("No. of trials selected: ", choice, flush=True)
                return choice
        if user_input.lower() == 'exit':
            print("\nExiting...", flush=True)
            return None
        print("Invalid input. Please enter a positive number.", flush=True)

def get_user_no_workers_selection():
    while True:
        user_input = input("\nEnter the number of workers you want SMAC to use to run the optimization or \"exit\" to exit: ").strip()

        if user_input.isdigit():
            choice = int(user_input)
            if choice > 0:
                print("No. of workers selected: ", choice, flush=True)
                return choice
        if user_input.lower() == 'exit':
            print("\nExiting...", flush=True)
            return None
        print("Invalid input. Please enter a positive number.", flush=True)

if __name__ == '__main__':

    dataset_name = get_user_dataset_selection()
    n_trials = get_user_no_trials_selection()
    n_workers = get_user_no_workers_selection()

    if dataset_name and n_trials and n_workers:

        incumbent, default_cost, incumbent_cost = main(dataset_name, n_trials=n_trials, n_workers=n_workers)

        print(f"Incumbent configuration: {incumbent}", flush=True)
        print(f"R2 Score of baseline (default) configuration: {1-default_cost:.4f}", flush=True)
        print(f"R2 Score of incumbent configuration: {1-incumbent_cost:.4f}", flush=True)
