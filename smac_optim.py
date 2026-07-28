import os
import time
from pathlib import Path
from functools import partial

from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, EqualsCondition

from smac import HyperparameterOptimizationFacade as HPOFacade, Scenario, MultiFidelityFacade as MFFacade

import models.ffn as ffn
import models.randomforest as rf
import models.xgb as xgb
import models.pfn as pfn
import models.extratrees as et
import models.hgb as hgb

import utils.model_utils as mu
import utils.optim_utils as ou

# Hyperparameters

model = Categorical('model', ['ffn', 'xgb', 'randomforest', 'pfn', 'extratrees', 'hgb'], default='ffn')

batch_size = Categorical('batch_size', [32, 64, 128, 256],  default=64)
no_of_layers = Integer('no_of_layers', (1, 4), default=2)
no_of_units = Categorical('no_of_units', [32, 64, 128, 256, 512], default=32)
learning_rate_ffn = Float('learning_rate_ffn', (1e-5, 1e-1), default=1e-3, log=True)
weight_decay = Float('weight_decay', (1e-6, 1e-2), default=1e-4, log=False)
batch_norm = Categorical('batch_norm', [True, False], default=False)
num_epochs = Categorical('num_epochs', [50, 100, 200, 300, 500], default=50)

n_estimators_xgb = Categorical('n_estimators_xgb', [50, 100, 200, 300, 500, 800, 1000], default=100)
learning_rate_xgb = Float('learning_rate_xgb', (1e-3, 0.3), default=0.1, log=True)
max_depth_xgb = Integer('max_depth_xgb', (3, 8), default=6)
subsample = Float('subsample', (0.7, 1), default=0.7)
colsample_bytree = Float('colsample_bytree',(0.5,1.0),default=0.8)
reg_lambda = Float('reg_lambda', (1e-3, 100), default=1e-3, log=True)

n_estimators_rf = Categorical('n_estimators_rf', [50, 100, 200, 500], default=100)
max_depth_rf = Categorical('max_depth_rf', ['None', 5, 10, 20, 30, 50], default=5)
min_samples_leaf_rf = Integer('min_samples_leaf_rf', (1, 20), default=1)
max_features_rf = Categorical('max_features_rf', [0.3, 0.5, 0.7, 1.0], default=1.0)

n_estimators_et = Categorical('n_estimators_et', [50, 100, 200, 500], default=100)
max_depth_et = Categorical('max_depth_et', ['None', 5, 10, 20, 30, 50], default=5)
min_samples_leaf_et = Integer('min_samples_leaf_et', (1, 20), default=1)
max_features_et = Categorical('max_features_et', [0.3, 0.5, 0.7, 1.0], default=1.0)

max_iter_hgb = Categorical('max_iter_hgb', [100, 200, 300, 500, 800, 1000], default=300)
learning_rate_hgb = Float('learning_rate_hgb', (1e-2, 0.3), default=0.1, log=True)
max_leaf_nodes_hgb = Integer('max_leaf_nodes_hgb', (15, 255), default=31)
min_samples_leaf_hgb = Integer('min_samples_leaf_hgb', (5, 100), default=20)
l2_reg_hgb = Float('l2_reg_hgb', (1e-3, 100), default=1e-3, log=True)
max_features_hgb = Categorical('max_features_hgb', [0.3, 0.5, 0.7, 1.0], default=1.0)
max_bins_hgb = Categorical('max_bins_hgb', [64, 128, 255], default=255)

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
cond_rf_2 = EqualsCondition(min_samples_leaf_rf, model, 'randomforest')
cond_rf_3 = EqualsCondition(max_features_rf, model, 'randomforest')
cond_rf_4 = EqualsCondition(n_estimators_rf, model, 'randomforest')

cond_et_1 = EqualsCondition(max_depth_et, model, 'extratrees')
cond_et_2 = EqualsCondition(min_samples_leaf_et, model, 'extratrees')
cond_et_3 = EqualsCondition(max_features_et, model, 'extratrees')
cond_et_4 = EqualsCondition(n_estimators_et, model, 'extratrees')

cond_hgb_1 = EqualsCondition(max_iter_hgb, model, 'hgb')
cond_hgb_2 = EqualsCondition(learning_rate_hgb, model, 'hgb')
cond_hgb_3 = EqualsCondition(max_leaf_nodes_hgb, model, 'hgb')
cond_hgb_4 = EqualsCondition(min_samples_leaf_hgb, model, 'hgb')
cond_hgb_5 = EqualsCondition(l2_reg_hgb, model, 'hgb')
cond_hgb_6 = EqualsCondition(max_features_hgb, model, 'hgb')
cond_hgb_7 = EqualsCondition(max_bins_hgb, model, 'hgb')

# Create Configuration Space
configspace = ConfigurationSpace()

configspace.add([
    model,
    batch_size, no_of_layers, no_of_units, learning_rate_ffn, weight_decay, batch_norm, num_epochs,
    n_estimators_xgb, learning_rate_xgb, max_depth_xgb, subsample, colsample_bytree, reg_lambda,
    n_estimators_rf, max_depth_rf, min_samples_leaf_rf, max_features_rf,
    n_estimators_et, max_depth_et, min_samples_leaf_et, max_features_et,
    max_iter_hgb, learning_rate_hgb, max_leaf_nodes_hgb, min_samples_leaf_hgb, l2_reg_hgb, max_features_hgb, max_bins_hgb
])

configspace.add([
    cond_ffn_1, cond_ffn_2, cond_ffn_3, cond_ffn_4, cond_ffn_5, cond_ffn_6, cond_ffn_7,
    cond_xgb_1, cond_xgb_2, cond_xgb_3, cond_xgb_4, cond_xgb_5, cond_xgb_6,
    cond_rf_1, cond_rf_2, cond_rf_3, cond_rf_4,
    cond_et_1, cond_et_2, cond_et_3, cond_et_4,
    cond_hgb_1, cond_hgb_2, cond_hgb_3, cond_hgb_4, cond_hgb_5, cond_hgb_6, cond_hgb_7
])

def hpo_train(config, seed, dataset_name=None):

    model = config['model']

    print(
        f"[START][PID {os.getpid()}] Evaluating model={mu.MODELS[model]}",
        flush=True
    )

    score = 0

    if model == 'ffn':
        r2_score_across_folds = ffn.pipelined_model(
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
            max_depth=(None if config['max_depth_rf'] == 'None' else config['max_depth_rf']),
            min_samples_leaf=config['min_samples_leaf_rf'],
            max_features=float(config['max_features_rf']),
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'extratrees':
        r2_score_across_folds = et.pipelined_model(
            dataset_name=dataset_name,
            n_estimators=int(config['n_estimators_et']),
            max_depth=(None if config['max_depth_et'] == 'None' else config['max_depth_et']),
            min_samples_leaf=config['min_samples_leaf_et'],
            max_features=float(config['max_features_et']),
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'hgb':
        r2_score_across_folds = hgb.pipelined_model(
            dataset_name=dataset_name,
            max_iter=int(config['max_iter_hgb']),
            learning_rate=config['learning_rate_hgb'],
            max_leaf_nodes=config['max_leaf_nodes_hgb'],
            min_samples_leaf=config['min_samples_leaf_hgb'],
            l2_reg=config['l2_reg_hgb'],
            max_features=float(config['max_features_hgb']),
            max_bins=config['max_bins_hgb'],
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'pfn':
        _, __, ___, r2_score = pfn.train(
            dataset_name=dataset_name,
            seed=seed,
        )

        score = r2_score

    print(
        f"[RESULT][PID {os.getpid()}] "
        f"Model={mu.MODELS[model]}, "
        f"R2={score:.4f}\n"
        f"Configuration: {config}",
        flush=True
    )

    return 1 - score

def run_hp_optimizer(dataset_name, n_trials, n_workers, walltime_limit):

    train_function = partial(
        hpo_train,
        dataset_name=dataset_name
    )

    # Create Scenario
    scenario = Scenario(
        configspace=configspace,
        name=f'smac_hp_optim_{dataset_name}',
        output_directory=Path(__file__).resolve().parent / 'outputs' / 'hpo',
        n_trials=n_trials,
        n_workers=n_workers,
        walltime_limit=walltime_limit,
        deterministic=True,
        # SMAC evaluates incumbent contender config multiple times across different seeds to check if the score returned first time wasn't a fluke. With deterministic=True, we are avoiding evaluating the same configuration multiple times across multiple seeds since our target function returns CV score, and it already averages out the noise.
    )

    smac = HPOFacade(scenario=scenario, target_function=train_function)

    start_time = time.perf_counter()
    incumbent = smac.optimize()
    time_taken = time.perf_counter() - start_time

    path = Path(__file__).resolve().parent / 'outputs' / 'hpo'
    ou.log_execution_time(dataset_name, time_taken, path)

    return smac, incumbent

def mfo_train(config, seed, budget, dataset_name=None):

    model = config['model']

    print(
        f"[START][PID {os.getpid()}] Evaluating model={mu.MODELS[model]}",
        flush=True
    )

    score = 0

    if model == 'ffn':
        r2_score_across_folds = ffn.pipelined_model(
            dataset_name=dataset_name,
            batch_size=int(config['batch_size']),
            no_of_layers=config['no_of_layers'],
            no_of_units=int(config['no_of_units']),
            learning_rate=config['learning_rate_ffn'],
            weight_decay=config['weight_decay'],
            batch_norm=config['batch_norm'],
            num_epochs=int(budget),
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'xgb':
        r2_score_across_folds = xgb.pipelined_model(
            dataset_name=dataset_name,
            n_estimators=int(budget),
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
            n_estimators=int(budget),
            max_depth=(None if config['max_depth_rf'] == 'None' else config['max_depth_rf']),
            min_samples_leaf=config['min_samples_leaf_rf'],
            max_features=float(config['max_features_rf']),
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'extratrees':
        r2_score_across_folds = et.pipelined_model(
            dataset_name=dataset_name,
            n_estimators=int(budget),
            max_depth=(None if config['max_depth_et'] == 'None' else config['max_depth_et']),
            min_samples_leaf=config['min_samples_leaf_et'],
            max_features=float(config['max_features_et']),
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'hgb':
        r2_score_across_folds = hgb.pipelined_model(
            dataset_name=dataset_name,
            max_iter=int(budget),
            learning_rate=config['learning_rate_hgb'],
            max_leaf_nodes=config['max_leaf_nodes_hgb'],
            min_samples_leaf=config['min_samples_leaf_hgb'],
            l2_reg=config['l2_reg_hgb'],
            max_features=float(config['max_features_hgb']),
            max_bins=config['max_bins_hgb'],
            seed=seed
        )

        score = mu.average(r2_score_across_folds)

    elif model == 'pfn':
        _, __, ___, r2_score = pfn.train(
            dataset_name=dataset_name,
            seed=seed,
        )

        score = r2_score

    print(
        f"[RESULT][PID {os.getpid()}] "
        f"Model={mu.MODELS[model]}, "
        f"R2={score:.4f}\n"
        f"Configuration: {config}",
        flush=True
    )

    return 1 - score

def run_mf_optimizer(dataset_name, n_trials, n_workers, walltime_limit):

    train_function = partial(
        mfo_train,
        dataset_name=dataset_name
    )

    scenario = Scenario(
        configspace=configspace,
        name=f'smac_mf_optim_{dataset_name}',
        output_directory=Path(__file__).resolve().parent / 'outputs' / 'multi-fidelity',
        n_trials=n_trials,
        n_workers=n_workers,
        walltime_limit=walltime_limit,
        deterministic=True,
        min_budget=50,
        max_budget=1000
    )

    smac = MFFacade(scenario=scenario, target_function=train_function)

    start_time = time.perf_counter()
    incumbent = smac.optimize()
    time_taken = time.perf_counter() - start_time

    path = Path(__file__).resolve().parent / 'outputs' / 'multi-fidelity'
    ou.log_execution_time(dataset_name, time_taken, path)
    return smac, incumbent


