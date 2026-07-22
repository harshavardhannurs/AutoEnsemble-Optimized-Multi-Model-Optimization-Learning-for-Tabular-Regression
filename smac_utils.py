"""SMAC model selection and hyperparameter optimization."""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    Integer,
)
from smac import HyperparameterOptimizationFacade, Scenario

from models.ensemble import ensemble_pipelined_model
from models.extra_trees import extra_trees_pipelined_model
from models.ffn import ffn_pipelined_model
from models.hgb import hgb_pipelined_model
from models.knn import knn_pipelined_model
from models.random_forest import random_forest_pipelined_model
from models.xgboost import xgboost_pipelined_model
from utils import model_utils as mu


ROOT = Path(__file__).resolve().parent


def config_to_dict(config):
    return {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in dict(config).items()
    }


class LiveOutput(io.TextIOBase):
    """Copy model output to the terminal and Streamlit."""

    def __init__(self, callback=None):
        self.callback = callback
        self.buffer = ""

    def write(self, text):
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        self.buffer += text

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip() and self.callback:
                self.callback(line)
        return len(text)

    def flush(self):
        sys.__stdout__.flush()


def get_configspace():
    """Create the conditional model space defined in hp_space.md."""
    space = ConfigurationSpace(seed=42)
    model = Categorical(
        "model",
        ["ffn", "knn", "random_forest", "extra_trees", "xgboost", "hgb"],
        default="ffn",
    )
    parameters = {
        "ffn": [
            Integer("ffn_no_of_layers", (1, 4), default=2),
            Integer("ffn_no_of_units", (32, 512), default=128),
            Float("ffn_learning_rate", (1e-5, 1e-2), default=1e-3, log=True),
            Integer("ffn_batch_size", (32, 256), default=64),
            Float("ffn_weight_decay", (1e-6, 1e-3), default=1e-4, log=True),
            Categorical("ffn_batch_norm", [False, True], default=False),
            Integer("ffn_num_epochs", (100, 1000), default=100),
        ],
        "knn": [
            Integer("knn_n_neighbors", (3, 50), default=5),
            Categorical("knn_weights", ["uniform", "distance"], default="uniform"),
            Categorical("knn_p", [1, 2], default=2),
        ],
        "random_forest": [
            Integer("rf_n_estimators", (100, 500), default=100),
            Integer("rf_max_depth", (5, 30), default=10),
            Integer("rf_min_samples_leaf", (1, 10), default=1),
            Float("rf_max_features", (0.5, 1.0), default=1.0),
        ],
        "extra_trees": [
            Integer("et_n_estimators", (100, 500), default=100),
            Integer("et_max_depth", (5, 30), default=10),
            Integer("et_min_samples_leaf", (1, 10), default=1),
            Float("et_max_features", (0.5, 1.0), default=1.0),
        ],
        "xgboost": [
            Integer("xgb_n_estimators", (100, 1000), default=100),
            Float("xgb_learning_rate", (0.01, 0.3), default=0.1, log=True),
            Integer("xgb_max_depth", (3, 10), default=6),
            Float("xgb_subsample", (0.5, 1.0), default=1.0),
            Float("xgb_colsample_bytree", (0.5, 1.0), default=1.0),
        ],
        "hgb": [
            Integer("hgb_max_iter", (100, 500), default=100),
            Float("hgb_learning_rate", (0.01, 0.3), default=0.1, log=True),
            Integer("hgb_max_leaf_nodes", (15, 63), default=31),
            Integer("hgb_min_samples_leaf", (10, 50), default=20),
            Float("hgb_l2_regularization", (1e-6, 1.0), default=1e-4, log=True),
        ],
    }

    all_parameters = [parameter for group in parameters.values() for parameter in group]
    space.add([model, *all_parameters])
    space.add(
        [
            EqualsCondition(parameter, model, model_name)
            for model_name, group in parameters.items()
            for parameter in group
        ]
    )
    return space


def run_configuration(config, dataset_name, n_splits):
    """Run the pipeline selected by a SMAC configuration."""
    config = config_to_dict(config)
    model = config["model"]

    if model == "ffn":
        result = ffn_pipelined_model(
            dataset_name=dataset_name,
            batch_size=config["ffn_batch_size"],
            no_of_layers=config["ffn_no_of_layers"],
            no_of_units=config["ffn_no_of_units"],
            learning_rate=config["ffn_learning_rate"],
            weight_decay=config["ffn_weight_decay"],
            batch_norm=config["ffn_batch_norm"],
            num_epochs=config["ffn_num_epochs"],
            n_splits=n_splits,
        )
    elif model == "knn":
        result = knn_pipelined_model(
            dataset_name=dataset_name,
            n_neighbors=config["knn_n_neighbors"],
            weights=config["knn_weights"],
            p=config["knn_p"],
            n_splits=n_splits,
        )
    elif model == "random_forest":
        result = random_forest_pipelined_model(
            dataset_name=dataset_name,
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            min_samples_leaf=config["rf_min_samples_leaf"],
            max_features=config["rf_max_features"],
            n_splits=n_splits,
        )
    elif model == "extra_trees":
        result = extra_trees_pipelined_model(
            dataset_name=dataset_name,
            n_estimators=config["et_n_estimators"],
            max_depth=config["et_max_depth"],
            min_samples_leaf=config["et_min_samples_leaf"],
            max_features=config["et_max_features"],
            n_splits=n_splits,
        )
    elif model == "xgboost":
        result = xgboost_pipelined_model(
            dataset_name=dataset_name,
            n_estimators=config["xgb_n_estimators"],
            learning_rate=config["xgb_learning_rate"],
            max_depth=config["xgb_max_depth"],
            subsample=config["xgb_subsample"],
            colsample_bytree=config["xgb_colsample_bytree"],
            n_splits=n_splits,
        )
    else:
        result = hgb_pipelined_model(
            dataset_name=dataset_name,
            max_iter=config["hgb_max_iter"],
            learning_rate=config["hgb_learning_rate"],
            max_leaf_nodes=config["hgb_max_leaf_nodes"],
            min_samples_leaf=config["hgb_min_samples_leaf"],
            l2_regularization=config["hgb_l2_regularization"],
            n_splits=n_splits,
        )

    return result[2]


def get_initial_configurations(space, n_trials):
    """Start with one default configuration for each available model."""
    prefixes = {
        "ffn": "ffn_",
        "knn": "knn_",
        "random_forest": "rf_",
        "extra_trees": "et_",
        "xgboost": "xgb_",
        "hgb": "hgb_",
    }
    configurations = []
    for model, prefix in prefixes.items():
        values = {"model": model}
        values.update(
            {
                parameter.name: parameter.default_value
                for parameter in space.values()
                if parameter.name.startswith(prefix)
            }
        )
        configurations.append(Configuration(space, values=values))
    return configurations[:n_trials]


def clean_best_config(config):
    """Remove internal model prefixes from the result shown to users."""
    config = config_to_dict(config)
    model = config.pop("model")
    prefix = {
        "ffn": "ffn_",
        "knn": "knn_",
        "random_forest": "rf_",
        "extra_trees": "et_",
        "xgboost": "xgb_",
        "hgb": "hgb_",
    }[model]
    return {"model": model, **{name.removeprefix(prefix): value for name, value in config.items()}}


def build_stage_two_ensemble(trial_results):
    """Select strong model families and weight them by Stage-1 CV performance."""
    best_by_model = {}
    for result in trial_results:
        model = result["config"]["model"]
        if model not in best_by_model or result["mean_r2"] > best_by_model[model]["mean_r2"]:
            best_by_model[model] = result

    ranked = sorted(best_by_model.values(), key=lambda result: result["mean_r2"], reverse=True)
    best_score = ranked[0]["mean_r2"]
    selected = [result for result in ranked if result["mean_r2"] >= best_score - 0.03][:4]
    if len(selected) < 2:
        selected = ranked[:2]

    scores = np.array([result["mean_r2"] for result in selected])
    weights = np.exp((scores - scores.max()) / 0.02)
    weights /= weights.sum()
    return [
        {
            "name": result["config"]["model"],
            "weight": float(weight),
            "stage_one_r2": result["mean_r2"],
            "config": result["config"],
        }
        for result, weight in zip(selected, weights)
    ]


def optimize_models(
    dataset_name,
    n_trials=10,
    n_splits=5,
    output_directory=None,
    progress_callback=None,
):
    """Return the best model configuration and its mean CV R2."""
    trial_number = 0
    trial_results = []

    def report(message):
        print(message, flush=True)
        if progress_callback:
            progress_callback(message)

    def objective(config, seed=42):
        nonlocal trial_number
        trial_number += 1
        np.random.seed(seed)
        torch.manual_seed(seed)

        report(f"Trial {trial_number}/{n_trials} started")
        report(f"Configuration: {clean_best_config(config)}")
        with contextlib.redirect_stdout(LiveOutput(progress_callback)):
            r2_scores = run_configuration(config, dataset_name, n_splits)

        mean_r2 = float(mu.average_test_loss(r2_scores))
        trial_results.append({"config": clean_best_config(config), "mean_r2": mean_r2})
        report(f"Trial {trial_number}/{n_trials} finished - Mean R2: {mean_r2:.4f}")
        return 1.0 - mean_r2 if np.isfinite(mean_r2) else 1e10

    scenario = Scenario(
        get_configspace(),
        deterministic=True,
        n_trials=n_trials,
        output_directory=output_directory or ROOT / "smac_runs" / dataset_name,
    )
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        n_configs=0,
        additional_configs=get_initial_configurations(scenario.configspace, n_trials),
    )
    optimizer = HyperparameterOptimizationFacade(
        scenario,
        objective,
        initial_design=initial_design,
        overwrite=True,
    )
    optimizer.optimize()

    best_individual = max(trial_results, key=lambda result: result["mean_r2"])
    selected_models = build_stage_two_ensemble(trial_results)
    report("Stage 2 ensemble started")
    for member in selected_models:
        report(
            f"Selected {member['name']} - Stage 1 R2: {member['stage_one_r2']:.4f} "
            f"- Weight: {member['weight']:.4f}"
        )

    with contextlib.redirect_stdout(LiveOutput(progress_callback)):
        _, _, ensemble_scores = ensemble_pipelined_model(
            dataset_name,
            selected_models,
            n_splits=n_splits,
        )
    ensemble_r2 = float(mu.average_test_loss(ensemble_scores))
    report(f"Stage 2 ensemble finished - Mean R2: {ensemble_r2:.4f}")

    if ensemble_r2 > best_individual["mean_r2"]:
        return {
            "model": "ensemble",
            "selected_models": selected_models,
        }, ensemble_r2

    report("Stage 2 did not beat the best individual model; keeping the individual model.")
    return best_individual["config"], best_individual["mean_r2"]
