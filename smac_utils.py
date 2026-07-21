"""SMAC optimization for the feed-forward neural network."""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario

from models.ffn import ffn_pipelined_model
from utils import model_utils as mu


ROOT = Path(__file__).resolve().parent


def config_to_dict(config):
    return {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in dict(config).items()
    }


class LiveOutput(io.TextIOBase):
    """Copy training output to the terminal and a UI callback."""

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


def get_ffn_configspace():
    """Create the FFN search space defined in hp_space.md."""
    space = ConfigurationSpace(seed=42)
    space.add(
        [
            Integer("no_of_layers", (1, 4), default=2),
            Integer("no_of_units", (32, 512), default=128),
            Float("learning_rate", (1e-5, 1e-2), default=1e-3, log=True),
            Integer("batch_size", (32, 256), default=64),
            Float("weight_decay", (1e-6, 1e-3), default=1e-4, log=True),
            Categorical("batch_norm", [False, True], default=False),
            Integer("num_epochs", (100, 1000), default=100),
        ]
    )
    return space


def optimize_ffn(
    dataset_name,
    n_trials=10,
    n_splits=5,
    output_directory=None,
    progress_callback=None,
):
    """Return the best FFN configuration and its mean cross-validation R2."""
    trial_number = 0

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
        report(f"Hyperparameters: {config_to_dict(config)}")

        with contextlib.redirect_stdout(LiveOutput(progress_callback)):
            _, _, r2_scores = ffn_pipelined_model(
                dataset_name=dataset_name,
                batch_size=config["batch_size"],
                no_of_layers=config["no_of_layers"],
                no_of_units=config["no_of_units"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                batch_norm=config["batch_norm"],
                num_epochs=config["num_epochs"],
                n_splits=n_splits,
            )

        mean_r2 = float(mu.average_test_loss(r2_scores))
        report(f"Trial {trial_number}/{n_trials} finished - Mean R2: {mean_r2:.4f}")
        return 1.0 - mean_r2 if np.isfinite(mean_r2) else 1e10

    scenario = Scenario(
        get_ffn_configspace(),
        deterministic=True,
        n_trials=n_trials,
        output_directory=output_directory or ROOT / "smac_runs" / dataset_name,
    )
    optimizer = HyperparameterOptimizationFacade(
        scenario,
        objective,
        overwrite=True,
    )
    best_config = optimizer.optimize()
    best_cost = optimizer.runhistory.get_cost(best_config)

    return config_to_dict(best_config), 1.0 - float(best_cost)
