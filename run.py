import argparse
from pathlib import Path

import smac_optim as optim

import train_and_test as tt
from utils.data_utils import DATASET_DICT

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Dataset on which to train the model and generate predictions.",
        choices=["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1",
                 "exam_dataset"]
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used to ensure reproducible results for all stochastic "
            "operations (e.g., NumPy, PyTorch, and scikit-learn). "
            "Default is 42."
        )
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./predictions.npy"),
        help=(
            "Path where the generated predictions will be saved."
            " By default, they are just saved to './predictions.npy'."
        )
    )

    parser.add_argument(
        "--hpo",
        type=str,
        default="off",
        choices=["off", "smac_hpo", "smac_mfo"],
        help=(
            "Hyperparameter optimization method to use. "
            "'off' uses the bundled optimization results if available, "
            "otherwise trains a baseline model with default hyperparameters. "
            "'smac_hpo' runs SMAC's HyperparameterOptimizationFacade. "
            "'smac_mfo' runs SMAC's MultiFidelityFacade."
        ),
    )

    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=200,
        help=(
            "Maximum number of configurations to evaluate during hyperparameter optimization."
            " Default is 200."
        ),
    )

    parser.add_argument(
        "--hpo-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers used during hyperparameter optimization. "
            "Default is 1."
        )
    )

    parser.add_argument(
        "--hpo-timeout",
        type=float,
        default=86400,
        help=(
            "Maximum time (in seconds) allotted for hyperparameter optimization."
            " Default is 86400 (24 hours)."
        ),
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.003,
        help=(
            "Maximum acceptable difference from the best optimization cost when "
            "selecting models for the ensemble. "
            "Default is 0.003."
        )
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help=(
            "Controls how strongly better-performing models are weighted in the "
            "ensemble. Lower values place more emphasis on the best models. "
            "Default is 0.01."
        )
    )

    args = parser.parse_args()

    if args.hpo == 'off':

        tt.train(
            dataset_name=args.task,
            save_path=args.output_path,
            seed=args.seed,
        )

    if args.hpo == 'smac_hpo':

        optim.run_hp_optimizer(
            dataset_name=args.task,
            n_trials=args.hpo_trials,
            n_workers=args.hpo_workers,
            walltime_limit=args.hpo_timeout
        )

        # tt.train(
        #     dataset_name=args.task,
        #     seed=args.seed,
        #     save_path=args.output_path
        # )

    if args.hpo == 'smac_mfo':

        optim.run_mf_optimizer(
            dataset_name=args.task,
            n_trials=args.hpo_trials,
            n_workers=args.hpo_workers,
            walltime_limit=args.hpo_timeout
        )

        # tt.train(
        #     dataset_name=args.task,
        #     seed=args.seed,
        #     save_path=args.output_path
        # )

    # for k, v in DATASET_DICT.items():
    #     optim.run_mf_optimizer(
    #         dataset_name=v,
    #         n_trials=200,
    #         n_workers=1,
    #         walltime_limit=86400
    #     )