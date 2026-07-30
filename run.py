import argparse
from pathlib import Path

import smac_optim as optim

import train_and_test as tt
from utils.data_utils import DATASET_DICT

if __name__=='__main__':

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     "--task",
    #     type=str,
    #     required=True,
    #     help="The name of the task to run on.",
    #     choices=["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1",
    #              "exam_dataset"]
    # )
    #
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=42,
    #     help=(
    #         "Random seed for reproducibility if you are using and randomness,"
    #         " i.e. torch, numpy, pandas, sklearn, etc."
    #     )
    # )
    #
    # parser.add_argument(
    #     "--output-path",
    #     type=Path,
    #     default=Path("./predictions.npy"),
    #     help=(
    #         "The path to save the predictions to."
    #         " By default this will just save to './predictions.npy'."
    #     )
    # )
    #
    # parser.add_argument(
    #     "--hpo",
    #     type=str,
    #     default="off",
    #     choices=["off", "smac_hpo", "smac_mfo"],
    #     help=(
    #         "Optional HPO layer."
    #     ),
    # )
    #
    # parser.add_argument(
    #     "--hpo-trials",
    #     type=int,
    #     default=200,
    #     help="Number of cheap HPO configurations to evaluate when --hpo is enabled. Default is 200 trials.",
    # )
    #
    # parser.add_argument(
    #     "--hpo-workers",
    #     type=int,
    #     default=1,
    #     help="Number of workers to use for HPO. Default is 1 worker.",
    # )
    #
    # parser.add_argument(
    #     "--hpo-timeout",
    #     type=float,
    #     default=86400,
    #     help="Optional HPO-only timeout in seconds. Default is 1 day.",
    # )
    #
    # args = parser.parse_args()
    #
    # if args.hpo == 'off':
    #
    #     tt.train(
    #         dataset_name=args.task,
    #         seed=args.seed,
    #         save_path=args.output_path
    #     )

    # if args.hpo == 'smac_hpo':
    #
    #     optim.run_hp_optimizer(
    #         dataset_name=args.task,
    #         n_trials=args.hpo_trials,
    #         n_workers=args.hpo_workers,
    #         walltime_limit=args.hpo_timeout
    #     )
    #
    # if args.hpo == 'smac_mfo':
    #     optim.run_mf_optimizer(
    #         dataset_name=args.task,
    #         n_trials=args.hpo_trials,
    #         n_workers=args.hpo_workers,
    #         walltime_limit=args.hpo_timeout
    #     )

    for k, v in DATASET_DICT.items():
        optim.run_mf_optimizer(
            dataset_name=v,
            n_trials=200,
            n_workers=1,
            walltime_limit=86400
        )