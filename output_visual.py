"""Export SMAC JSON output to a flat, spreadsheet-friendly CSV file.

The script accepts either a single SMAC run directory (for example
``outputs/smac_optim_bike_sharing_demand/0``) or a parent directory containing
one or more runs. If no directory is supplied, every SMAC run below
``outputs/`` is exported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIRECTORY = PROJECT_ROOT / "outputs"
DEFAULT_CSV_NAME = "smac_trials.csv"

# SMAC's StatusType integer values. The numeric status is also retained in the
# CSV so output from a different SMAC version remains unambiguous.
STATUS_NAMES = {
    0: "RUNNING",
    1: "SUCCESS",
    2: "CRASHED",
    3: "TIMEOUT",
    4: "MEMORYOUT",
}

FIXED_COLUMNS = [
    "run_directory",
    "scenario_name",
    "dataset",
    "run_seed",
    "optimization_finished",
    "config_id",
    "model",
    "config_origin",
    "is_incumbent",
    "status",
    "status_code",
    "cost",
    "r2_score",
    "duration_seconds",
    "cpu_time_seconds",
    "start_time_unix",
    "end_time_unix",
    "trial_seed",
    "instance",
    "budget",
]


def _load_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required SMAC file not found: {path}")
        return {}

    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not read JSON file {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested dictionaries while keeping lists as JSON strings."""
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, nested_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child_prefix, nested_value))
        return flattened
    if isinstance(value, (list, tuple)):
        return {prefix: json.dumps(value, ensure_ascii=False)}
    return {prefix: value}


def _find_run_directories(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != "runhistory.json":
            raise ValueError("A file input must be named runhistory.json")
        return [path.parent]
    if not path.exists():
        raise FileNotFoundError(f"SMAC output path does not exist: {path}")
    if (path / "runhistory.json").is_file():
        return [path]
    return sorted(
        {runhistory.parent for runhistory in path.rglob("runhistory.json")},
        key=lambda item: str(item),
    )


def _dataset_name(scenario_name: Any) -> str:
    name = str(scenario_name or "")
    prefix = "smac_optim_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _safe_r2(cost: Any, status_code: Any) -> float | None:
    """Convert this project's SMAC loss (1 - R2) back to R2."""
    if status_code != 1 or isinstance(cost, bool) or not isinstance(cost, (int, float)):
        return None
    if not math.isfinite(cost):
        return None
    return 1.0 - float(cost)


def _rows_for_run(run_directory: Path) -> list[dict[str, Any]]:
    runhistory = _load_json(run_directory / "runhistory.json", required=True)
    scenario = _load_json(run_directory / "scenario.json")
    optimization = _load_json(run_directory / "optimization.json")
    intensifier = _load_json(run_directory / "intensifier.json")

    configs = runhistory.get("configs", {})
    origins = runhistory.get("config_origins", {})
    trials = runhistory.get("data", [])
    incumbent_ids = {str(value) for value in intensifier.get("incumbent_ids", [])}

    if not isinstance(configs, dict) or not isinstance(origins, dict):
        raise ValueError(f"Invalid configuration data in {run_directory / 'runhistory.json'}")
    if not isinstance(trials, list):
        raise ValueError(f"Invalid trial data in {run_directory / 'runhistory.json'}")

    scenario_name = scenario.get("name", run_directory.parent.name)
    rows: list[dict[str, Any]] = []
    for trial_number, trial in enumerate(trials, start=1):
        if not isinstance(trial, dict):
            continue

        config_id = str(trial.get("config_id", ""))
        config = configs.get(config_id, {})
        if not isinstance(config, dict):
            config = {}

        status_code = trial.get("status")
        cost = trial.get("cost")
        row: dict[str, Any] = {
            "run_directory": os.path.relpath(run_directory, PROJECT_ROOT),
            "scenario_name": scenario_name,
            "dataset": _dataset_name(scenario_name),
            "run_seed": scenario.get("seed", run_directory.name),
            "optimization_finished": optimization.get("finished"),
            "trial_number": trial_number,
            "config_id": trial.get("config_id"),
            "model": config.get("model"),
            "config_origin": origins.get(config_id),
            "is_incumbent": config_id in incumbent_ids,
            "status": STATUS_NAMES.get(status_code, f"UNKNOWN_{status_code}"),
            "status_code": status_code,
            "cost": cost,
            "r2_score": _safe_r2(cost, status_code),
            "duration_seconds": trial.get("time"),
            "cpu_time_seconds": trial.get("cpu_time"),
            "start_time_unix": trial.get("starttime"),
            "end_time_unix": trial.get("endtime"),
            "trial_seed": trial.get("seed"),
            "instance": trial.get("instance"),
            "budget": trial.get("budget"),
        }
        row.update(
            {
                f"additional_info.{key}": value
                for key, value in _flatten("", trial.get("additional_info", {})).items()
                if key
            }
        )
        row.update({f"hp.{key}": value for key, value in config.items() if key != "model"})
        rows.append(row)

    return rows


def _ordered_columns(rows: Iterable[dict[str, Any]]) -> list[str]:
    rows = list(rows)
    all_columns = {key for row in rows for key in row}
    fixed = ["trial_number", *FIXED_COLUMNS]
    ordered_fixed = [column for column in fixed if column in all_columns]
    hyperparameters = sorted(column for column in all_columns if column.startswith("hp."))
    additional_info = sorted(
        column for column in all_columns if column.startswith("additional_info.")
    )
    remaining = sorted(all_columns - set(ordered_fixed) - set(hyperparameters) - set(additional_info))
    return [*ordered_fixed, *hyperparameters, *additional_info, *remaining]


def export_smac_results(
    input_path: str | Path = DEFAULT_OUTPUTS_DIRECTORY,
    csv_path: str | Path | None = None,
) -> Path:
    """Export all runs under *input_path* and return the generated CSV path."""
    input_path = Path(input_path)
    run_directories = _find_run_directories(input_path)
    if not run_directories:
        raise FileNotFoundError(f"No runhistory.json files found below {input_path.resolve()}")

    rows = [row for run_directory in run_directories for row in _rows_for_run(run_directory)]
    if not rows:
        raise ValueError("SMAC run history contains no trial rows yet")

    if csv_path is None:
        output_directory = input_path if input_path.is_dir() else input_path.parent
        csv_path = output_directory / DEFAULT_CSV_NAME
    csv_path = Path(csv_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = csv_path.with_suffix(f"{csv_path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=_ordered_columns(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(csv_path)
    return csv_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flatten one or more SMAC JSON run histories into a CSV file."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_OUTPUTS_DIRECTORY,
        help="SMAC run directory, parent output directory, or runhistory.json file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="csv_path",
        help=f"Destination CSV path (default: <input>/{DEFAULT_CSV_NAME}).",
    )
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()
    csv_path = export_smac_results(arguments.input_path, arguments.csv_path)
    print(f"Created CSV: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
