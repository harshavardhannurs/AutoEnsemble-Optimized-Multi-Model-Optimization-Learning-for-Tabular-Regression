import json
from pathlib import Path


def log_execution_time(dataset_name, elapsed_seconds):
    file_path = Path(__file__).resolve().parent.parent / 'outputs' / 'optim_exec_times.txt'

    with open(file_path, "a", encoding='utf-8') as f:
        f.write(f"{dataset_name} = {elapsed_seconds:.2f}\n")

def _extract_incumbent_id(path):

    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        ids = obj.get('incumbent_ids')
        return ids[0] if ids else None
    except Exception:
        return None


def _extract_cost(path, config_id):

    if config_id is None or not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        costs = [
            item.get('cost')
            for item in obj.get('data', [])
            if item.get('config_id') == config_id and item.get('cost') is not None
        ]
        return min(costs) if costs else None
    except Exception:
        return None


def _extract_config(path, config_id):

    if config_id is None or not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        configs = obj.get('configs', {})
        return configs.get(str(config_id))
    except Exception:
        return None

def get_incumbent_ids(dataset_name):
    base_dir = Path(__file__).resolve().parent.parent / 'outputs'
    path_1 = base_dir / 'hpo' / f'smac_hp_optim_{dataset_name}' / '0' / 'intensifier.json'
    path_2 = base_dir / 'multi-fidelity' / f'smac_mf_optim_{dataset_name}' / '0' / 'intensifier.json'

    id_1 = _extract_incumbent_id(path_1)
    id_2 = _extract_incumbent_id(path_2)

    if id_1 is None and id_2 is None:
        raise FileNotFoundError(f"No valid intensifier.json or incumbent IDs found for '{dataset_name}' in either path.")

    return id_1, id_2


def get_incumbent_costs(dataset_name):
    base_dir = Path(__file__).resolve().parent.parent / 'outputs'
    path_1 = base_dir / 'hpo' / f'smac_hp_optim_{dataset_name}' / '0' / 'runhistory.json'
    path_2 = base_dir / 'multi-fidelity' / f'smac_mf_optim_{dataset_name}' / '0' / 'runhistory.json'

    inc_id_1, inc_id_2 = get_incumbent_ids(dataset_name)

    cost_1 = _extract_cost(path_1, inc_id_1)
    cost_2 = _extract_cost(path_2, inc_id_2)

    if cost_1 is None and cost_2 is None:
        raise ValueError(f"No valid runhistory cost data found for '{dataset_name}' in either path.")

    return cost_1, cost_2


def get_incumbent_config(dataset_name):
    base_dir = Path(__file__).resolve().parent.parent / 'outputs'
    path_1 = base_dir / 'hpo' / f'smac_hp_optim_{dataset_name}' / '0' / 'runhistory.json'
    path_2 = base_dir / 'multi-fidelity' / f'smac_mf_optim_{dataset_name}' / '0' / 'runhistory.json'

    cost_1, cost_2 = get_incumbent_costs(dataset_name)
    inc_id_1, inc_id_2 = get_incumbent_ids(dataset_name)

    c1 = cost_1 if cost_1 is not None else float('inf')
    c2 = cost_2 if cost_2 is not None else float('inf')

    if c1 == float('inf') and c2 == float('inf'):
        raise FileNotFoundError(f"No valid incumbent configurations could be retrieved for '{dataset_name}'.")

    if c1 <= c2:
        config = _extract_config(path_1, inc_id_1)
        target_path, target_id = path_1, inc_id_1
    else:
        config = _extract_config(path_2, inc_id_2)
        target_path, target_id = path_2, inc_id_2

    if config is None:
        raise KeyError(f"Configuration ID '{target_id}' could not be loaded from {target_path}")

    return config