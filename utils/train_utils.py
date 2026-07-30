from pathlib import Path
import json

import numpy as np

def _extract_top_unique_data_configs(data):

    seen = set()
    result = []

    for item in data:
        if not item.get('config') or 'model' not in item['config']:
            continue

        if item['config']['model'] not in seen:
            seen.add(item['config']['model'])
            result.append(item)

    return result

def _extract_top_configs_per_model(path):

    if not path.is_file():
        return None

    try:

        obj = json.loads(path.read_text(encoding='utf-8'))
        configs = obj.get('configs', {})
        data = obj.get('data', [])[::-1] # Reverse the order to get the best configurations first (in MFO case)

        data_with_config_info = []

        for item in data:
            config_id = str(item.get('config_id'))
            item['config'] = configs.get(config_id)
            data_with_config_info.append(item)

        data_with_config_info_st = sorted(data_with_config_info, key=lambda x: x.get('cost', float('inf')))

        result = _extract_top_unique_data_configs(data_with_config_info_st)

        return result

    except Exception as e:
        return None

def get_best_configs(configs, threshold=0.003):

    best_configs = [configs[0]]

    for i in range(1, len(configs)):
        if (configs[i]['cost'] - configs[0]['cost']) <= threshold:
            best_configs.append(configs[i])

    return best_configs

def get_best_model_configs(configs):

    best_model_configs = []

    for config in configs:
        best_model_configs.append(config['config'])

    return best_model_configs

def get_top_configs_per_model(dataset_name):

    base_dir = Path(__file__).resolve().parent.parent / 'outputs'
    path_1 = base_dir / 'hpo' / f'smac_hp_optim_{dataset_name}' / '0' / 'runhistory.json'
    path_2 = base_dir / 'multi-fidelity' / f'smac_mf_optim_{dataset_name}' / '0' / 'runhistory.json'

    if not path_1.exists() and not path_2.exists():
        raise FileNotFoundError(f"No runhistory files exist for dataset '{dataset_name}' at either path.")

    best_configs_1 = _extract_top_configs_per_model(path_1)
    best_configs_2 = _extract_top_configs_per_model(path_2)

    if best_configs_1 is None and best_configs_2 is None:
        raise ValueError(f"No valid runhistory data found for '{dataset_name}' in either path.")
    elif best_configs_1 is None:
        return best_configs_2
    elif best_configs_2 is None:
        return best_configs_1

    return best_configs_1

def get_weights(best_configs, T=0.01):

    r2_scores = np.array([(1 - config['cost']) for config in best_configs])
    scaled_scores = (r2_scores - np.max(r2_scores)) / T
    exp_scores = np.exp(scaled_scores)

    return exp_scores / np.sum(exp_scores)