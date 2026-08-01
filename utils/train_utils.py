from pathlib import Path
import json

import numpy as np

import torch

import scipy.optimize as opt

import models.ffn as ffn
import models.randomforest as rf
import models.xgb as xgb
import models.pfn as pfn
import models.extratrees as et
import models.hgb as hgb

import utils.data_utils as du
import utils.model_utils as mu

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
        data = obj.get('data', [])

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

def get_weights(best_configs, temperature=0.01):

    r2_scores = np.array([(1 - config['cost']) for config in best_configs])
    scaled_scores = (r2_scores - np.max(r2_scores)) / temperature
    exp_scores = np.exp(scaled_scores)

    return exp_scores / np.sum(exp_scores)

def get_model_hps(config):

    """
    Extracts model hyperparameters from a configuration dictionary.
    """

    model_name = config['model']
    match model_name:
        case 'ffn':
            return (
                config.get('batch_size'),
                config.get('learning_rate_ffn'),
                config.get('no_of_layers'),
                config.get('no_of_units'),
                config.get('weight_decay'),
                config.get('batch_norm'),
                config.get('num_epochs')
            )
        case 'xgb':
            return (
                config.get('n_estimators_xgb'),
                config.get('learning_rate_xgb'),
                config.get('max_depth_xgb'),
                config.get('subsample'),
                config.get('colsample_bytree'),
                config.get('reg_lambda')
            )
        case 'randomforest':
            return (
                config.get('n_estimators_rf'),
                config.get('max_depth_rf'),
                config.get('min_samples_leaf_rf'),
                config.get('max_features_rf')
            )
        case 'pfn':
            return config.get('n_estimators_pfn')
        case 'extratrees':
            return (
                config.get('n_estimators_et'),
                config.get('max_depth_et'),
                config.get('min_samples_leaf_et'),
                config.get('max_features_et')
            )
        case 'hgb':
            return (
                config.get('max_iter_hgb'),
                config.get('learning_rate_hgb'),
                config.get('max_leaf_nodes_hgb'),
                config.get('min_samples_leaf_hgb'),
                config.get('l2_reg_hgb'),
                config.get('max_features_hgb'),
                config.get('max_bins_hgb')
            )
        case _:
            raise ValueError(f"Unsupported model type: {model_name}")

def train_final_model(dataset_name, config, seed):
    """
    Trains a model on all available data and returns predictions.
    """
    model_name = config['model']
    model_hps = get_model_hps(config)
    
    match model_name:
        case 'ffn':
            return ffn.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
        case 'xgb':
            return xgb.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
        case 'randomforest':
            return rf.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
        case 'pfn':
            return pfn.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
        case 'extratrees':
            return et.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
        case 'hgb':
            return hgb.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
        case _:
            raise ValueError(f"Unsupported model type: {model_name}")

def get_oof_predictions(dataset_name, config, seed):

    oof_predictions_list = []
    model_name = config['model']
    num_folds = du.get_num_folds(dataset_name)

    # Determine model_hps based on model type
    model_hps = get_model_hps(config)

    for fold in range(1, num_folds + 1):

        print("Evaluating OOF predictions for fold ", fold, "...")

        X_train_fold, y_train_fold, X_val_fold, y_val_fold = du.get_oof_data(dataset_name, fold, seed)

        if y_val_fold is None or y_val_fold.empty:
             continue

        match model_name:
            case 'xgb':
                model = xgb.setup_model(*model_hps)
                transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False)
                X_train_fold_t = du.to_dense(transformer_X.fit_transform(X_train_fold))
                X_val_fold_t = du.to_dense(transformer_X.transform(X_val_fold))
                model.fit(X_train_fold_t, y_train_fold.values.ravel())
                y_hat = model.predict(X_val_fold_t)
            
            case 'randomforest':
                model = rf.setup_model(*model_hps)
                transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False)
                X_train_fold_t = du.to_dense(transformer_X.fit_transform(X_train_fold))
                X_val_fold_t = du.to_dense(transformer_X.transform(X_val_fold))
                model.fit(X_train_fold_t, y_train_fold.values.ravel())
                y_hat = model.predict(X_val_fold_t)

            case 'extratrees':
                model = et.setup_model(*model_hps)
                transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False)
                X_train_fold_t = du.to_dense(transformer_X.fit_transform(X_train_fold))
                X_val_fold_t = du.to_dense(transformer_X.transform(X_val_fold))
                model.fit(X_train_fold_t, y_train_fold.values.ravel())
                y_hat = model.predict(X_val_fold_t)

            case 'hgb':
                model = hgb.setup_model(*model_hps)
                transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False)
                X_train_fold_t = du.to_dense(transformer_X.fit_transform(X_train_fold))
                X_val_fold_t = du.to_dense(transformer_X.transform(X_val_fold))
                model.fit(X_train_fold_t, y_train_fold.values.ravel())
                y_hat = model.predict(X_val_fold_t)

            case 'pfn':
                train_size = X_train_fold.shape[0]
                subsample_cap = 900 if train_size > 6000 else train_size
                
                if isinstance(model_hps, (tuple, list)) and len(model_hps) > 0:
                    n_estimators = model_hps[0]
                else:
                    n_estimators = model_hps

                model = pfn.create_regressor(
                    n_estimators=n_estimators,
                    subsample_cap=subsample_cap,
                    seed=seed
                )
                transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold)
                X_train_fold_t = transformer_X.fit_transform(X_train_fold)
                X_val_fold_t = transformer_X.transform(X_val_fold)
                model.fit(X_train_fold_t, y_train_fold.values.ravel())
                y_hat = model.predict(X_val_fold_t)

            case 'ffn':
                # FFN training
                
                batch_size, learning_rate, no_of_layers, no_of_units, weight_decay, batch_norm, num_epochs = model_hps
                transformer_X, transformer_y = du.data_transformer(X_train_fold, y_train_fold)
                X_train_t = du.to_dense(transformer_X.fit_transform(X_train_fold))
                X_val_t = du.to_dense(transformer_X.transform(X_val_fold))
                y_train_t = transformer_y.fit_transform(y_train_fold)
                
                train_loader = du.get_dataloader(X_train_t, y_train_t, batch_size=batch_size)
                device = mu.set_device()

                model, optimizer, loss_function = ffn.setup_model(
                    no_of_feats=X_train_t.shape[1],
                    no_of_layers=no_of_layers,
                    no_of_units=no_of_units,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    batch_norm=batch_norm
                )
                model = model.to(device)
                
                model.train()
                for epoch in range(num_epochs):
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = loss_function(outputs, batch_y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if (epoch + 1) % 10 == 0:
                        print(f'Epoch {epoch + 1}/{num_epochs} completed')
                
                model.eval()
                with torch.no_grad():
                    X_val_tensor = du.to_tensor(X_val_t).to(device)
                    y_hat = model(X_val_tensor).cpu().numpy()
                y_hat = mu.inverse_transform_target(y_hat, transformer_y)

        oof_predictions_list.append(y_hat.flatten())

    return np.concatenate(oof_predictions_list)

def find_weights(oof_predictions, y_train):
    
    """
    Finds weights for each model using Non-Negative Least Squares (NNLS).
    
    Args:
        oof_predictions: Dict mapping model_name to OOF prediction array.
        y_train: Ground truth targets for the training set.
        
    Returns:
        Dict mapping model_name to weight.
    """

    model_names = list(oof_predictions.keys())
    A = np.column_stack([oof_predictions[name] for name in model_names])
    b = y_train.values.ravel()

    weights, _ = opt.nnls(A, b)
    
    # Normalize weights so they sum to 1
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    else:
        # Fallback if all weights are zero (rare)
        weights = np.ones(len(weights)) / len(weights)
        
    return dict(zip(model_names, weights))
