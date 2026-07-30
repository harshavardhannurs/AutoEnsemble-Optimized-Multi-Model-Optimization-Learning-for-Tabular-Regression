import numpy as np

import utils.train_utils as tu

import models.ffn as ffn
import models.randomforest as rf
import models.xgb as xgb
import models.pfn as pfn
import models.extratrees as et
import models.hgb as hgb

def train(dataset_name, delta, temperature, save_path, seed):

    try:
        top_configs_per_model = tu.get_top_configs_per_model(dataset_name=dataset_name)
        best_configs = tu.get_best_configs(top_configs_per_model, delta=delta)
        model_configs = tu.get_best_model_configs(best_configs)

        predictions = []

        print('Top performing models were found to be: ', [config['model'] for config in model_configs])
        print('Training these models with the best configurations found from SMAC optim files...')

        for config in model_configs:

            model_name = config['model']

            match model_name:
                case 'ffn':
                    model_hps = (
                        config.get('batch_size'),
                        config.get('learning_rate_ffn'),
                        config.get('no_of_layers'),
                        config.get('no_of_units'),
                        config.get('weight_decay'),
                        config.get('batch_norm'),
                        config.get('num_epochs')
                    )
                    m, y, y_hat = ffn.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
                case 'xgb':
                    model_hps = (
                        config.get('n_estimators_xgb'),
                        config.get('learning_rate_xgb'),
                        config.get('max_depth_xgb'),
                        config.get('subsample'),
                        config.get('colsample_bytree'),
                        config.get('reg_lambda')
                    )
                    m, y, y_hat = xgb.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
                case 'randomforest':
                    model_hps = (
                        config.get('n_estimators_rf'),
                        config.get('max_depth_rf'),
                        config.get('min_samples_leaf_rf'),
                        config.get('max_features_rf')
                    )
                    m, y, y_hat = rf.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
                case 'pfn':
                    m, y, y_hat = pfn.train(dataset_name=dataset_name, seed=seed)
                case 'extratrees':
                    model_hps = (
                        config.get('n_estimators_et'),
                        config.get('max_depth_et'),
                        config.get('min_samples_leaf_et'),
                        config.get('max_features_et')
                    )
                    m, y, y_hat = et.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
                case 'hgb':
                    model_hps = (
                        config.get('max_iter_hgb'),
                        config.get('learning_rate_hgb'),
                        config.get('max_leaf_nodes_hgb'),
                        config.get('min_samples_leaf_hgb'),
                        config.get('l2_reg_hgb'),
                        config.get('max_features_hgb'),
                        config.get('max_bins_hgb')
                    )
                    m, y, y_hat = hgb.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)
                case _:
                    raise ValueError(f"Unsupported model type: {model_name}")

            predictions.append(y_hat)

        # Now find the weighted average of predictions here

        predictions = np.array(predictions)
        weights = tu.get_weights(best_configs, temperature=temperature)

        print('Finding weighted average of predictions...')

        y_hat_weighted_average = np.average(predictions, axis=0, weights=weights)

        # Save the predictions
        np.save(save_path, y_hat_weighted_average)

        print("Predictions saved at ", save_path)

    except (FileNotFoundError, ValueError) as e:

        base_config = {
            'batch_size': 32,
            'learning_rate_ffn': 1e-3,
            'no_of_layers': 3,
            'no_of_units': 64,
            'weight_decay': 1e-3,
            'batch_norm': False,
            'num_epochs': 300
        }

        print(
            f"Failed to load or match configuration ({e}). "
            f"Training with the baseline model FFN with default parameters {base_config}... "
            "NOTE: This may lead to suboptimal results. "
            "Please consider running the SMAC optimizer to find the best configuration first and then run the command. "
            , flush=True)

        model_hps = (
            base_config.get('batch_size'),
            base_config.get('learning_rate_ffn'),
            base_config.get('no_of_layers'),
            base_config.get('no_of_units'),
            base_config.get('weight_decay'),
            base_config.get('batch_norm'),
            base_config.get('num_epochs')
        )

        m, y_hat = ffn.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)

        np.save(save_path, np.asarray(y_hat))

        print("Predictions saved at ", save_path)
