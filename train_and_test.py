import numpy as np

import utils.optim_utils as ou

import models.ffn as ffn
import models.randomforest as rf
import models.xgb as xgb
import models.pfn as pfn
import models.extratrees as et
import models.hgb as hgb

def train_model(dataset_name, seed, path):

    try:
        config = ou.get_incumbent_config(dataset_name)
        model = config.get('model')

        match model:
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

                m, y, y_hat, r2_score = ffn.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)

                print(f"FFN yields an average R2 Score of ", r2_score)

            case 'xgb':

                model_hps = (
                    config.get('n_estimators_xgb'),
                    config.get('learning_rate_xgb'),
                    config.get('max_depth_xgb'),
                    config.get('subsample'),
                    config.get('colsample_bytree'),
                    config.get('reg_lambda')
                )

                m, y, y_hat, r2_score = xgb.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)

                print(f"XGB yields an average R2 Score of ", r2_score)

            case 'randomforest':

                model_hps = (
                    config.get('n_estimators_rf'),
                    config.get('max_depth_rf'),
                    config.get('min_samples_leaf_rf'),
                    config.get('max_features_rf')
                )

                m, y, y_hat, r2_score = rf.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)

                print(f"Random Forest yields an average R2 Score of ", r2_score)

            case 'pfn':

                m, y, y_hat, r2_score = pfn.train(dataset_name=dataset_name, seed=seed)

                print(f"TabPFN yields an average R2 Score of ", r2_score)

            case 'extratrees':

                model_hps = (
                    config.get('n_estimators_et'),
                    config.get('max_depth_et'),
                    config.get('min_samples_leaf_et'),
                    config.get('max_features_et')
                )

                m, y, y_hat, r2_score = et.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)

                print(f"Extratrees yields an average R2 Score of ", r2_score)

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

                m, y, y_hat, r2_score = hgb.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)

                print(f"HGB yields an average R2 Score of ", r2_score)

            case _:
                raise ValueError(f"Unsupported model type: {model}")

        np.save(path, y_hat)

    except (KeyError, FileNotFoundError, ValueError) as e:

        print(
            f"Failed to load or match configuration ({e}). " 
            "Training with the baseline model FFN... "
            "NOTE: This may lead to suboptimal results. " 
            "Please consider running the SMAC optimizer to find the best configuration first and then run the model. "
              , flush=True)

        base_config = {
            'batch_size': 64,
            'learning_rate_ffn': 1e-3,
            'no_of_layers': 2,
            'no_of_units': 32,
            'weight_decay': 1e-4,
            'batch_norm': False,
            'num_epochs': 50
        }

        model_hps = (
            base_config.get('batch_size'),
            base_config.get('learning_rate_ffn'),
            base_config.get('no_of_layers'),
            base_config.get('no_of_units'),
            base_config.get('weight_decay'),
            base_config.get('batch_norm'),
            base_config.get('num_epochs')
        )

        m, y, y_hat, r2_score = ffn.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)

        np.save(path, np.asarray(y_hat))

        print(f"The baseline hyperparameter configuration {base_config} for FFN yields an average R2 Score of", r2_score)
