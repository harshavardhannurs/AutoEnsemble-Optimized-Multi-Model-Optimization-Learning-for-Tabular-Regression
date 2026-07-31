import numpy as np

import utils.train_utils as tu
import utils.data_utils as du

import models.ffn as ffn

def train(dataset_name, delta, temperature, save_path, seed):

    try:
        # Step 1: Fetch top configs per model type
        top_configs_per_model = tu.get_top_configs_per_model(dataset_name=dataset_name)
        
        # Step 2: Get OOF predictions to calculate weights
        print('Calculating optimal weights using NNLS and OOF predictions...')
        oof_preds_dict = {}
        _, y_train_all, _, _ = du.get_all_folds_data(dataset_name)
        
        for config_dict in top_configs_per_model:
            config = config_dict['config']
            model_name = config['model']
            print(f'Generating OOF predictions for {model_name}...')
            oof_preds = tu.get_oof_predictions(dataset_name, config)
            oof_preds_dict[model_name] = oof_preds
        
        weights_dict = tu.find_weights(oof_preds_dict, y_train_all)
        print(f'Optimal weights found: {weights_dict}')

        # Step 3: Train models with non-zero weights and get test predictions
        final_predictions = []
        final_weights = []
        
        print('Training models with non-zero weights on the full dataset...')
        for config_dict in top_configs_per_model:
            config = config_dict['config']
            model_name = config['model']
            weight = weights_dict.get(model_name, 0.0)
            
            if weight > 0:
                print(f'Training final {model_name} model...')
                _, _, y_hat = tu.train_final_model(dataset_name, config, seed)
                final_predictions.append(y_hat)
                final_weights.append(weight)
        
        if not final_predictions:
            raise ValueError("No models with positive weights found.")

        # Step 4: Weighted sum of predictions
        print('Calculating weighted sum of predictions...')
        final_predictions = np.array(final_predictions)
        y_hat_weighted_average = np.average(final_predictions, axis=0, weights=final_weights)

        # Save the predictions
        np.save(save_path, y_hat_weighted_average)
        print("Predictions saved at ", save_path)

    except (FileNotFoundError, ValueError, Exception) as e:

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
            1 # num_epochs
        )

        m, y_test, y_hat = ffn.train(dataset_name=dataset_name, model_hps=model_hps, seed=seed)

        np.save(save_path, np.asarray(y_hat))

        print("Predictions saved at ", save_path)
