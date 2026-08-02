# Hyperparameter Space

## Models
- XGBoost – XGBRegressor
- Random Forest – RandomForestRegressor
- TabPFN - TabPFNRegressor
- Feed Forward Neural Network

## Hyperparameter Space
- XGBoost
    - n_estimators: 50-500
    - learning_rate: 1e-3–0.3 (log scale)
    - max_depth: 2-12
    - subsample: 0.5-1.0
    - colsample_bytree: 0.5-1.0
    - reg_lambda: 1e-3 - 100 (log scale)
- Random Forest
    - n_estimators: 50-500
    - max_depth: None, 5-50
    - min_samples_leaf: 1-20
    - max_features: 0.5-1.0
- TabPFN
    - n_estimators: 4-16
- Feed Forward Neural Network
    - batch_size: 32 - 256
    - no_of_layers: 1-4
    - no_of_units: 32 - 512
    - learning_rate: 1e-5 - 1e-1 (log scale)
    - weight_decay: 1e-6 - 1e-2 (log scale)
    - batch_norm: True/False
    - num_epochs: 50-500
- Extra Trees
    - n_estimators: 50-500
    - max_depth: None, 5-50
    - min_samples_leaf: 1-20
    - max_features: 0.3-1.0
- Histogram Gradient Boosting
    - max_iter: 100-1000
    - learning_rate: 1e-2-0.3 (log scale)
    - max_leaf_nodes: 15-255
    - min_samples_leaf: 5-100
    - l2_regularization: 1e-3-100 (log scale)
    - max_features: 0.3-1.0
    - max_bins: 64-255

