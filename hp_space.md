# Hyperparameter Space

## Models
- XGBoost – XGBRegressor
- Random Forest – RandomForestRegressor
- KNN – KNeighborsRegressor
- Feed Forward Neural Network

## Hyperparameter Space
- XGBoost
    - n_estimators: 100-1000
    - learning_rate: 0.01–0.3 (log scale)
    - max_depth: 3-10
    - subsample: 0.5-1
    - colsample_bytree: 0.5-1
- Random Forest
    - n_estimators: 100-500
    - max_depth: 5-30
    - min_samples_leaf: 1-10
    - max_features: 0.5-1
- KNN
    - n_neighbors: 3-50
    - weights: uniform/distance
    - p: 1/2
- Feed Forward Neural Network
    - no_of_layers: 1-4
    - no_of_units: 32 - 512
    - learning_rate: 1e-5 - 1e-2 (log scale)
    - batch_size: 32 - 256
    - weight_decay: 1e-6 - 1e-3 (log scale)
    - batch_norm: True/False
    - num_epochs: 100-1000

