"""Shared cross-validation helpers for scikit-learn style regressors."""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from utils import data_utils as du
from utils import model_utils as mu


def prepare_data(dataset_name):
    data = du.get_combined_dataset(dataset_name)
    X, y = du.split_features_and_target(data)
    return du.split_train_test(X, y, test_size=0.2)


def run_cv(X_train, y_train, model_factory, n_splits=5):
    """Fit and score a regressor on consistently transformed CV folds."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_losses = np.zeros((n_splits, 1))
    validation_losses = np.zeros((n_splits, 1))
    r2_scores = np.zeros(n_splits)

    for fold, (train_indices, validation_indices) in enumerate(kfold.split(X_train)):
        X_train_fold = X_train.iloc[train_indices]
        X_validation_fold = X_train.iloc[validation_indices]
        y_train_fold = y_train.iloc[train_indices]
        y_validation_fold = y_train.iloc[validation_indices]

        transformer_X, transformer_y = du.data_transformer(X_train_fold, y_train_fold)
        X_train_fold = transformer_X.fit_transform(X_train_fold)
        X_validation_fold = transformer_X.transform(X_validation_fold)
        X_train_fold = du.to_dense(X_train_fold)
        X_validation_fold = du.to_dense(X_validation_fold)
        y_train_fold = transformer_y.fit_transform(y_train_fold)
        y_validation_fold = transformer_y.transform(y_validation_fold)

        model = model_factory()
        model.fit(X_train_fold, y_train_fold.ravel())

        train_predictions = model.predict(X_train_fold)
        validation_predictions = model.predict(X_validation_fold)
        train_losses[fold, 0] = mean_squared_error(y_train_fold, train_predictions)
        validation_losses[fold, 0] = mean_squared_error(
            y_validation_fold, validation_predictions
        )

        y_true, y_pred = mu.get_predictions_2(
            y_validation_fold,
            validation_predictions.reshape(-1, 1),
            transformer_y,
        )
        r2_scores[fold] = mu.get_r2_score(y_true, y_pred)

        print(
            f"CV Fold: {fold + 1}/{n_splits} -> "
            f"Train Loss: {train_losses[fold, 0]:.4f} || "
            f"Validation Loss: {validation_losses[fold, 0]:.4f} || "
            f"R2: {r2_scores[fold]:.4f}"
        )

    return train_losses, validation_losses, r2_scores
