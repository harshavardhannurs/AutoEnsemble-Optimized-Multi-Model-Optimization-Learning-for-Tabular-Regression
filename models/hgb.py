"""Histogram gradient-boosting regression pipeline."""

from sklearn.ensemble import HistGradientBoostingRegressor

from models.sklearn_cv import prepare_data, run_cv as run_sklearn_cv


def setup_model(
    max_iter,
    learning_rate,
    max_leaf_nodes,
    min_samples_leaf,
    l2_regularization,
):
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=42,
    )


def run_cv(X_train, y_train, model_hps, n_splits=5):
    return run_sklearn_cv(
        X_train,
        y_train,
        model_factory=lambda: setup_model(*model_hps),
        n_splits=n_splits,
    )


def hgb_pipelined_model(
    dataset_name,
    max_iter,
    learning_rate,
    max_leaf_nodes,
    min_samples_leaf,
    l2_regularization,
    n_splits=5,
):
    X_train, _, y_train, _ = prepare_data(dataset_name)
    return run_cv(
        X_train,
        y_train,
        model_hps=(
            max_iter,
            learning_rate,
            max_leaf_nodes,
            min_samples_leaf,
            l2_regularization,
        ),
        n_splits=n_splits,
    )
