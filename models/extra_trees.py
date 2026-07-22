"""Extra-trees regression pipeline."""

from sklearn.ensemble import ExtraTreesRegressor

from models.sklearn_cv import prepare_data, run_cv as run_sklearn_cv


def setup_model(n_estimators, max_depth, min_samples_leaf, max_features):
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )


def run_cv(X_train, y_train, model_hps, n_splits=5):
    return run_sklearn_cv(
        X_train,
        y_train,
        model_factory=lambda: setup_model(*model_hps),
        n_splits=n_splits,
    )


def extra_trees_pipelined_model(
    dataset_name,
    n_estimators,
    max_depth,
    min_samples_leaf,
    max_features,
    n_splits=5,
):
    X_train, _, y_train, _ = prepare_data(dataset_name)
    return run_cv(
        X_train,
        y_train,
        model_hps=(n_estimators, max_depth, min_samples_leaf, max_features),
        n_splits=n_splits,
    )
