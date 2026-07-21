"""K-nearest-neighbors regression pipeline."""

from sklearn.neighbors import KNeighborsRegressor

from models.sklearn_cv import prepare_data, run_cv as run_sklearn_cv


def setup_model(n_neighbors, weights, p):
    return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)


def run_cv(X_train, y_train, model_hps, n_splits=5):
    return run_sklearn_cv(
        X_train,
        y_train,
        model_factory=lambda: setup_model(*model_hps),
        n_splits=n_splits,
    )


def knn_pipelined_model(dataset_name, n_neighbors, weights, p, n_splits=5):
    X_train, _, y_train, _ = prepare_data(dataset_name)
    return run_cv(
        X_train,
        y_train,
        model_hps=(n_neighbors, weights, p),
        n_splits=n_splits,
    )
