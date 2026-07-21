"""XGBoost regression pipeline."""

from xgboost import XGBRegressor

from models.sklearn_cv import prepare_data, run_cv as run_sklearn_cv


def setup_model(
    n_estimators,
    learning_rate,
    max_depth,
    subsample,
    colsample_bytree,
):
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def run_cv(X_train, y_train, model_hps, n_splits=5):
    return run_sklearn_cv(
        X_train,
        y_train,
        model_factory=lambda: setup_model(*model_hps),
        n_splits=n_splits,
    )


def xgboost_pipelined_model(
    dataset_name,
    n_estimators,
    learning_rate,
    max_depth,
    subsample,
    colsample_bytree,
    n_splits=5,
):
    X_train, _, y_train, _ = prepare_data(dataset_name)
    return run_cv(
        X_train,
        y_train,
        model_hps=(
            n_estimators,
            learning_rate,
            max_depth,
            subsample,
            colsample_bytree,
        ),
        n_splits=n_splits,
    )
