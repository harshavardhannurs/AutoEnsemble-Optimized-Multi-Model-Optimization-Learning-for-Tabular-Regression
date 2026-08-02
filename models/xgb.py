import numpy as np
import pandas as pd

from xgboost import XGBRegressor

import utils.data_utils as du
import utils.model_utils as mu

def setup_model(
        n_estimators,
        learning_rate,
        max_depth,
        subsample,
        colsample_bytree,
        reg_lambda,
        seed=42
):
    xbgr = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        n_jobs=1,
        random_state=seed
    )

    return xbgr

def run_cv(
        dataset_name,
        model_hps,
        n_splits=None,
        seed=42
):
    if n_splits is None:
        n_splits = du.get_num_folds(dataset_name)

    r2_score_across_folds = np.zeros((n_splits,))

    n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda = model_hps

    for i in range(n_splits):
        fold_number = i + 1
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = du.get_fold_data(dataset_name, fold_number, seed)

        if y_val_fold is None:
            print(f"[CV][Split {i + 1}] No validation target found. Skipping R2 calculation for this fold.")
            continue

        transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False)

        X_train_fold = transformer_X.fit_transform(X_train_fold)
        X_val_fold = transformer_X.transform(X_val_fold)

        X_train_fold = du.to_dense(X_train_fold)
        X_val_fold = du.to_dense(X_val_fold)

        y_train_fold = y_train_fold.values.ravel()
        y_val_fold = y_val_fold.values.ravel()

        model = setup_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            seed=seed
        )

        model.fit(X_train_fold, y_train_fold)

        y_hat = model.predict(X_val_fold)

        r2_score_across_folds[i] = mu.get_r2_score(y_val_fold, y_hat)

        print(
            f"[CV][Split {i + 1}] "
            f"Model=XGBoost, "
            f"R2={r2_score_across_folds[i]:.4f}"
        )

    return r2_score_across_folds

def pipelined_model(
        dataset_name,
        n_estimators,
        learning_rate,
        max_depth,
        subsample,
        colsample_bytree,
        reg_lambda,
        n_splits=None,
        seed=42
):

    model_hps = (n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda)

    r2_score_across_folds = run_cv(
        dataset_name=dataset_name,
        model_hps=model_hps,
        n_splits=n_splits,
        seed=seed
    )

    return r2_score_across_folds

def train(
        dataset_name,
        model_hps,
        seed=42
):
    n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda = model_hps

    if dataset_name != du.DATASET_DICT[6]:
        X_train, X_test, y_train, y_test = du.split_train_test(
            *du.split_features_and_target(du.get_combined_dataset(dataset_name)), seed=seed)
    else:
        X_train, y_train, X_test, y_test = du.get_all_folds_data(dataset_name)

    if y_test is None:
        y_test = pd.DataFrame()

    transformer_X, _ = du.data_transformer(X_train, y_train, scale_numeric=False, impute_missing=False)
    X_train= du.to_dense(transformer_X.fit_transform(X_train))
    X_test= du.to_dense(transformer_X.transform(X_test))
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    model = setup_model(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        seed=seed
    )

    print('Training XGBoost...')
    model.fit(X_train, y_train)

    print('Evaluating XGBoost on test set...')
    y_hat = model.predict(X_test)

    return model, y_test, y_hat

if __name__ == '__main__':
    pass

