import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

import utils.data_utils as du
import utils.model_utils as mu

def setup_model(
        max_iter,
        learning_rate,
        max_leaf_nodes,
        min_samples_leaf,
        l2_reg,
        max_features,
        max_bins,
        seed=42
):
    hbgr = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_reg,
        max_features=max_features,
        max_bins=max_bins,
        random_state=seed
    )

    return hbgr

def run_cv(
        dataset_name,
        model_hps,
        n_splits=None,
        seed=42
):
    if n_splits is None:
        n_splits = du.get_num_folds(dataset_name)

    r2_score_across_folds = np.zeros((n_splits,))

    max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_reg, max_features, max_bins = model_hps

    for i in range(n_splits):
        fold_number = i + 1
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = du.get_fold_data(dataset_name, fold_number, seed)

        if y_val_fold is None:
            print(f"[CV][Split {i + 1}] No validation target found. Skipping R2 calculation for this fold.")
            continue

        transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False)

        X_train_fold = transformer_X.fit_transform(X_train_fold)
        X_val_fold = transformer_X.transform(X_val_fold)

        y_train_fold = y_train_fold.values.ravel()
        y_val_fold = y_val_fold.values.ravel()

        model = setup_model(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_reg=l2_reg,
            max_features=max_features,
            max_bins=max_bins,
            seed=seed
        )

        model.fit(X_train_fold, y_train_fold)

        y_hat = model.predict(X_val_fold)

        r2_score_across_folds[i] = mu.get_r2_score(y_val_fold, y_hat)

        print(
            f"[CV][Split {i + 1}] "
            f"Model=HGB, "
            f"R2={r2_score_across_folds[i]:.4f}"
        )

    return r2_score_across_folds

def pipelined_model(
        dataset_name,
        max_iter,
        learning_rate,
        max_leaf_nodes,
        min_samples_leaf,
        l2_reg,
        max_features,
        max_bins,
        n_splits=None,
        seed=42
):

    model_hps = (max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_reg, max_features, max_bins)

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

    max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_reg, max_features, max_bins = model_hps

    if dataset_name != du.DATASET_DICT[6]:
        X_train, y_train, X_test, y_test = du.split_train_test(
            du.split_features_and_target(du.get_combined_dataset(dataset_name)), seed)
    else:
        X_train, y_train, X_test, y_test = du.get_all_folds_data(dataset_name)

    if y_test is None:
        y_test = pd.DataFrame()

    transformer_X, _ = du.data_transformer(X_train, y_train, scale_numeric=False, impute_missing=False)
    X_train = du.to_dense(transformer_X.fit_transform(X_train))
    X_test = du.to_dense(transformer_X.transform(X_test))
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    model = setup_model(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_reg=l2_reg,
        max_features=max_features,
        max_bins=max_bins,
        seed=seed
    )

    print('Training Histogram Gradient Boost...')
    model.fit(X_train, y_train)

    print('Evaluating Histogram Gradient Boost on test set...')
    y_hat = model.predict(X_test)

    return model, y_test, y_hat

if __name__ == '__main__':
    pass

