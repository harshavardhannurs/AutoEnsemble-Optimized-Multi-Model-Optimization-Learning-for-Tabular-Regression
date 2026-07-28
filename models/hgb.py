import numpy as np

from sklearn.model_selection import KFold

from sklearn.ensemble import HistGradientBoostingRegressor

import utils.data_utils as du
import utils.model_utils as mu

def prepare_data(dataset_name, seed=42):

    data = du.get_combined_dataset(dataset_name)
    X, y = du.split_features_and_target(data)
    X_train, X_test, y_train, y_test = du.split_train_test(X, y, test_size=0.2, seed=seed)

    return X_train, X_test, y_train, y_test

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
        X_train,
        y_train,
        model_hps,
        n_splits=10,
        seed=42
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    r2_score_across_folds = np.zeros((n_splits,))

    max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_reg, max_features, max_bins = model_hps

    for i, (train_indices, val_indices) in enumerate(kf.split(X_train)):

        X_train_fold = X_train.iloc[train_indices]
        X_val_fold = X_train.iloc[val_indices]
        y_train_fold = y_train.iloc[train_indices]
        y_val_fold = y_train.iloc[val_indices]

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
        n_splits=10,
        seed=42
):

    if dataset_name != du.DATASET_DICT[6]:
        X_train, _, y_train, _ = prepare_data(dataset_name, seed=seed)
    else:
        X_train, y_train = du.get_exam_train_data()

    model_hps = (max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_reg, max_features, max_bins)

    r2_score_across_folds = run_cv(
        X_train=X_train,
        y_train=y_train,
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
        X_train, X_test, y_train, _ = prepare_data(dataset_name, seed=seed)
    else:
        X_train, y_train = du.get_exam_train_data()
        X_test = du.get_exam_test_data()

    transformer_X, _ = du.data_transformer(X_train, y_train, scale_numeric=False, impute_missing=False)
    X_train = du.to_dense(transformer_X.fit_transform(X_train))
    X_test = du.to_dense(transformer_X.transform(X_test))
    y_train = y_train.values.ravel()

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

    print('Training the model...')
    model.fit(X_train, y_train)

    print('Evaluating the model on test set...')
    y_hat = model.predict(X_test)

    return model, y_hat

if __name__ == '__main__':
    pass

