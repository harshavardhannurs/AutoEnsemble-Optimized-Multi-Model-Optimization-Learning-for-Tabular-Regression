import numpy as np

from tabpfn import TabPFNRegressor

from sklearn.model_selection import KFold

import utils.data_utils as du
import utils.model_utils as mu

def prepare_data(dataset_name, seed=42):

    data = du.get_combined_dataset(dataset_name)
    X, y = du.split_features_and_target(data)
    X_train, X_test, y_train, y_test = du.split_train_test(X, y, test_size=0.2, seed=seed)

    return X_train, X_test, y_train, y_test

def setup_model(
        n_estimators,
        seed=42
):
    model = TabPFNRegressor(
        n_estimators=n_estimators,
        random_state=seed
    )
    return model

def run_cv(
        X_train,
        y_train,
        model_hps,
        n_splits=10,
        seed=42
):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    r2_score_across_folds = np.zeros((n_splits,))

    for i, (train_indices, val_indices) in enumerate(kf.split(X_train)):

        X_train_fold = X_train.iloc[train_indices]
        X_val_fold = X_train.iloc[val_indices]
        y_train_fold = y_train.iloc[train_indices]
        y_val_fold = y_train.iloc[val_indices]

        transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False, encode_categorical=False)
        # TabPFN natively handles missing values and feature scaling, one-hot-encoding is ineffective as per the documentation

        X_train_fold = transformer_X.fit_transform(X_train_fold)
        X_val_fold = transformer_X.transform(X_val_fold)

        X_train_fold = du.to_dense(X_train_fold)
        X_val_fold = du.to_dense(X_val_fold)
        y_train_fold = du.to_dense(y_train_fold).values.ravel()
        y_val_fold = du.to_dense(y_val_fold).values.ravel()

        n_estimators = model_hps

        model = setup_model(n_estimators=n_estimators, seed=seed)

        model.fit(X_train_fold, y_train_fold)

        # Evaluate the model on the validation set
        y_hat = model.predict(X_val_fold)

        r2_score_across_folds[i] = mu.get_r2_score(y_val_fold, y_hat)

        print(
            f"[CV][Split {i + 1}] "
            f"Model=TabPFN, "
            f"R2={r2_score_across_folds[i]:.4f}"
        )

    return r2_score_across_folds

def pipelined_model(
        dataset_name,
        n_estimators=10,
        n_splits=10,
        seed=42
):

    X_train, X_test, y_train, y_test = prepare_data(dataset_name, seed=seed)

    r2_score_across_folds = run_cv(
        X_train=X_train,
        y_train=y_train,
        model_hps=n_estimators,
        n_splits=n_splits,
        seed=seed
    )

    return r2_score_across_folds

if __name__=='__main__':

    r2_score_across_folds = pipelined_model(
        dataset_name='wine_quality',
        n_estimators=20,
        n_splits=10,
        seed=0
    )

    print("The current hyperparameter configuration for TabPFN yields an average R2 Score of ",
          mu.average(r2_score_across_folds))
