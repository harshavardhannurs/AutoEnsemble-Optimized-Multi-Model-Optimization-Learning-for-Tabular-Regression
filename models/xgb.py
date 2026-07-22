import numpy as np

from sklearn.model_selection import KFold

import xgboost as xgb
from xgboost import XGBRegressor

import utils.data_utils as du
import utils.model_utils as mu

def prepare_data(dataset_name):

    data = du.get_combined_dataset(dataset_name)
    X, y = du.split_features_and_target(data)
    X_train, X_test, y_train, y_test = du.split_train_test(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

def setup_model(
        n_estimators,
        learning_rate,
        max_depth,
        subsample,
        colsample_bytree,
        reg_lambda
):
    xbgr = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        n_jobs=-1,
        random_state=42
    )

    return xbgr

def run_cv(
        X_train,
        y_train,
        model_hps,
        n_splits=10
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2_score_across_folds = np.zeros((n_splits,))

    n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda = model_hps

    for i, (train_indices, val_indices) in enumerate(kf.split(X_train)):

        X_train_fold = X_train.iloc[train_indices]
        X_val_fold = X_train.iloc[val_indices]
        y_train_fold = y_train.iloc[train_indices]
        y_val_fold = y_train.iloc[val_indices]

        transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False)

        X_train_fold = transformer_X.fit_transform(X_train_fold)
        X_val_fold = transformer_X.transform(X_val_fold)

        # No need to transform the target as random forests split features like: xj < threshold. Scaling only adds complexity

        X_train_fold = du.to_dense(X_train_fold)
        X_val_fold = du.to_dense(X_val_fold)
        y_train_fold = du.to_dense(y_train_fold).values.ravel()
        y_val_fold = du.to_dense(y_val_fold).values.ravel()

        model = setup_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda
        )

        model.fit(X_train_fold, y_train_fold)

        # Evaluate the model on the validation set
        y_hat = model.predict(X_val_fold)

        r2_score_across_folds[i] = mu.get_r2_score(y_val_fold, y_hat)

    return r2_score_across_folds

def pipelined_model(
        dataset_name,
        n_estimators,
        learning_rate,
        max_depth,
        subsample,
        colsample_bytree,
        reg_lambda,
        n_splits=10
):
    X_train, X_test, y_train, y_test = prepare_data(dataset_name)

    model_hps = (n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda)

    r2_score_across_folds = run_cv(
        X_train=X_train,
        y_train=y_train,
        model_hps=model_hps,
        n_splits=n_splits
    )

    return r2_score_across_folds

if __name__ == '__main__':
    r2_score_across_folds = pipelined_model(
        dataset_name='bike_sharing_demand',
        n_estimators=100,
        learning_rate=0.01,
        max_depth=10,
        subsample=1,
        colsample_bytree=1,
        reg_lambda=0.001,
    )

    print("The current hyperparameter configuration yields an average R2 Score of ",
          mu.average(r2_score_across_folds))


