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

def run_cv(
        X_train,
        y_train,
        n_estimators,
        n_splits=5,
        seed=42
):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

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

        train_size = X_train.shape[0]
        subsample_cap = 900 if train_size > 6000 else train_size

        model = TabPFNRegressor(
            n_estimators=n_estimators,  # recommended in the documentation
            ignore_pretraining_limits=True,
            inference_config={
                'SUBSAMPLE_SAMPLES': subsample_cap,
            },
            random_state=seed
        )

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
        n_estimators,
        n_splits=5,
        seed=42
):

    if dataset_name != du.DATASET_DICT[6]:
        X_train, _, y_train, _ = prepare_data(dataset_name, seed=seed)
    else:
        X_train, y_train = du.get_exam_train_data()

    r2_score_across_folds = run_cv(
        X_train=X_train,
        y_train=y_train,
        n_splits=n_splits,
        n_estimators=n_estimators,
        seed=seed
    )

    return r2_score_across_folds

def train(
        dataset_name,
        seed=42
):

    if dataset_name != du.DATASET_DICT[6]:
        X_train, X_test, y_train, _ = prepare_data(dataset_name, seed=seed)
    else:
        X_train, y_train = du.get_exam_train_data()
        X_test = du.get_exam_test_data()

    X_train = du.to_numpy(X_train)
    X_test = du.to_numpy(X_test)
    y_train = y_train.values.ravel()

    train_size = X_train.shape[0]
    subsample_cap = 900 if train_size > 6000 else train_size

    model = TabPFNRegressor(
        n_estimators=16, # recommended in the documentation
        ignore_pretraining_limits=True,
        inference_config= {
            'SUBSAMPLE_SAMPLES': subsample_cap,
        },
        random_state=seed
    )
    # To prevent OOM errors (during testing, TabPFN on MPS failed when training data size was approximately greater than 6000)

    print('Training the model...')
    model.fit(X_train, y_train)

    print('Evaluating the model on test set...')

    batch_size = 100
    y_hat_batch = []

    for i in range(0, len(X_test), batch_size):
        y_hat_batch.append(model.predict(X_test[i:i + batch_size]))
    # Split X_test into batches to avoid memory issues
    y_hat = np.concatenate(y_hat_batch, axis=0)

    return model, y_hat

if __name__=='__main__':
    pass
