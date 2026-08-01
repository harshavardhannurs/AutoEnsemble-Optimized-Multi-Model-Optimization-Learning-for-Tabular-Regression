import os
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn import TabPFNRegressor

import utils.data_utils as du
import utils.model_utils as mu


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABPFN_CHECKPOINT = PROJECT_ROOT / 'tabpfn-v3-regressor-v3_default.ckpt'


def get_checkpoint_path():
    """Return the local TabPFN checkpoint path and prevent online fallback."""

    configured_path = os.environ.get('AUTOML_TABPFN_MODEL_PATH')
    checkpoint_path = Path(configured_path).expanduser() if configured_path else DEFAULT_TABPFN_CHECKPOINT

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'Local TabPFN checkpoint not found at '
            f"'{checkpoint_path}'. Place tabpfn-v3-regressor-v3_default.ckpt "
            'in the project root or set AUTOML_TABPFN_MODEL_PATH.'
        )

    return checkpoint_path


def create_regressor(n_estimators, subsample_cap, seed=42):
    """Create TabPFN using only the checkpoint stored on the local filesystem."""

    checkpoint_path = get_checkpoint_path()
    print(f'Using local TabPFN checkpoint: {checkpoint_path}')

    return TabPFNRegressor(
        model_path=checkpoint_path,
        device='cpu',
        n_estimators=int(n_estimators),
        ignore_pretraining_limits=True,
        inference_config={
            'SUBSAMPLE_SAMPLES': int(subsample_cap),
        },
        random_state=seed
    )

def run_cv(
        dataset_name,
        n_estimators,
        n_splits=None,
        seed=42
):

    if n_splits is None:
        n_splits = du.get_num_folds(dataset_name)

    r2_score_across_folds = np.zeros((n_splits,))

    for i in range(n_splits):
        fold_number = i + 1
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = du.get_fold_data(dataset_name, fold_number, seed)

        if y_val_fold is None:
            print(f"[CV][Split {i + 1}] No validation target found. Skipping R2 calculation for this fold.")
            continue

        transformer_X, _ = du.data_transformer(X_train_fold, y_train_fold, scale_numeric=False, impute_missing=False, encode_categorical=False)
        # TabPFN natively handles missing values and feature scaling, one-hot-encoding is ineffective as per the documentation

        X_train_fold = transformer_X.fit_transform(X_train_fold)
        X_val_fold = transformer_X.transform(X_val_fold)

        X_train_fold = du.to_dense(X_train_fold)
        X_val_fold = du.to_dense(X_val_fold)
        y_train_fold = du.to_dense(y_train_fold).values.ravel()
        y_val_fold = du.to_dense(y_val_fold).values.ravel()

        train_size = X_train_fold.shape[0]
        subsample_cap = 900 if train_size > 6000 else train_size

        model = create_regressor(
            n_estimators=n_estimators,
            subsample_cap=subsample_cap,
            seed=seed
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
        n_splits=None,
        seed=42
):

    r2_score_across_folds = run_cv(
        dataset_name=dataset_name,
        n_splits=n_splits,
        n_estimators=n_estimators,
        seed=seed
    )

    return r2_score_across_folds

def train(
        dataset_name,
        model_hps,
        seed=42
):

    if dataset_name != du.DATASET_DICT[6]:
        X_train, X_test, y_train, y_test = du.split_train_test(
            *du.split_features_and_target(du.get_combined_dataset(dataset_name)), seed)
    else:
        X_train, y_train, X_test, y_test = du.get_all_folds_data(dataset_name)

    if y_test is None:
        y_test = pd.DataFrame()

    X_train = du.to_numpy(X_train)
    X_test = du.to_numpy(X_test)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    train_size = X_train.shape[0]
    subsample_cap = 900 if train_size > 6000 else train_size

    if isinstance(model_hps, (tuple, list)) and len(model_hps) > 0:
        n_estimators = model_hps[0]
    elif isinstance(model_hps, (int, float)):
        n_estimators = int(model_hps)
    else:
        n_estimators = 8 # Default

    model = create_regressor(
        n_estimators=n_estimators,
        subsample_cap=subsample_cap,
        seed=seed
    )
    # To prevent OOM errors (during testing, TabPFN on MPS failed when training data size was approximately greater than 6000)

    print('Training TabPFN...')
    model.fit(X_train, y_train)

    print('Evaluating TabPFN on test set...')

    batch_size = 100
    y_hat_batch = []

    for i in range(0, len(X_test), batch_size):
        y_hat_batch.append(model.predict(X_test[i:i + batch_size]))
    # Split X_test into batches to avoid memory issues
    y_hat = np.concatenate(y_hat_batch, axis=0)

    return model, y_test, y_hat

if __name__=='__main__':
    pass
