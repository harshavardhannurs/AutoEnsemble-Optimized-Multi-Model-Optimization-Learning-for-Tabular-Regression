import numpy as np

from tabpfn import TabPFNRegressor

import utils.data_utils as du
import utils.model_utils as mu

def prepare_data(dataset_name, seed=42):

    data = du.get_combined_dataset(dataset_name)
    X, y = du.split_features_and_target(data)
    X_train, X_test, y_train, y_test = du.split_train_test(X, y, test_size=0.2, seed=seed)

    return X_train, X_test, y_train, y_test

def train(
        dataset_name,
        seed=42
):

    X_train, X_test, y_train, y_test = prepare_data(dataset_name, seed=seed)
    X_train = du.to_numpy(X_train)
    X_test = du.to_numpy(X_test)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    train_size = X_train.shape[0]
    subsample_cap = 900 if train_size > 6000 else train_size

    model = TabPFNRegressor(
        n_estimators=16, # recommended in the documentation
        ignore_pretraining_limits=True,
        inference_config={
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

    r2_score = mu.get_r2_score(y_test, y_hat)

    return model, y_test, y_hat, r2_score

if __name__=='__main__':
    pass
