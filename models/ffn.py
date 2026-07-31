import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

import utils.data_utils as du
import utils.model_utils as mu


def setup_model(
        no_of_feats,
        no_of_layers,
        no_of_units,
        learning_rate,
        weight_decay,
        batch_norm=False,
        seed=42
):

    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    class FFN(nn.Module):
        def __init__(self, n_feats, n_layers, n_units):
            super().__init__()

            self.n_layers = n_layers
            self.n_units = n_units
            self.n_feats = n_feats
            self.layers = nn.ModuleDict()

            self.layers['input'] = nn.Linear(n_feats, n_units)

            if batch_norm:
                self.layers['bn_input'] = nn.BatchNorm1d(n_units)

            for i in range(self.n_layers):
                self.layers[f'fc{i+1}'] = nn.Linear(n_units, n_units)
                if batch_norm:
                    self.layers[f'bn_fc{i+1}'] = nn.BatchNorm1d(n_units)

            self.layers['output'] = nn.Linear(n_units, 1)

        def forward(self, x):

            if batch_norm:
                x = F.relu(self.layers['bn_input'](self.layers['input'](x)))
                for i in range(self.n_layers):
                    x = F.relu(self.layers[f'bn_fc{i + 1}'](self.layers[f'fc{i + 1}'](x)))
                x = self.layers['output'](x)
                return x

            x = F.relu(self.layers['input'](x))
            for i in range(self.n_layers):
                x = F.relu(self.layers[f'fc{i + 1}'](x))
            x = self.layers['output'](x)

            return x

    model = FFN(no_of_feats, no_of_layers, no_of_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_function = nn.MSELoss()

    return model, optimizer, loss_function

def run_cv(
        dataset_name,
        model_hps,
        batch_size=64,
        num_epochs=500,
        n_splits=None,
        seed=42
):

    device = mu.set_device()

    if n_splits is None:
        n_splits = du.get_num_folds(dataset_name)

    train_loss_across_folds = np.zeros((n_splits, num_epochs))
    val_loss_across_folds = np.zeros((n_splits, num_epochs))
    r2_score_across_folds = np.zeros((n_splits,))

    no_of_layers, no_of_units, learning_rate, weight_decay, batch_norm = model_hps

    for i in range(n_splits):
        fold_number = i + 1
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = du.get_fold_data(dataset_name, fold_number)

        # Handle cases where y_val_fold might be None (e.g., exam_dataset)
        if y_val_fold is None:
            # If no validation target, we can't compute R2 for this fold. 
            # This happens for exam_dataset where only X_test is provided in the fold.
            # However, for CV we usually need a validation set with targets.
            # If it's a single fold without y_test, CV isn't really possible in the traditional sense.
            print(f"[CV][Split {i + 1}] No validation target found. Skipping R2 calculation for this fold.")
            continue

        # Now apply the transformation
        transformer_X, transformer_y = du.data_transformer(X_train_fold, y_train_fold)

        X_train_fold = transformer_X.fit_transform(X_train_fold)
        X_val_fold = transformer_X.transform(X_val_fold)

        y_train_fold = transformer_y.fit_transform(y_train_fold)
        y_val_fold = transformer_y.transform(y_val_fold)

        X_train_fold = du.to_dense(X_train_fold)
        X_val_fold = du.to_dense(X_val_fold)
        y_train_fold = du.to_dense(y_train_fold).ravel()
        y_val_fold = du.to_dense(y_val_fold).ravel()

        no_of_feats = X_train_fold.shape[1]

        model, optimizer, loss_function = setup_model(
            no_of_feats=no_of_feats,
            no_of_layers=no_of_layers,
            no_of_units=no_of_units,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_norm=batch_norm,
            seed=seed
        )

        best_loss = np.inf
        best_weights = None

        model.to(device)

        train_ldr = du.get_dataloader(X_train_fold, y_train_fold, batch_size=batch_size)
        val_ldr = du.get_dataloader(X_val_fold, y_val_fold, is_test=True)

        train_loss_across_epochs = np.zeros((num_epochs,))
        val_loss_across_epochs = np.zeros((num_epochs,))

        for epochi in range(num_epochs):

            model.train()

            batch_losses = []

            for X, y in train_ldr:
                X = X.to(device)
                y = y.to(device)

                y_hat = model(X)

                loss = loss_function(y_hat, y)
                batch_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_across_epochs[epochi] = np.mean(batch_losses)

            model.eval()

            with torch.no_grad():

                X_val, y_val = next(iter(val_ldr))

                X_val = X_val.to(device)
                y_val = y_val.to(device)

                y_val_hat = model(X_val)

                val_loss_across_epochs[epochi] = loss_function(y_val_hat, y_val).item()

            if val_loss_across_epochs[epochi] < best_loss:
                best_loss = val_loss_across_epochs[epochi]
                best_weights = deepcopy(model.state_dict())

        model.load_state_dict(best_weights)

        model.eval()

        with torch.no_grad():

            X_val, y_val = next(iter(val_ldr))

            X_val = X_val.to(device)
            y_val = y_val.to(device)

            y_val_hat = model(X_val)

        y = mu.inverse_transform_target(y_val.cpu().numpy(), transformer_y)
        y_hat = mu.inverse_transform_target(y_val_hat.cpu().numpy(), transformer_y)

        r2_score_across_folds[i] = mu.get_r2_score(y, y_hat)

        train_loss_across_folds[i, :] = train_loss_across_epochs
        val_loss_across_folds[i, :] = val_loss_across_epochs

        print(
            f"[CV][Split {i + 1}] "
            f"Model=Feed Forward Neural Network, "
            f"R2={r2_score_across_folds[i]:.4f}"
        )

    return r2_score_across_folds

def pipelined_model(
        dataset_name,
        batch_size,
        no_of_layers,
        no_of_units,
        learning_rate,
        weight_decay,
        batch_norm,
        num_epochs,
        n_splits=None,
        seed=42
):

    r2_score_across_folds = run_cv(
        dataset_name=dataset_name,
        model_hps=(no_of_layers, no_of_units, learning_rate, weight_decay, batch_norm),
        batch_size=batch_size,
        num_epochs=num_epochs,
        n_splits=n_splits,
        seed=seed
    )

    return r2_score_across_folds

def train(
    dataset_name,
    model_hps,
    seed=42
):

    batch_size, learning_rate, no_of_layers, no_of_units, weight_decay, batch_norm, num_epochs = model_hps

    # Use all folds for final training/testing
    X_train, y_train, X_test, y_test = du.get_all_folds_data(dataset_name)

    transformer_X, transformer_y = du.data_transformer(X_train, y_train)
    X_train = du.to_dense(transformer_X.fit_transform(X_train))
    X_test = du.to_dense(transformer_X.transform(X_test))
    y_train = transformer_y.fit_transform(y_train)

    if y_test is not None:
        y_test = transformer_y.transform(y_test)
        y_test = mu.inverse_transform_target(y_test, transformer_y)
    else:
        y_test = np.array([])

    no_of_feats = X_train.shape[1]

    train_ldr = du.get_dataloader(X_train, y_train, batch_size=batch_size)
    X_test = du.to_tensor(X_test)

    model, optimizer, loss_function = setup_model(
        no_of_feats=no_of_feats,
        no_of_layers=no_of_layers,
        no_of_units=no_of_units,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_norm=batch_norm,
        seed=seed
    )

    device = mu.set_device()

    model.to(device)
    model.train()
    print('Training FFN...')
    for epochi in range(num_epochs):

        for X, y in train_ldr:
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)

            loss = loss_function(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epochi + 1) % 10 == 0:
            print(f'Epoch {epochi + 1}/{num_epochs} completed')

    # Now use the trained model to make predictions on the test set

    print('Evaluating FFN on test set...')
    model.eval()

    with torch.no_grad():
        X_test = X_test.to(device)

        y_hat = model(X_test)

    y_hat = mu.inverse_transform_target(y_hat.cpu().numpy(), transformer_y)

    return model, y_test, y_hat

if __name__ == '__main__':
    pass