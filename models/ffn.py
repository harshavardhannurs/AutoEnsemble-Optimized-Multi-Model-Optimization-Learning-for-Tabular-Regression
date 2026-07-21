import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold

import utils.data_utils as du
import utils.model_utils as mu

def setup_model(
        no_of_feats,
        no_of_layers,
        no_of_units,
        learning_rate,
        weight_decay,
        batch_norm=False
):

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

def prepare_data(dataset_name):

    data = du.get_combined_dataset(dataset_name)
    X, y = du.split_features_and_target(data)
    X_train, X_test, y_train, y_test = du.split_train_test(X, y, test_size=0.2)

    # if hasattr(X_train, "to_numpy"):
    #     X_train = X_train.to_numpy()
    # if hasattr(X_test, "to_numpy"):
    #     X_test = X_test.to_numpy()
    # if hasattr(y_train, "to_numpy"):
    #     y_train = y_train.to_numpy()
    # if hasattr(y_test, "to_numpy"):
    #     y_test = y_test.to_numpy()

    return X_train, X_test, y_train, y_test

def run_cv(
        X_train,
        y_train,
        model_hps,
        batch_size=64,
        num_epochs=500,
        n_splits=10
):

    device = mu.set_device()

    kf = KFold(n_splits=n_splits, shuffle=True)

    train_loss_across_folds = np.zeros((n_splits, num_epochs))
    val_loss_across_folds = np.zeros((n_splits, num_epochs))

    for i, (train_indices, val_indices) in enumerate(kf.split(X_train)):

        X_train_fold = X_train.iloc[train_indices]
        X_val_fold = X_train.iloc[val_indices]
        y_train_fold = y_train.iloc[train_indices]
        y_val_fold = y_train.iloc[val_indices]

        # Now apply the transformation
        transformer_X, transformer_y = du.data_transformer(X_train_fold, y_train_fold)

        X_train_fold = transformer_X.fit_transform(X_train_fold)
        X_val_fold = transformer_X.transform(X_val_fold)

        y_train_fold = transformer_y.fit_transform(y_train_fold)
        y_val_fold = transformer_y.transform(y_val_fold)

        # Convert sparse matrices to dense arrays if needed
        if hasattr(X_train_fold, "toarray"):
            X_train_fold = X_train_fold.toarray()
        if hasattr(X_val_fold, "toarray"):
            X_val_fold = X_val_fold.toarray()
        if hasattr(y_train_fold, "toarray"):
            y_train_fold = y_train_fold.toarray()
        if hasattr(y_val_fold, "toarray"):
            y_val_fold = y_val_fold.toarray()

        no_of_feats = X_train_fold.shape[1]
        no_of_layers, no_of_units, learning_rate, weight_decay, batch_norm = model_hps

        model, optimizer, loss_function = setup_model(
            no_of_feats=no_of_feats,
            no_of_layers=no_of_layers,
            no_of_units=no_of_units,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_norm=batch_norm
        )

        model.to(device)

        train_ldr = du.get_dataloader(X_train_fold, y_train_fold, batch_size=batch_size)
        val_ldr = du.get_dataloader(X_val_fold, y_val_fold, is_test=True)

        train_loss_across_epochs = np.zeros((num_epochs,))
        val_loss_across_epochs = np.zeros((num_epochs,))

        print(f'CV Fold: {i + 1}/{n_splits}')

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

            print(f'Epoch {epochi + 1}/{num_epochs} -> Train Loss: {train_loss_across_epochs[epochi]:.4f} || Validation Loss: {val_loss_across_epochs[epochi]:.4f}')

        train_loss_across_folds[i, :] = train_loss_across_epochs
        val_loss_across_folds[i, :] = val_loss_across_epochs

    return train_loss_across_folds, val_loss_across_folds

def pipelined_model(
        dataset_name,
        batch_size,
        no_of_layers,
        no_of_units,
        learning_rate,
        weight_decay,
        batch_norm,
        num_epochs,
        n_splits
):

    X_train, X_test, y_train, y_test  = prepare_data(dataset_name)

    train_loss_across_folds, val_loss_across_folds = run_cv(
        X_train=X_train,
        y_train=y_train,
        model_hps=(no_of_layers, no_of_units, learning_rate, weight_decay, batch_norm),
        batch_size=batch_size,
        num_epochs=num_epochs,
        n_splits=n_splits
    )

    return train_loss_across_folds, val_loss_across_folds

if __name__ == '__main__':
    train_loss_across_folds, val_loss_across_folds = pipelined_model(
        dataset_name='bike_sharing_demand',
        batch_size=64,
        no_of_layers=2,
        no_of_units=128,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_norm=False,
        num_epochs=50,
        n_splits=5
    )

    mu.plot_losses(train_loss_across_folds, val_loss_across_folds)