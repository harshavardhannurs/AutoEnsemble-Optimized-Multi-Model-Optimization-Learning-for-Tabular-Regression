import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.data_utils as du

def set_device():

    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    return device

def prepare_data(dataset_name, batch_size=64):
    data = du.get_combined_dataset(dataset_name)
    X, y = du.split_features_and_target(data)
    X_train, X_test, y_train, y_test = du.split_train_test(X, y, test_size=0.2)

    transformer_X, transformer_y = du.data_transformer(X_train, y_train)

    X_train = transformer_X.fit_transform(X_train)
    X_test = transformer_X.transform(X_test)

    y_train = transformer_y.fit_transform(y_train)
    y_test = transformer_y.transform(y_test)

    # Convert sparse matrices to dense arrays if needed
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    train_ldr, test_ldr = du.create_dataloaders(X_train, y_train, X_test, y_test, batch_size=batch_size)

    return train_ldr, test_ldr, X_train.shape[1]

def setup_model(
        dataset_name,
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

def train_model(
        model,
        optimizer,
        loss_function,
        train_ldr,
        test_ldr,
        num_epochs=200,
):

    device = set_device()

    model.to(device)

    train_losses = np.zeros((num_epochs,))
    test_losses = np.zeros((num_epochs,))

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

        train_losses[epochi] = np.mean(batch_losses)
        print(f'Train Loss: Epoch {epochi + 1}, Loss: {train_losses[epochi]:.4f}')

        model.eval()

        with torch.no_grad():

            X, y = next(iter(test_ldr))

            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)

            test_losses[epochi] = loss_function(y_hat, y).item()

            print(f'Train Loss: Epoch {epochi + 1}, Loss: {train_losses[epochi]:.4f} \n Test Loss: Epoch {epochi + 1}, Loss: {test_losses[epochi]:.4f}')

    return model, train_losses, test_losses

def plot_losses(train_losses, test_losses):

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss')

    ax2.plot(test_losses, label='Test Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Test Loss')

    plt.show()

def pipelined_model(
        dataset_name,
        batch_size,
        no_of_layers,
        no_of_units,
        learning_rate,
        weight_decay,
        batch_norm,
        num_epochs
):

    train_ldr, test_ldr, no_of_feats = prepare_data(dataset_name=dataset_name, batch_size=batch_size)

    model, optimizer, loss_function = setup_model(
        dataset_name,
        no_of_feats=no_of_feats,
        no_of_layers=no_of_layers,
        no_of_units=no_of_units,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_norm=batch_norm
    )

    model, train_losses, test_losses = train_model(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_ldr=train_ldr,
        test_ldr=test_ldr,
        num_epochs=num_epochs
    )
    # Use this trained model for predictions

    return model, train_losses, test_losses

if __name__ == '__main__':
    model, train_losses, test_losses = pipelined_model(
        dataset_name='brazilian_houses',
        batch_size=64,
        no_of_layers=3,
        no_of_units=128,
        learning_rate=0.001,
        weight_decay=0.001,
        batch_norm=False,
        num_epochs=500
    )

    plot_losses(train_losses, test_losses)

    avg_test_loss = np.mean(test_losses)

    print(f"Model training completed with the average test loss of {avg_test_loss:.4f}. You can now use the trained model for predictions.")

