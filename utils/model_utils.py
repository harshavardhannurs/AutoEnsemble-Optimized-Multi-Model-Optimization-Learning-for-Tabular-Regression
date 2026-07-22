import numpy as np
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def set_device():

    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    return device

def plot_losses(train_loss_across_folds, val_loss_across_folds):

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(len(train_loss_across_folds)):
        ax1.plot(train_loss_across_folds[i], label=f'Fold {i + 1}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss')
    ax1.legend()

    for i in range(len(val_loss_across_folds)):
        ax2.plot(val_loss_across_folds[i], label=f'Fold {i + 1}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Loss')
    ax2.legend()

    fig.suptitle('Training and Validation Losses')

    plt.show()

def get_predictions(y, y_hat, transformer_y):

    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # inverse_transformers_ expects the arrays to be 2D

    y = transformer_y.named_transformers_['scaler'].inverse_transform(y)
    y_hat = transformer_y.named_transformers_['scaler'].inverse_transform(y_hat)

    return y.flatten(), y_hat.flatten()

def average_test_loss(X):

    return np.mean(X)

def get_r2_score(y, y_hat):

    r2 = r2_score(y, y_hat)
    return r2
