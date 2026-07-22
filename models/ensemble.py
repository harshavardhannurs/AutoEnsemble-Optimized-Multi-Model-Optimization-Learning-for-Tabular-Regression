"""Stage-two weighted ensemble built from tuned model configurations."""

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor

from models.extra_trees import setup_model as setup_extra_trees
from models.ffn import setup_model as setup_ffn
from models.hgb import setup_model as setup_hgb
from models.knn import setup_model as setup_knn
from models.random_forest import setup_model as setup_random_forest
from models.sklearn_cv import prepare_data, run_cv as run_sklearn_cv
from models.xgboost import setup_model as setup_xgboost
from utils import data_utils as du
from utils import model_utils as mu


class FFNRegressor(RegressorMixin, BaseEstimator):
    """Minimal scikit-learn adapter around the existing PyTorch FFN."""

    def __init__(
        self,
        no_of_layers=2,
        no_of_units=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_norm=False,
        batch_size=64,
        num_epochs=100,
    ):
        self.no_of_layers = no_of_layers
        self.no_of_units = no_of_units
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_norm = batch_norm
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y):
        torch.manual_seed(42)
        self.device_ = mu.set_device()
        self.model_, optimizer, loss_function = setup_ffn(
            no_of_feats=X.shape[1],
            no_of_layers=self.no_of_layers,
            no_of_units=self.no_of_units,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_norm=self.batch_norm,
        )
        self.model_.to(self.device_)
        train_loader = du.get_dataloader(
            X,
            np.asarray(y).reshape(-1, 1),
            batch_size=min(self.batch_size, len(y)),
        )

        for epoch in range(self.num_epochs):
            self.model_.train()
            for features, target in train_loader:
                features = features.to(self.device_)
                target = target.to(self.device_)
                loss = loss_function(self.model_(features), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Ensemble FFN Epoch: {epoch + 1}/{self.num_epochs}")
        return self

    def predict(self, X):
        self.model_.eval()
        features = du.to_tensor(X).to(self.device_)
        with torch.no_grad():
            predictions = self.model_(features)
        return predictions.cpu().numpy().ravel()


def make_estimator(member):
    config = member["config"]
    model = member["name"]

    if model == "ffn":
        return FFNRegressor(**{key: value for key, value in config.items() if key != "model"})
    if model == "knn":
        return setup_knn(config["n_neighbors"], config["weights"], config["p"])
    if model == "random_forest":
        return setup_random_forest(
            config["n_estimators"],
            config["max_depth"],
            config["min_samples_leaf"],
            config["max_features"],
        )
    if model == "extra_trees":
        return setup_extra_trees(
            config["n_estimators"],
            config["max_depth"],
            config["min_samples_leaf"],
            config["max_features"],
        )
    if model == "xgboost":
        return setup_xgboost(
            config["n_estimators"],
            config["learning_rate"],
            config["max_depth"],
            config["subsample"],
            config["colsample_bytree"],
        )
    return setup_hgb(
        config["max_iter"],
        config["learning_rate"],
        config["max_leaf_nodes"],
        config["min_samples_leaf"],
        config["l2_regularization"],
    )


def setup_model(selected_models):
    return VotingRegressor(
        estimators=[(member["name"], make_estimator(member)) for member in selected_models],
        weights=[member["weight"] for member in selected_models],
        n_jobs=1,
    )


def ensemble_pipelined_model(dataset_name, selected_models, n_splits=5):
    X_train, _, y_train, _ = prepare_data(dataset_name)
    return run_sklearn_cv(
        X_train,
        y_train,
        model_factory=lambda: setup_model(selected_models),
        n_splits=n_splits,
    )
