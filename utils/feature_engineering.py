"""Generic, target-independent feature engineering for tabular data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAugmenter(TransformerMixin, BaseEstimator):
    """Add schema-agnostic numeric features while retaining original columns.

    The transformer adds:

    * ``log1p`` features for sufficiently right-skewed, non-negative columns.
    * The number of missing numeric values in each row.
    * Row-wise numeric mean, standard deviation, minimum, and maximum.

    Columns eligible for log transformation are learned from the training data
    in :meth:`fit`, so validation and test data do not influence feature
    selection. The target is never used.

    Parameters
    ----------
    skew_threshold:
        Minimum training-set skewness required for a non-negative numeric
        column to receive a ``log1p`` feature.
    feature_prefix:
        Prefix used for generated column names.
    """

    def __init__(
        self,
        skew_threshold: float = 0.75,
        feature_prefix: str = "generic__",
    ) -> None:
        self.skew_threshold = skew_threshold
        self.feature_prefix = feature_prefix

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureAugmenter":
        """Learn numeric and log-transform column selections from ``X``."""
        frame = self._validate_frame(X)
        if self.skew_threshold < 0:
            raise ValueError("skew_threshold must be non-negative.")
        if not self.feature_prefix:
            raise ValueError("feature_prefix must not be empty.")

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.numeric_columns_ = self._numeric_columns(frame)
        self.log_columns_ = self._find_log_columns(
            frame, self.numeric_columns_
        )

        generated_names = self._generated_feature_names()
        collisions = set(frame.columns).intersection(generated_names)
        if collisions:
            names = ", ".join(sorted(map(str, collisions)))
            raise ValueError(
                f"Generated feature names collide with input columns: {names}"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` with fitted generic features appended."""
        self._check_is_fitted()
        frame = self._validate_frame(X)
        self._validate_input_columns(frame)

        result = frame.copy()
        numeric = frame.loc[:, self.numeric_columns_]

        for column in self.log_columns_:
            result[f"{self.feature_prefix}log1p__{column}"] = np.log1p(
                numeric[column]
            )

        result[f"{self.feature_prefix}numeric_missing_count"] = (
            numeric.isna().sum(axis=1)
        )
        result[f"{self.feature_prefix}row_mean"] = numeric.mean(axis=1)
        result[f"{self.feature_prefix}row_std"] = numeric.std(axis=1, ddof=0)
        result[f"{self.feature_prefix}row_min"] = numeric.min(axis=1)
        result[f"{self.feature_prefix}row_max"] = numeric.max(axis=1)
        return result

    def get_feature_names_out(
        self, input_features: Iterable[object] | None = None
    ) -> np.ndarray:
        """Return original and generated feature names in output order."""
        self._check_is_fitted()
        if input_features is None:
            original = list(self.feature_names_in_)
        else:
            original = list(input_features)
            if original != list(self.feature_names_in_):
                raise ValueError(
                    "input_features must match the columns seen during fit."
                )

        return np.asarray(
            original + self._generated_feature_names(), dtype=object
        )

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureAugmenter expects a pandas DataFrame.")
        if not X.columns.is_unique:
            raise ValueError("Input column names must be unique.")
        return X

    @staticmethod
    def _numeric_columns(frame: pd.DataFrame) -> list[object]:
        numeric = frame.select_dtypes(include=[np.number]).columns
        return list(numeric)

    def _find_log_columns(
        self, frame: pd.DataFrame, numeric_columns: list[object]
    ) -> list[object]:
        selected = []
        for column in numeric_columns:
            observed = frame[column].dropna()
            if observed.empty or not np.isfinite(observed).all():
                continue
            if observed.min() < 0 or observed.nunique() < 3:
                continue
            if observed.skew() >= self.skew_threshold:
                selected.append(column)
        return selected

    def _generated_feature_names(self) -> list[str]:
        log_names = [
            f"{self.feature_prefix}log1p__{column}"
            for column in self.log_columns_
        ]
        aggregate_names = [
            f"{self.feature_prefix}numeric_missing_count",
            f"{self.feature_prefix}row_mean",
            f"{self.feature_prefix}row_std",
            f"{self.feature_prefix}row_min",
            f"{self.feature_prefix}row_max",
        ]
        return log_names + aggregate_names

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError(
                "FeatureAugmenter must be fitted before it is used."
            )

    def _validate_input_columns(self, frame: pd.DataFrame) -> None:
        fitted_columns = list(self.feature_names_in_)
        if list(frame.columns) != fitted_columns:
            raise ValueError(
                "Input columns and order must match the data used during fit."
            )
