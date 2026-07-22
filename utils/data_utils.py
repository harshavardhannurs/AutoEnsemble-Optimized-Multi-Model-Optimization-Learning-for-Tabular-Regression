from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def get_combined_dataset(dataset_name):

    """
    Combines all folds of a dataset into a single DataFrame.
    """

    utils_dir = Path(__file__).resolve().parent

    project_root = utils_dir.parent

    data_dir = project_root/'data'/dataset_name

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")

    folds = []

    for fold_dir in data_dir.iterdir():
        X_train = pd.read_parquet(fold_dir/'X_train.parquet')
        y_train = pd.read_parquet(fold_dir/'y_train.parquet')
        X_test = pd.read_parquet(fold_dir/'X_test.parquet')
        y_test = pd.read_parquet(fold_dir/'y_test.parquet')

        if isinstance(y_train, pd.Series):
            y_train = y_train.to_frame()
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_frame()

        if y_train.shape[1] == 1:
            y_train.columns = ['target']
        if y_test.shape[1] == 1:
            y_test.columns = ['target']

        train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        # `reset_index()` is used because pandas allows us to assign custom index values to rows. When we reset the index, the original custom indices are removed and replaced with the default integer index (`0, 1, 2, ...`). By default, pandas preserves the old index by storing it as a new column named `"index"`. Setting `drop=True` prevents this extra column from being created and discards the old index completely.

        fold_df = pd.concat([train_df, test_df], ignore_index=True)
        # ignore_index=True tells pandas to discard the original row indices when combining DataFrames and create a new default integer index (0, 1, 2, ...) for the resulting DataFrame. Without it, pandas preserves the original indices, which can lead to duplicate index values after operations like concat().
        folds.append(fold_df)
        # we can append any type of object to a python list (numerical, dictionary, custom objects, etc.). Also, it need not be a homogenous collection.

    dataset = pd.concat(folds, ignore_index=True)

    dataset = dataset.drop_duplicates().reset_index(drop=True)

    return dataset

def split_features_and_target(df, target_column='target'):

    """
    Splits the DataFrame into features (X) and target (y).
    """

    X = df.drop(columns=[target_column])
    y = df[target_column].to_frame(name=getattr(df[target_column], "name", "target"))

    return X, y

def split_train_test(X, y, test_size=0.2):

    """
    Split the data into train and test sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test

def to_numpy(x):
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    elif hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()
    return x

def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x

def to_tensor(x):

    """
    Converts a numpy array into a PyTorch tensor.
    """

    if hasattr(x, "toarray"):
        x = x.toarray()

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()
    elif isinstance(x, pd.DataFrame):
        return torch.from_numpy(x.to_numpy(copy=True)).float()
    elif isinstance(x, pd.Series):
        return torch.from_numpy(x.to_numpy(copy=True)).float()
    else:
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)}")

def get_dataloader(X, y, batch_size=64, is_test=False):

    """
    Creates dataloader out of two tensors of the same shape. Dataloaders enable splitting of datasets into mini-batches.
    """

    X, y = map(to_tensor, (X, y))

    y = y.reshape(-1, 1)

    dataset = TensorDataset(X, y)

    if not is_test:
        ldr = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    else:
        ldr = DataLoader(dataset, batch_size=dataset.tensors[0].shape[0])

    return ldr

def data_transformer(X, y, categorical_strategy='most_frequent', non_categorical_strategy='mean', scale_numeric=True, impute_missing=True):

    """
    This function creates a transformer object that scales numerical features and encodes categorical features along with imputing missing values.
    """

    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.to_list()
    # select_dtypes return a subset of the DataFrame’s columns based on the column dtypes.

    non_categorical_columns = X.select_dtypes(exclude=['object', 'category']).columns.to_list()

    cat_steps = []

    if impute_missing:
        cat_steps.append(('imputer', SimpleImputer(strategy=categorical_strategy)))
    cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
    categorical_pipeline = Pipeline(cat_steps)

    non_cat_steps = []
    if impute_missing:
        non_cat_steps.append(('imputer', SimpleImputer(strategy=non_categorical_strategy)))

    if scale_numeric:
        non_cat_steps.append(('scaler', StandardScaler()))
    non_categorical_pipeline = (
        Pipeline(non_cat_steps)
        if non_cat_steps
        else "passthrough"
    )

    transformer_X = ColumnTransformer([
        ('transform_categorical', categorical_pipeline, categorical_columns),
        ('transform_non_categorical', non_categorical_pipeline, non_categorical_columns)
    ], remainder='passthrough')

    transformer_y = ColumnTransformer([
        ('scaler', StandardScaler(), y.columns.to_list()),
    ])

    return transformer_X, transformer_y

def drop_na_rows(df):

    """
    Drop rows with missing values.
    """

    return df.dropna(how='any')