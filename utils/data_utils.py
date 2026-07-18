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

def combine_folds(dataset_name):

    """
    Combines all folds of a dataset into a single DataFrame.
    """

    data_dir = Path(f'../data/{dataset_name}')

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

    combined_df = pd.concat(folds, ignore_index=True)

    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    return combined_df

def split_features_and_target(df, target_column='target'):

    """
    Splits the DataFrame into features (X) and target (y).
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

def split_train_test(X, y, test_size=0.2):

    """
    Split the data into train and test sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    return X_train, X_test, y_train, y_test

def to_tensor(x):

    """
    Converts a numpy array into a PyTorch tensor.
    """

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)}")

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):

    """
    Creates dataloaders for training and testing. Dataloaders enable splitting of datasets into mini-batches.
    """

    X_train, y_train, X_test, y_test = map(to_tensor, (X_train, y_train, X_test, y_test))

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_ldr = DataLoader(train_data, batch_size=batch_size, drop_last=True)
    test_ldr = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

    return train_ldr, test_ldr

def data_transformer(df, categorical_strategy='most_frequent', non_categorical_strategy='mean'):

    """
    This function creates a transformer object that scales numerical features and encodes categorical features along with imputing missing values.
    """

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.to_list()
    # select_dtypes return a subset of the DataFrame’s columns based on the column dtypes.

    non_categorical_columns = df.select_dtypes(exclude=['object', 'category']).columns.to_list()

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=categorical_strategy)),
        ('encoder', OneHotEncoder())
    ])

    non_categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=non_categorical_strategy)),
        ('scaler', StandardScaler())
    ])

    transformer = ColumnTransformer([
        ('transform_categorical', categorical_pipeline, categorical_columns),
        ('transform_non_categorical', non_categorical_pipeline, non_categorical_columns)
    ], remainder='passthrough')

    return transformer # remember to convert sparse matrix to dense if needed using .toarray() after transformation.

def drop_na_rows(df):

    """
    Drop rows with missing values.
    """

    return df.dropna(how='any')

def test_fns():

    datasets = [
        'brazilian_houses',
        'bike_sharing_demand',
        'superconductivity',
        'wine_quality',
        'yprop_4_1'
    ]

    for d in datasets:

        x = combine_folds('brazilian_houses')
        print(pd.isna(x).sum().sum())

test_fns()
