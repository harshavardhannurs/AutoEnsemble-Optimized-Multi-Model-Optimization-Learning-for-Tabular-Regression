from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

DATASET_DICT = {1: 'bike_sharing_demand', 2: 'brazilian_houses', 3: 'superconductivity', 4: 'wine_quality', 5: 'yprop_4_1', 6: 'exam_dataset'}

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

    return eng_features(dataset_name, dataset)

def split_features_and_target(df, target_column='target'):

    """
    Splits the DataFrame into features (X) and target (y).
    """

    X = df.drop(columns=[target_column])
    y = df[target_column].to_frame(name=getattr(df[target_column], "name", "target"))

    return X, y

def split_train_test(X, y, test_size=0.2, seed=42):

    """
    Split the data into train and test sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=seed)

    return X_train, X_test, y_train, y_test

def get_num_folds(dataset_name):

    """
    Returns the number of fold directories for a dataset.
    """

    if dataset_name == DATASET_DICT[6]:
        return 10

    utils_dir = Path(__file__).resolve().parent
    project_root = utils_dir.parent
    data_dir = project_root / 'data' / dataset_name

    if not data_dir.exists():
        return 0

    num_folds = 0
    for fold_dir in data_dir.iterdir():
        if fold_dir.is_dir() and fold_dir.name.isdigit():
            num_folds += 1
    return num_folds

def get_fold_data(dataset_name, fold_number, seed=42):
    
    """
    Returns X_train, y_train, X_val, y_val for a specific fold of a dataset.
    """
    
    if dataset_name == DATASET_DICT[6]:
        # Special handling for exam_dataset to split fold 1 into 10 folds
        # fold_number here will be 1 to 10
        utils_dir = Path(__file__).resolve().parent
        project_root = utils_dir.parent
        base_fold_dir = project_root / 'data' / dataset_name / '1'
        
        X_train_full = pd.read_parquet(base_fold_dir/'X_train.parquet').reset_index(drop=True)
        y_train_full = pd.read_parquet(base_fold_dir/'y_train.parquet').reset_index(drop=True)
        X_test_all = pd.read_parquet(base_fold_dir/'X_test.parquet').reset_index(drop=True)
        
        # We use a fixed seed for KFold to ensure consistency across calls
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        folds = list(kf.split(X_train_full))
        
        train_idx, val_idx = folds[fold_number - 1]
        
        X_train = X_train_full.iloc[train_idx].reset_index(drop=True)
        y_train = y_train_full.iloc[train_idx].reset_index(drop=True)
        X_val = X_train_full.iloc[val_idx].reset_index(drop=True)
        y_val = y_train_full.iloc[val_idx].reset_index(drop=True)
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_frame()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_frame()
            
        if y_train.shape[1] == 1:
            y_train.columns = ['target']
        if y_val.shape[1] == 1:
            y_val.columns = ['target']
            
        # Apply feature engineering
        X_train = eng_features(dataset_name, X_train)
        X_val = eng_features(dataset_name, X_val)
        
        # We also need the test data for prediction in some cases
        X_test = eng_features(dataset_name, X_test_all)
        
        return X_train, y_train, X_val, y_val

    utils_dir = Path(__file__).resolve().parent
    project_root = utils_dir.parent
    fold_dir = project_root / 'data' / dataset_name / str(fold_number)

    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory does not exist: {fold_dir}")

    X_train = pd.read_parquet(fold_dir/'X_train.parquet').reset_index(drop=True)
    y_train = pd.read_parquet(fold_dir/'y_train.parquet').reset_index(drop=True)
    X_val = pd.read_parquet(fold_dir/'X_test.parquet').reset_index(drop=True)

    y_val_path = fold_dir/'y_test.parquet'
    if y_val_path.exists():
        y_val = pd.read_parquet(y_val_path).reset_index(drop=True)
    else:
        y_val = None

    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()
    if y_val is not None and isinstance(y_val, pd.Series):
        y_val = y_val.to_frame()

    if y_train.shape[1] == 1:
        y_train.columns = ['target']
    if y_val is not None and y_val.shape[1] == 1:
        y_val.columns = ['target']

    # Apply feature engineering
    X_train = eng_features(dataset_name, X_train)
    X_val = eng_features(dataset_name, X_val)

    return X_train, y_train, X_val, y_val

def get_all_folds_data(dataset_name):

    """
    Returns X_train, y_train, X_test, y_test combined from all folds.
    """
    
    if dataset_name == DATASET_DICT[6]:
        # Special handling for exam_dataset
        utils_dir = Path(__file__).resolve().parent
        project_root = utils_dir.parent
        fold_dir = project_root / 'data' / dataset_name / '1'
        
        X_train = pd.read_parquet(fold_dir/'X_train.parquet').reset_index(drop=True)
        y_train = pd.read_parquet(fold_dir/'y_train.parquet').reset_index(drop=True)
        X_test = pd.read_parquet(fold_dir/'X_test.parquet').reset_index(drop=True)
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_frame()
        if y_train.shape[1] == 1:
            y_train.columns = ['target']
            
        X_train = eng_features(dataset_name, X_train)
        X_test = eng_features(dataset_name, X_test)
        
        return X_train, y_train, X_test, pd.DataFrame(columns=y_train.columns)

    # For bike_sharing_demand and others, since the dataset is readily split into training and validation sets
    # for each fold, we use the union of all X_test as the 'full' dataset
    # to avoid redundant training and leakage during OOF evaluation.

    num_folds = get_num_folds(dataset_name)
    X_test_list, y_test_list = [], []
    
    for fold in range(1, num_folds + 1):
        X_tr, y_tr, X_te, y_te = get_fold_data(dataset_name, fold)
        # In these datasets, the union of X_test across all folds 
        # constitutes the complete training dataset.
        X_test_list.append(X_te)
        if y_te is not None:
            y_test_list.append(y_te)
            
    X_train_full = pd.concat(X_test_list, axis=0).reset_index(drop=True)
    if y_test_list:
        y_train_full = pd.concat(y_test_list, axis=0).reset_index(drop=True)
    else:
        y_train_full = None

    return X_train_full, y_train_full, None, None

def get_oof_data(dataset_name, val_fold_number, seed):

    """
    Returns X_train, y_train and X_val, y_val for a specific fold.
    For pre-split datasets, each fold already contains X_train and X_test (validation).
    For exam_dataset, we use the 10-fold split.
    """

    if dataset_name == DATASET_DICT[6]:
        # For exam_dataset, get_fold_data already does the splitting
        return get_fold_data(dataset_name, val_fold_number, seed)

    return get_fold_data(dataset_name, val_fold_number, seed)


def eng_features(dataset_name, df):

    """
    Returns a DataFrame with generic and dataset-specific engineered features.
    """

    df = df.copy()

    # Generic engineering features

    target_column = None
    if 'target' in df.columns:
        target_column = df['target']
        df = df.drop(columns=['target'])

    df['mean'] = df.mean(axis=1, numeric_only=True)
    df['std'] = df.std(axis=1, numeric_only=True)
    df['min'] = df.min(axis=1, numeric_only=True)
    df['max'] = df.max(axis=1, numeric_only=True)

    # Dataset-specific engineering features

    if dataset_name == DATASET_DICT[1]:

        # bike_sharing_demand
        df['wind_chill_effect'] = df['windspeed'] * df['temp']
        df['temp_difference'] = df['feel_temp'] - df['temp']

    elif dataset_name == DATASET_DICT[2]:

        # brazilian_houses
        df['rooms_per_area'] = df['rooms'] / df['area'].replace(0, np.nan)
        df['bathrooms_per_area'] = df['bathroom'] / df['area'].replace(0, np.nan)
        df['parking_per_area'] = df['parking_spaces'] / df['area'].replace(0, np.nan)
        df['hoa_per_area(BRL)'] = df['hoa_(BRL)'] / df['area'].replace(0, np.nan)
        df['tax_per_area(BRL)'] = df['property_tax_(BRL)'] / df['area'].replace(0, np.nan)

    elif dataset_name == DATASET_DICT[3]:

        # superconductivity
        df['atomic_mass_shift'] = df['wtd_mean_atomic_mass'] - df['mean_atomic_mass']
        df['fie_shift'] = df['wtd_mean_fie'] - df['mean_fie']
        df['radius_shift'] = df['wtd_mean_atomic_radius'] - df['mean_atomic_radius']
        df['density_shift'] = df['wtd_mean_Density'] - df['mean_Density']
        df['valence_shift'] = df['wtd_mean_Valence'] - df['mean_Valence']
        df['atomic_mass_variability'] = df['std_atomic_mass'] / df['mean_atomic_mass'].replace(0, np.nan)
        df['radius_variability'] = df['std_atomic_radius'] / df['mean_atomic_radius'].replace(0, np.nan)
        df['density_variability'] = df['std_Density'] / df['mean_Density'].replace(0, np.nan)

    elif dataset_name == DATASET_DICT[4]:

        # wine_quality
        df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']
        df['sugar_alcohol_ratio'] = df['residual_sugar'] / df['alcohol'].replace(0, np.nan)
        df['free_sulfur_dioxide_ratio'] = df['free_sulfur_dioxide'] / df['total_sulfur_dioxide'].replace(0, np.nan)
        df['density_alcohol_interaction'] = df['density'] * df['alcohol']

    elif dataset_name == DATASET_DICT[6]:

        # exam_dataset
        df["living_area_ratio"] = df["TOT_LVG_AREA"] / df["LND_SQFOOT"].replace(0, np.nan)
        df["unused_land"] = df["LND_SQFOOT"] - df["TOT_LVG_AREA"]
        df["total_accessibility"] = (
                df["RAIL_DIST"] +
                df["HWY_DIST"] +
                df["CNTR_DIST"]
        )

        c = df[[
            "RAIL_DIST",
            "OCEAN_DIST",
            "WATER_DIST",
            "CNTR_DIST",
            "SUBCNTR_DI",
            "HWY_DIST",
        ]]

        df["avg_distance"] = c.mean(axis=1)
        df["closest_amenity"] = c.min(axis=1)
        df["furthest_amenity"] = c.max(axis=1)
        df["lat_long_interaction"] = (
                df["LATITUDE"] * df["LONGITUDE"]
        )
        df["age_squared"] = df["age"] ** 2
        df["quality_age"] = (
                df["structure_quality"] /
                (df["age"] + 1)
        )
        df["quality_area"] = (
                df["structure_quality"] *
                df["TOT_LVG_AREA"]
        )

    # For yprop_4_1, we couldn't add any new columns due to non-meaningful feature names.

    if target_column is not None:
        df = pd.concat([df, target_column], axis=1)

    return df

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

def data_transformer(X, y, categorical_strategy='most_frequent', non_categorical_strategy='mean', scale_numeric=True, impute_missing=True, encode_categorical=True):

    """
    Creates preprocessing transformers for the features and target.

    Categorical features can be optionally imputed and one-hot encoded.
    Numerical features can be optionally imputed and standardized.
    """

    categorical_columns = X.select_dtypes(include=['object', 'category', 'string']).columns.to_list()
    # select_dtypes return a subset of the DataFrame’s columns based on the column dtypes.

    non_categorical_columns = X.select_dtypes(exclude=['object', 'category']).columns.to_list()

    cat_steps = []

    if impute_missing:
        cat_steps.append(('imputer', SimpleImputer(strategy=categorical_strategy)))
    if encode_categorical:
        cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    categorical_pipeline = (
        Pipeline(cat_steps)
        if cat_steps
        else "passthrough"
    )

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