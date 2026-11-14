# src/modeling/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    
# ─── Column groups (derived from EDA) 

# Highly right-skewed → log1p
LOG_COLS = ['Rainfall', 'Evaporation']
# Moderately right-skewed → log1p 
LOG_COLS_2 = ['WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']
CONT_COLS = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm',
             'Pressure9am', 'Pressure3pm',
             'Humidity9am', 'Humidity3pm',
             'Sunshine']
ORDINAL_COLS = ['Cloud9am', 'Cloud3pm']
CAT_COLS = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
BINARY_COLS = ['RainToday']
DATE_COLS = ['Month', 'DayOfYear']

# Everything that goes into the model
INPUT_COLS = (LOG_COLS + LOG_COLS_2 + CONT_COLS + ORDINAL_COLS
              + CAT_COLS + BINARY_COLS + DATE_COLS)

def add_date_features(df):
    """Deterministic — safe to apply before or after split."""
    df = df.copy()
    df['Month']      = df['Date'].dt.month
    df['DayOfYear']  = df['Date'].dt.dayofyear
    return df  

def _encode_binary(X):
    return np.where(X == 'Yes', 1, np.where(X == 'No', 0, -1)).astype(int)
    
def build_preprocessor():
    """
    Returns an unfitted ColumnTransformer.
    Must be .fit() on train only, then .transform() on val/test.
    """
    # Skewed_pipe -> median impute + log1p
    skewed_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one')),
    ])

    # Symmetric continous -> median impute
    symmetric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    ordinal_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])
    
    # Nominal categorical -> 'Unknown' fill + one-hot
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    
    binary_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encode', FunctionTransformer(_encode_binary, validate=False, feature_names_out='one-to-one')),
    ])
    
    date_pipe = 'passthrough' # no preprocessing

    preprocessor = ColumnTransformer([
        ('skewed',    skewed_pipe,    LOG_COLS + LOG_COLS_2),
        ('symmetric', symmetric_pipe, CONT_COLS),
        ('ordinal',   ordinal_pipe,   ORDINAL_COLS),
        ('cat',       cat_pipe,       CAT_COLS),
        ('binary',    binary_pipe,    BINARY_COLS),
        ('date',      date_pipe,      DATE_COLS),
    ], remainder='drop')

    return preprocessor

