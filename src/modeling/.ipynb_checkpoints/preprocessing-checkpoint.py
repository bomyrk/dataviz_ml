# scr/modelling/preprocessing.py

# src/modeling/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CAP_COLS = ['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
            'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
LOG_COLS = ['Rainfall']
CAT_COLS = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']


def build_preprocessor():
    """Return a fitted-or-unfitted ColumnTransformer.
    Call .fit(X_train) before .transform(X_val/X_test)."""

    numeric_pipe = Pipeline([
        ('cap', QuantileTransformer(output_distribution='uniform', n_quantiles=1000, random_state=42)),
        # or: ('cap', FunctionTransformer(lambda X: X.clip(*get_caps(X))))
    ])

    log_pipe = Pipeline([
        ('log', FunctionTransformer(np.log1p)),
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, CAP_COLS),
        ('log', log_pipe, LOG_COLS),
        ('cat', cat_pipe, CAT_COLS),
    ])

    return preprocessor


def fit_and_transform(preprocessor, X_train, X_val, X_test):
    """Fit on train only, transform all."""
    preprocessor.fit(X_train)
    return (preprocessor.transform(X_train),
            preprocessor.transform(X_val),
            preprocessor.transform(X_test))   