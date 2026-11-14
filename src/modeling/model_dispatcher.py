# src/modeling/model_dispatcher.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .preprocessing import build_preprocessor
from . import config


def get_models():
    """
    Returns a dict of named pipelines.
    Each pipeline is unfitted — train.py will .fit() inside each fold.
    """
    models = {
        'logistic_regression': Pipeline([
            ('preprocess', build_preprocessor()),
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, random_state=config.SEED,
                                         class_weight='balanced', solver='lbfgs')),
        ]),
        'linear_svc': Pipeline([
            ('preprocess', build_preprocessor()),
            ('scale', StandardScaler()),
            ('model', LinearSVC(class_weight='balanced', random_state=config.SEED,
                                max_iter=10000, C=1.0)),
        ]),
        'xgboost': Pipeline([
            ('preprocess', build_preprocessor()),
            ('model', XGBClassifier(
                n_estimators=878,
                max_depth=8,
                learning_rate=0.028,
                min_child_weight=9,
                scale_pos_weight=3.5,
                subsample=0.8351,
                colsample_bytree=0.605,
                reg_alpha=3.62e-05,
                reg_lambda=2.58e-08,
                random_state=config.SEED,
                eval_metric='logloss',
            )),
        ]),
    }

    return models
