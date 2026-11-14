# src/modeling/tune.py
import argparse
import pandas as pd
import numpy as np
import optuna
import matplotlib
matplotlib.use('Agg')  # headless, no display needed   
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc

from . import config
from .preprocessing import INPUT_COLS, add_date_features, build_preprocessor
from .model_dispatcher import get_models


def parse_args():
    parser = argparse.ArgumentParser(description="Tune hyperparameters with Optuna")
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'logistic_regression', 'linear_svc'])
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--n-folds', type=int, default=config.NBR_SPLITS)
    return parser.parse_args()


# ─── Objective functions (one per model) ─────────────────────────────────────

def objective_xgboost(trial, train, n_folds):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':    trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample':    trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':    trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':   trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    from xgboost import XGBClassifier
    scores = []

    for fold in range(n_folds):
        tr = train[train['kfold'] != fold]
        va = train[train['kfold'] == fold]

        X_tr, y_tr = tr[INPUT_COLS], (tr['RainTomorrow'] == 'Yes').astype(int)
        X_va, y_va = va[INPUT_COLS], (va['RainTomorrow'] == 'Yes').astype(int)

        pipe = _build_pipe_xgb(params)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]

        precision, recall, _ = precision_recall_curve(y_va, proba)
        scores.append(auc(recall, precision))

    return np.mean(scores)


def objective_lr(trial, train, n_folds):
    params = {
        'C': trial.suggest_float('C', 0.001, 10.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),  # 0=LR, 1=Enet
    }

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    scores = []

    for fold in range(n_folds):
        tr = train[train['kfold'] != fold]
        va = train[train['kfold'] == fold]

        X_tr, y_tr = tr[INPUT_COLS], (tr['RainTomorrow'] == 'Yes').astype(int)
        X_va, y_va = va[INPUT_COLS], (va['RainTomorrow'] == 'Yes').astype(int)

        pipe = _build_pipe_lr(params)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_va)[:, 1]

        precision, recall, _ = precision_recall_curve(y_va, proba)
        scores.append(auc(recall, precision))

    return np.mean(scores)


def objective_svc(trial, train, n_folds):
    params = {
        'C': trial.suggest_float('C', 0.001, 10.0, log=True),
    }

    scores = []

    for fold in range(n_folds):
        tr = train[train['kfold'] != fold]
        va = train[train['kfold'] == fold]

        X_tr, y_tr = tr[INPUT_COLS], (tr['RainTomorrow'] == 'Yes').astype(int)
        X_va, y_va = va[INPUT_COLS], (va['RainTomorrow'] == 'Yes').astype(int)

        pipe = _build_pipe_svc(params)
        pipe.fit(X_tr, y_tr)
        score = pipe.decision_function(X_va)

        precision, recall, _ = precision_recall_curve(y_va, score)
        scores.append(auc(recall, precision))

    return np.mean(scores)


# ─── Pipeline builders (same as model_dispatcher but param-driven) ───────────

def _build_pipe_xgb(params):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    return Pipeline([
        ('preprocess', build_preprocessor()),
        ('model', XGBClassifier(
            **params,
            scale_pos_weight=3.5,
            random_state=config.SEED,
            eval_metric='logloss',
        )),
    ])


def _build_pipe_lr(params):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ('preprocess', build_preprocessor()),
        ('scale', StandardScaler()),
        ('model', LogisticRegression(
            C=params['C'],
            l1_ratio=params['l1_ratio'],
            penalty='elasticnet',
            solver='saga',
            max_iter=5000,
            random_state=config.SEED,
            class_weight='balanced',
        )),
    ])


def _build_pipe_svc(params):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    return Pipeline([
        ('preprocess', build_preprocessor()),
        ('scale', StandardScaler()),
        ('model', LinearSVC(
            C=params['C'],
            class_weight='balanced',
            random_state=config.SEED,
            max_iter=10000,
        )),
    ])


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    train = pd.read_csv(config.TRAIN_FOLDS_FILE, parse_dates=['Date'])
    train = add_date_features(train)

    # Suppress Optuna's per-trial logging (too verbose)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objectives = {
        'xgboost': objective_xgboost,
        'logistic_regression': objective_lr,
        'linear_svc': objective_svc,
    }

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(
        lambda trial: objectives[args.model](trial, train, args.n_folds),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Report
    print(f"\nBest PR-AUC: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save
    import json
    from pathlib import Path
    out = config.REPORTS_DIR / f"tuned_{args.model}.json"
    out.write_text(json.dumps({'best_value': study.best_value, 'best_params': study.best_params}, indent=2))
    print(f"\nSaved → {out}")

    # Importance plot
    try:
        import optuna.visualization.matplotlib as vis_mpl
        ax = vis_mpl.plot_param_importances(study)
        ax.figure.savefig(str(config.REPORTS_DIR / "figures" / f"importance_{args.model}.png"), dpi=150, bbox_inches='tight')
        plt.close(ax.figure)
        print(f"Importance plot saved.")
    except Exception as e:
        print(f"(No plot: {e})")


if __name__ == "__main__":
    main()   
