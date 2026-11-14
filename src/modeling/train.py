# src/modeling/train.py
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score

from . import config
from .preprocessing import INPUT_COLS, add_date_features
from .model_dispatcher import get_models

def parse_args():
    parser = argparse.ArgumentParser(description="Train models on chronological folds")
    parser.add_argument('--models', nargs='+',
                        default=['logistic_regression', 'linear_svc', 'xgboost'],
                        help="Models to train (default: all)")
    parser.add_argument('--n-folds', type=int, default=config.NBR_SPLITS)
    parser.add_argument('--seed', type=int, default=config.SEED)
    return parser.parse_args()
    
def run_fold(model_name, pipe, X_tr, y_tr, X_va, y_va):
    """Fit on train portion, evaluate on validation portion."""
    pipe.fit(X_tr, y_tr) # training
    
    # evaluation
    if hasattr(pipe.named_steps['model'], 'predict_proba'):
        proba = pipe.predict_proba(X_va)[:, 1]  # for LR and XGB
    else:
        proba = pipe.decision_function(X_va) # continous

    precision, recall, _ = precision_recall_curve(y_va, proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_va, (proba >= 0.5).astype(int))

    return {'model': model_name, 'pr_auc': pr_auc, 'f1': f1}

def main():
    args = parse_args()
    
    # Load
    train = pd.read_csv(config.TRAIN_FOLDS_FILE, parse_dates=['Date'])
    val   = pd.read_csv(config.VAL_FILE, parse_dates=['Date'])

    train = add_date_features(train)
    val   = add_date_features(val)

    all_models = get_models()
    models = {k: v for k, v in all_models.items() if k in args.models}
    results = []
    
    # for each fold
    for fold in range(args.n_folds):
        tr = train[train['kfold'] != fold]
        va = train[train['kfold'] == fold]

        for name, pipe in models.items():
            X_tr, y_tr = tr[INPUT_COLS], (tr['RainTomorrow'] == 'Yes').astype(int)
            X_va, y_va = va[INPUT_COLS], (va['RainTomorrow'] == 'Yes').astype(int)

            r = run_fold(name, pipe, X_tr, y_tr, X_va, y_va)
            r['fold'] = fold
            results.append(r)
            print(f"Fold {fold} | {name:>20} | PR-AUC={r['pr_auc']:.4f} | F1={r['f1']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(config.RESULTS_FILE, index=False)

    # Summary: mean ± std across folds
    print("\n── Summary ──")
    print(results_df.groupby('model')[['pr_auc', 'f1']].agg(['mean', 'std']))

if __name__ == "__main__":
    main()
