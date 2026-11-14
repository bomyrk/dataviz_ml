# src/modeling/evaluate.py
import joblib
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from . import config
from .preprocessing import INPUT_COLS, add_date_features
from .model_dispatcher import get_models

from sklearn.metrics import (precision_recall_curve, auc, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix — {model_name} (Test)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.yticks([0, 1], ['No', 'Yes'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_{model_name}.png', dpi=150)
    plt.close()
    print(f"  Confusion matrix → {save_dir / f'confusion_matrix_{model_name}.png'}")


def plot_roc_curve(y_true, proba, model_name, save_dir):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)

    prec, rec, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axes[0].plot(fpr, tpr, color='#e74c3c', linewidth=2,
                 label=f'ROC (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC Curve — {model_name} (Test)')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)

    # PR
    axes[1].plot(rec, prec, color='#2980b9', linewidth=2,
                 label=f'PR (AUC = {pr_auc:.4f})')
    axes[1].axhline(y=y_true.mean(), color='grey', linestyle='--',
                    label=f'Baseline ({y_true.mean():.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve — {model_name} (Test)')
    axes[1].legend(loc='lower left')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'roc_pr_{model_name}.png', dpi=150)
    plt.close()
    print(f"  ROC/PR curves → {save_dir / f'roc_pr_{model_name}.png'}")


def plot_feature_importance(pipe, model_name, save_dir, top_n=15):
    """Extract feature names from the fitted pipeline and plot top-n importance."""
    model = pipe.named_steps['model']

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("  (No feature importance available)")
        return

    # Reconstruct feature names from the fitted ColumnTransformer
    pre = pipe.named_steps['preprocess']
    feature_names = pre.get_feature_names_out()

    # Pair and sort
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(8, 7))
    plt.barh(range(len(imp_df)), imp_df['importance'].values, color='#2c3e50')
    plt.yticks(range(len(imp_df)), imp_df['feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features — {model_name} (Test)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_dir / f'importance_{model_name}.png', dpi=150)
    plt.close()
    print(f"  Feature importance → {save_dir / f'importance_{model_name}.png'}")
    print(imp_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='xgboost')
    args = parser.parse_args()

    # Load
    train = pd.read_csv(config.TRAIN_FOLDS_FILE, parse_dates=['Date'])
    val   = pd.read_csv(config.VAL_FILE, parse_dates=['Date'])
    test  = pd.read_csv(config.TEST_FILE, parse_dates=['Date'])

    train = add_date_features(train)
    val   = add_date_features(val)
    test  = add_date_features(test)

    # Combine train + val for final training
    train_val = pd.concat([train, val], ignore_index=True)

    # Get the pipeline
    models = get_models()
    pipe = models[args.model]

    # Fit on train+val
    X_fit = train_val[INPUT_COLS]
    y_fit = (train_val['RainTomorrow'] == 'Yes').astype(int)
    pipe.fit(X_fit, y_fit)

    # Evaluate on test (ONE TIME ONLY)
    X_test = test[INPUT_COLS]
    y_test = (test['RainTomorrow'] == 'Yes').astype(int)

    if hasattr(pipe.named_steps['model'], 'predict_proba'):
        proba = pipe.predict_proba(X_test)[:, 1]
    else:
        proba = pipe.decision_function(X_test)

    y_pred = (proba >= 0.5).astype(int)

    # Metrics
    prec, rec, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(rec, prec)
    roc_auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, y_pred)

    print(f"\nTEST | {args.model}")
    print(f"  PR-AUC  = {pr_auc:.4f}")
    print(f"  ROC-AUC = {roc_auc:.4f}")
    print(f"  F1      = {f1:.4f}")

    # Save metrics
    metrics = {'model': args.model, 'pr_auc': pr_auc, 'roc_auc': roc_auc, 'f1': f1}
    pd.DataFrame([metrics]).to_csv(config.TEST_METRICS, index=False)

    # Plots
    save_dir = config.REPORTS_DIR / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_test, y_pred, args.model, save_dir)
    plot_roc_curve(y_test, proba, args.model, save_dir)
    plot_feature_importance(pipe, args.model, save_dir)

    # Save the model
    model_path = config.MODELS_DIR / f"{args.model}.joblib"
    joblib.dump(pipe, model_path)
    print(f"Model saved → {model_path}")


if __name__ == "__main__":
    main()   
