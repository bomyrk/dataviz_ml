# src/modeling/regression_maxtemp.py
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from . import config
from .preprocessing import add_date_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=config.NBR_SPLITS)
    args = parser.parse_args()

    # Load
    train = pd.read_csv(config.TRAIN_FOLDS_FILE, parse_dates=['Date'])
    val   = pd.read_csv(config.VAL_FILE, parse_dates=['Date'])
    test  = pd.read_csv(config.TEST_FILE, parse_dates=['Date'])

    # Features (as specified in the requirement)
    FEATURES = ['MinTemp', 'Humidity3pm', 'Sunshine', 'Pressure3pm']
    TARGET   = 'MaxTemp'

    # Drop rows where target is NaN
    train = train.dropna(subset=[TARGET])
    val   = val.dropna(subset=[TARGET])
    test  = test.dropna(subset=[TARGET])

    # ─── Train on train+val, evaluate on test ───────────────────────────────
    train_val = pd.concat([train, val], ignore_index=True).dropna(subset=FEATURES + [TARGET])
    test      = test.dropna(subset=FEATURES + [TARGET])

    X_fit = train_val[FEATURES].values
    y_fit = train_val[TARGET].values
    X_test = test[FEATURES].values
    y_test = test[TARGET].values

    # Simple Linear Regression (interpretable, shows the relationship)
    model = LinearRegression()
    model.fit(X_fit, y_fit)
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nMaxTemp Regression (Test)")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:.2f} °C")
    print(f"  RMSE = {rmse:.2f} °C")
    print(f"\nCoefficients:")
    for f, c in zip(FEATURES, model.coef_):
        print(f"  {f:>12}: {c:+.4f}")
    print(f"  {'Intercept':>12}: {model.intercept_:.4f}")

    # ─── Plots ──────────────────────────────────────────────────────────────
    save_dir = config.REPORTS_DIR / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.1, s=5, color='#3498db')
    lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    axes[0].plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel('Actual MaxTemp (°C)')
    axes[0].set_ylabel('Predicted MaxTemp (°C)')
    axes[0].set_title(f'Predicted vs Actual (R² = {r2:.3f})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_test, residuals, alpha=0.1, s=5, color='#e74c3c')
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].set_xlabel('Actual MaxTemp (°C)')
    axes[1].set_ylabel('Residual (°C)')
    axes[1].set_title(f'Residuals (MAE = {mae:.2f} °C)')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'regression_maxtemp.png', dpi=150)
    plt.close()
    print(f"\nFigure → {save_dir / 'regression_maxtemp.png'}")


if __name__ == "__main__":
    main()   
