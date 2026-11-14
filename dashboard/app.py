
---

## 3. `dashboard/app.py` (Streamlit)

```python
# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modeling.preprocessing import add_date_features, INPUT_COLS

st.set_page_config(page_title="Rain Tomorrow Predictor", layout="wide")
st.title("🌧️ Rain Tomorrow — Australia")
st.caption("Predicts the probability of rain (≥ 1 mm) tomorrow from today's weather.")

# ─── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "xgboost.joblib"
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Run evaluate.py first.")
        st.stop()
    return joblib.load(model_path)

pipe = load_model()

# ─── Load sample data (for reference ranges) ─────────────────────────────────
@st.cache_data
def load_sample():
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "test.csv"
    if data_path.exists():
        return pd.read_csv(data_path, parse_dates=['Date'])
    return None

sample = load_sample()

# ─── Sidebar: feature inputs ─────────────────────────────────────────────────
st.sidebar.header("Today's Weather")

def slider_or_input(label, col, data):
    if data is not None and col in data.columns:
        s = data[col].dropna()
        if s.nunique() > 20:  # continuous
            return st.sidebar.slider(label, float(s.min()), float(s.max()),
                                     float(s.median()), 0.1)
        else:  # discrete
            return st.sidebar.selectbox(label, sorted(s.unique().astype(int)))
    return st.sidebar.number_input(label, value=0.0)

# Numeric
features = {}
features['MinTemp']     = st.sidebar.number_input('Min Temp (°C)', -10.0, 45.0, 15.0)
features['MaxTemp']     = st.sidebar.number_input('Max Temp (°C)', -5.0, 50.0, 25.0)
features['Rainfall']    = st.sidebar.number_input('Rainfall (mm)', 0.0, 200.0, 0.0)
features['Evaporation'] = st.sidebar.number_input('Evaporation (mm)', 0.0, 20.0, 2.0)
features['Sunshine']    = st.sidebar.number_input('Sunshine (h)', 0.0, 14.0, 8.0)
features['WindGustSpeed'] = st.sidebar.number_input('Wind Gust Speed (km/h)', 0.0, 200.0, 40.0)
features['WindSpeed9am']  = st.sidebar.number_input('Wind Speed 9am (km/h)', 0.0, 100.0, 15.0)
features['WindSpeed3pm']  = st.sidebar.number_input('Wind Speed 3pm (km/h)', 0.0, 100.0, 15.0)
features['Humidity9am']   = st.sidebar.number_input('Humidity 9am (%)', 0.0, 100.0, 60.0)
features['Humidity3pm']   = st.sidebar.number_input('Humidity 3pm (%)', 0.0, 100.0, 60.0)
features['Pressure9am']   = st.sidebar.number_input('Pressure 9am (hPa)', 950.0, 1050.0, 1015.0)
features['Pressure3pm']   = st.sidebar.number_input('Pressure 3pm (hPa)', 950.0, 1050.0, 1015.0)
features['Cloud9am']      = st.sidebar.selectbox('Cloud 9am (oktas)', range(9), index=5)
features['Cloud3pm']      = st.sidebar.selectbox('Cloud 3pm (oktas)', range(9), index=5)
features['Temp9am']       = st.sidebar.number_input('Temp 9am (°C)', -5.0, 45.0, 18.0)
features['Temp3pm']       = st.sidebar.number_input('Temp 3pm (°C)', -5.0, 50.0, 25.0)

# Categorical
COMPASS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
features['WindGustDir'] = st.sidebar.selectbox('Wind Gust Direction', COMPASS, index=0)
features['WindDir9am']  = st.sidebar.selectbox('Wind Direction 9am', COMPASS, index=0)
features['WindDir3pm']  = st.sidebar.selectbox('Wind Direction 3pm', COMPASS, index=0)
features['RainToday']   = st.sidebar.radio('Rain Today?', ['No', 'Yes'], horizontal=True)

# Location
if sample is not None:
    locations = sorted(sample['Location'].dropna().unique())
else:
    locations = ['Sydney', 'Brisbane', 'Perth', 'Melbourne', 'Adelaide']
features['Location'] = st.sidebar.selectbox('Location', locations)

# Date
features['Month']     = st.sidebar.number_input('Month', 1, 12, 11)
features['DayOfYear'] = st.sidebar.number_input('Day of Year', 1, 365, 314)

# ─── Predict ─────────────────────────────────────────────────────────────────
if st.button("Predict"):
    row = pd.DataFrame([features])
    # Ensure column order matches INPUT_COLS
    row = row[INPUT_COLS]

    proba = pipe.predict_proba(row)[0, 1]
    pred  = "Rain 🌧️" if proba >= 0.5 else "No Rain ☀️"

    st.subheader("Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of Rain", f"{proba:.1%}")
    col2.metric("Prediction", pred)
    col3.metric("Threshold", "≥ 50%")

    # Gauge
    st.progress(proba)
    st.caption("← Probability of rain tomorrow")

# ─── Model info tab ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Info", "📁 About"])

with tab2:
    st.subheader("Feature Importance (Top 15)")
    fig_path = Path(__file__).resolve().parents[1] / "reports" / "figures" / "importance_xgboost.png"
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info("Run `evaluate.py` to generate figures.")

    st.subheader("ROC / PR Curves")
    roc_path = Path(__file__).resolve().parents[1] / "reports" / "figures" / "roc_pr_xgboost.png"
    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)

with tab3:
    st.marked = None
    st.markdown("""
    **Model:** XGBoost (tuned with Optuna, 100 trials, 5-fold chronological CV)

    | Metric | Value |
    |---|---|
    | PR-AUC (test) | 0.7416 |
    | ROC-AUC (test) | 0.8863 |
    | F1 (test) | 0.6619 |

    **Split:** Chronological (Train: 2007–2014, Val: 2014–2015, Test: 2016–2017)

    **Preprocessing:** median impute + log1p (skewed), one-hot (categorical),
    label encode (binary). Fit on train only.
    """)   
