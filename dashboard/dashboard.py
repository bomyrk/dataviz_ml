# dashboard/dashboard.py
# Victor Ogunkunle: Interactive dashboard for the Rain in Australia project.
# Run from the project root:  streamlit run dashboard/dashboard.py

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling.preprocessing import INPUT_COLS  # Leonel's feature list

MODEL_PATH = ROOT / "models" / "xgboost.joblib"
DATA_PATHS = [ROOT / "data" / "weather_clean.csv",      # Toyo's cleaned file
              ROOT / "data" / "raw" / "weatherAUS.csv"]  # raw Kaggle file

RAIN_COLORS = {"Yes": "#2a78d6", "No": "#eb6834"}  # blue = rain, orange = dry
TEST_START = "2016-01-01"                           # Leonel's chronological test split

NUM_COLS = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
            "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
            "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
            "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="Rain Tomorrow Dashboard", page_icon="🌧️", layout="wide")


# Data
@st.cache_data
def load_data():
    path = next((p for p in DATA_PATHS if p.exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.dropna(subset=["RainTomorrow"])
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["RainTomorrowFlag"] = (df["RainTomorrow"] == "Yes").astype(int)
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def train_model(df):
    """Fallback: rebuild Leonel's tuned XGBoost pipeline on pre-2016 data and save it."""
    from src.modeling.model_dispatcher import get_models
    pipe = get_models()["xgboost"]
    fit = df[df["Date"] < TEST_START]
    pipe.fit(fit[INPUT_COLS], fit["RainTomorrowFlag"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)


df = load_data()
if df is None:
    st.error("Dataset not found. Put Toyo's `weather_clean.csv` in `data/` "
             "(or the Kaggle `weatherAUS.csv` in `data/raw/`).")
    st.stop()


# Sidebar filters (apply to every chart)
st.sidebar.header("Filters")
all_locations = sorted(df["Location"].unique())
locations = st.sidebar.multiselect("Locations", all_locations,
                                   placeholder="All locations")
years = st.sidebar.slider("Years", int(df["Year"].min()), int(df["Year"].max()),
                          (int(df["Year"].min()), int(df["Year"].max())))
months = st.sidebar.multiselect("Months", list(range(1, 13)),
                                format_func=lambda m: MONTHS[m - 1],
                                placeholder="All months")

view = df[df["Year"].between(*years)]
if locations:
    view = view[view["Location"].isin(locations)]
if months:
    view = view[view["Month"].isin(months)]

st.sidebar.caption(f"{len(view):,} of {len(df):,} days selected")


# Header and KPIs
st.title("🌧️ Will it rain tomorrow?")
st.caption("Rain in Australia, 2007–2017. Rain tomorrow means at least 1 mm.")

if view.empty:
    st.warning("No days match these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Days", f"{len(view):,}")
k2.metric("Rain-tomorrow rate", f"{view['RainTomorrowFlag'].mean():.1%}")
k3.metric("Average daily rainfall", f"{view['Rainfall'].mean():.1f} mm")
k4.metric("Locations", view["Location"].nunique())

tab6, tab7, tab8, tab_pred, tab_model = st.tabs([
    "Chart 6 · Scatter", "Chart 7 · Rain by category",
    "Chart 8 · Distribution slider", "🔮 Rain predictor", "📊 Model"])


# Chart 6: interactive scatter coloured by RainTomorrow
with tab6:
    st.subheader("How two of today's measurements relate to rain tomorrow")
    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X axis", NUM_COLS, index=NUM_COLS.index("Humidity3pm"))
    y_col = c2.selectbox("Y axis", NUM_COLS, index=NUM_COLS.index("Pressure3pm"))
    n_points = c3.slider("Points to show", 1000, 10000, 4000, 1000)

    pts = view.dropna(subset=[x_col, y_col])
    pts = pts.sample(min(n_points, len(pts)), random_state=42)
    fig6 = px.scatter(pts, x=x_col, y=y_col, color="RainTomorrow",
                      color_discrete_map=RAIN_COLORS, opacity=0.6,
                      category_orders={"RainTomorrow": ["No", "Yes"]},
                      hover_data=["Location", "Date"])
    fig6.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig6.update_layout(legend_title_text="Rain tomorrow", height=520)
    st.plotly_chart(fig6, width="stretch")
    st.caption("Rainy days next day cluster at high afternoon humidity and low pressure. "
               "Drag to zoom, double-click to reset, click the legend to hide a group.")


# Chart 7: filterable bar chart of rain rate by a category
with tab7:
    st.subheader("Rain-tomorrow rate by category")
    group_options = {"Cloud cover at 3pm (oktas)": "Cloud3pm",
                     "Rain today": "RainToday",
                     "Month": "Month",
                     "Wind gust direction": "WindGustDir",
                     "Location": "Location"}
    c1, c2 = st.columns([1, 2])
    group_label = c1.selectbox("Group by", list(group_options))
    group_col = group_options[group_label]

    rates = (view.dropna(subset=[group_col])
             .groupby(group_col)["RainTomorrowFlag"]
             .agg(rate="mean", days="count").reset_index())
    if group_col == "Month":
        rates[group_col] = rates[group_col].map(lambda m: MONTHS[int(m) - 1])
    elif group_col == "Cloud3pm":
        rates[group_col] = rates[group_col].astype(int).astype(str)
    elif group_col == "WindGustDir":
        rates[group_col] = pd.Categorical(rates[group_col], COMPASS, ordered=True)
        rates = rates.sort_values(group_col)
    rates[group_col] = rates[group_col].astype(str)

    keep = c2.multiselect("Show only", rates[group_col].tolist(),
                          placeholder="All groups")
    if keep:
        rates = rates[rates[group_col].isin(keep)]
    if st.checkbox("Sort by rain rate", value=(group_col == "Location")):
        rates = rates.sort_values("rate", ascending=False)

    fig7 = px.bar(rates, x=group_col, y="rate", hover_data={"days": ":,"},
                  labels={"rate": "Rain-tomorrow rate", group_col: group_label})
    fig7.update_traces(marker_color=RAIN_COLORS["Yes"],
                       hovertemplate="%{x}<br>Rain rate: %{y:.1%}<br>Days: %{customdata[0]:,}")
    fig7.add_hline(y=view["RainTomorrowFlag"].mean(), line_dash="dash",
                   line_color="gray", annotation_text="Overall rate",
                   annotation_position="top left")
    fig7.update_layout(yaxis_tickformat=".0%", height=480, bargap=0.25)
    fig7.update_xaxes(type="category")
    st.plotly_chart(fig7, width="stretch")


# Chart 8: distribution with a range slider
with tab8:
    st.subheader("Explore one measurement with a range slider")
    c1, c2 = st.columns([1, 2])
    feat = c1.selectbox("Measurement", NUM_COLS, index=NUM_COLS.index("Humidity3pm"))
    s = view[feat].dropna()
    lo, hi = float(s.min()), float(s.max())
    rng = c2.slider("Range", lo, hi, (lo, hi))

    sel = view[view[feat].between(*rng)]
    fig8 = px.histogram(sel, x=feat, color="RainTomorrow", nbins=40,
                        barmode="overlay", opacity=0.65,
                        color_discrete_map=RAIN_COLORS,
                        category_orders={"RainTomorrow": ["No", "Yes"]})
    fig8.update_layout(legend_title_text="Rain tomorrow", height=480,
                       yaxis_title="Days")
    st.plotly_chart(fig8, width="stretch")

    m1, m2 = st.columns(2)
    m1.metric("Days in range", f"{len(sel):,}")
    m2.metric("Rain-tomorrow rate in range",
              f"{sel['RainTomorrowFlag'].mean():.1%}" if len(sel) else "—",
              delta=(f"{sel['RainTomorrowFlag'].mean() - view['RainTomorrowFlag'].mean():+.1%} vs all"
                     if len(sel) else None))


# Prediction simulator
with tab_pred:
    st.subheader("Rain predictor")
    model = load_model()
    if model is None:
        st.warning(f"No trained model at `{MODEL_PATH.relative_to(ROOT)}`. "
                   "Ask Leonel for `xgboost.joblib`, or train it here.")
        if st.button("Train the model now (takes a few minutes)"):
            with st.spinner("Training XGBoost on 2007–2015 data…"):
                train_model(df)
            load_model.clear()
            st.rerun()
        st.stop()

    st.caption("Enter today's weather. Leave a box empty if it wasn't measured.")

    # Default values: median of each column; "Load a real day" overwrites them.
    if "pred_defaults" not in st.session_state:
        st.session_state.pred_defaults = None
    if st.button("🎲 Load a real day from the test period"):
        row = df[df["Date"] >= TEST_START].sample(1).iloc[0]
        st.session_state.pred_defaults = row
    real = st.session_state.pred_defaults

    def default(col):
        if real is not None:
            v = real[col]
            return None if pd.isna(v) else float(v)
        return float(df[col].median())

    def pick(options, col, fallback):
        v = real[col] if real is not None else fallback
        return options.index(v) if v in options else options.index(fallback)

    with st.form("predict"):
        c1, c2, c3, c4 = st.columns(4)
        loc = c1.selectbox("Location", all_locations, index=pick(all_locations, "Location", "Sydney"))
        date = c2.date_input("Date", value=real["Date"] if real is not None else pd.Timestamp("2016-11-23"))
        rain_today = c3.radio("Rain today?", ["No", "Yes"], horizontal=True,
                              index=pick(["No", "Yes"], "RainToday", "No"))
        rainfall = c4.number_input("Rainfall today (mm)", 0.0, 400.0, default("Rainfall") or 0.0)

        c1, c2, c3, c4 = st.columns(4)
        min_t = c1.number_input("Min temp (°C)", -10.0, 50.0, default("MinTemp"))
        max_t = c2.number_input("Max temp (°C)", -10.0, 50.0, default("MaxTemp"))
        t9 = c3.number_input("Temp 9am (°C)", -10.0, 50.0, default("Temp9am"))
        t3 = c4.number_input("Temp 3pm (°C)", -10.0, 50.0, default("Temp3pm"))

        c1, c2, c3, c4 = st.columns(4)
        h9 = c1.number_input("Humidity 9am (%)", 0.0, 100.0, default("Humidity9am"))
        h3 = c2.number_input("Humidity 3pm (%)", 0.0, 100.0, default("Humidity3pm"))
        p9 = c3.number_input("Pressure 9am (hPa)", 970.0, 1050.0, default("Pressure9am"))
        p3 = c4.number_input("Pressure 3pm (hPa)", 970.0, 1050.0, default("Pressure3pm"))

        c1, c2, c3, c4 = st.columns(4)
        gust = c1.number_input("Wind gust speed (km/h)", 0.0, 150.0, default("WindGustSpeed"))
        ws9 = c2.number_input("Wind speed 9am (km/h)", 0.0, 150.0, default("WindSpeed9am"))
        ws3 = c3.number_input("Wind speed 3pm (km/h)", 0.0, 150.0, default("WindSpeed3pm"))
        gust_dir = c4.selectbox("Wind gust direction", COMPASS, index=pick(COMPASS, "WindGustDir", "W"))

        c1, c2, c3, c4 = st.columns(4)
        dir9 = c1.selectbox("Wind direction 9am", COMPASS, index=pick(COMPASS, "WindDir9am", "N"))
        dir3 = c2.selectbox("Wind direction 3pm", COMPASS, index=pick(COMPASS, "WindDir3pm", "SE"))
        sun = c3.number_input("Sunshine (hours)", 0.0, 15.0, default("Sunshine"))
        evap = c4.number_input("Evaporation (mm)", 0.0, 150.0, default("Evaporation"))

        c1, c2, _, _ = st.columns(4)
        cl9 = c1.number_input("Cloud 9am (oktas, 0–8)", 0.0, 8.0, default("Cloud9am"), step=1.0)
        cl3 = c2.number_input("Cloud 3pm (oktas, 0–8)", 0.0, 8.0, default("Cloud3pm"), step=1.0)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        date = pd.Timestamp(date)
        values = {
            "Rainfall": rainfall, "Evaporation": evap, "WindGustSpeed": gust,
            "WindSpeed9am": ws9, "WindSpeed3pm": ws3, "MinTemp": min_t, "MaxTemp": max_t,
            "Temp9am": t9, "Temp3pm": t3, "Pressure9am": p9, "Pressure3pm": p3,
            "Humidity9am": h9, "Humidity3pm": h3, "Sunshine": sun,
            "Cloud9am": cl9, "Cloud3pm": cl3, "Location": loc, "WindGustDir": gust_dir,
            "WindDir9am": dir9, "WindDir3pm": dir3, "RainToday": rain_today,
            "Month": date.month, "DayOfYear": date.dayofyear,
        }
        row = pd.DataFrame([values])[INPUT_COLS]
        num = [c for c in INPUT_COLS if c not in
               ("Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday")]
        row[num] = row[num].astype(float)  # empty boxes become NaN; the pipeline imputes them

        proba = float(model.predict_proba(row)[0, 1])
        r1, r2 = st.columns([1, 1])
        r1.metric("Chance of rain tomorrow", f"{proba:.0%}")
        r1.markdown("### 🌧️ Rain expected" if proba >= 0.5 else "### ☀️ No rain expected")
        r1.caption("Decision threshold: 50%")
        if real is not None:
            r1.caption(f"What actually happened: rain tomorrow = **{real['RainTomorrow']}**")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": RAIN_COLORS["Yes"]},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
        gauge.update_layout(height=260, margin=dict(t=20, b=10, l=30, r=30))
        r2.plotly_chart(gauge, width="stretch")


# Model info (Leonel's results)
with tab_model:
    st.subheader("Best model: XGBoost (tuned)")
    metrics_file = ROOT / "reports" / "test_metrics.csv"
    if metrics_file.exists():
        m = pd.read_csv(metrics_file).iloc[0]
        a, b, c = st.columns(3)
        a.metric("PR-AUC (test)", f"{m['pr_auc']:.3f}")
        b.metric("ROC-AUC (test)", f"{m['roc_auc']:.3f}")
        c.metric("F1 (test)", f"{m['f1']:.3f}")
    st.caption("Trained on 2007–2015, tested once on 2016–2017. "
               "Compared against Logistic Regression and LinearSVC.")
    figs = ROOT / "reports" / "figures"
    for name, title in [("importance_xgboost.png", "Top features"),
                        ("roc_pr_xgboost.png", "ROC and precision-recall curves"),
                        ("confusion_matrix_xgboost.png", "Confusion matrix")]:
        if (figs / name).exists():
            st.markdown(f"**{title}**")
            st.image(str(figs / name), width="stretch")
