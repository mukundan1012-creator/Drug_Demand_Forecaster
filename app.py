import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from src.data_loader import load_data, DRUG_COLUMNS
from src.preprocessing import prepare_arima_series, prepare_prophet_df
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Drug Demand Forecaster", page_icon="💊", layout="wide")

st.sidebar.title("💊 Drug Demand Forecaster")
st.sidebar.markdown("Forecasts daily pharmacy drug demand using ARIMA and Prophet with per-category auto-selection.")
st.sidebar.markdown("---")

selected_drug = st.sidebar.selectbox("Select Drug Category (ATC Code)", DRUG_COLUMNS)

st.sidebar.markdown("---")
st.sidebar.markdown("**ATC Drug Categories:**")
atc_info = {
    "M01AB": "Anti-inflammatory (Acetic acid)",
    "M01AE": "Anti-inflammatory (Propionic acid)",
    "N02BA": "Analgesics (Salicylic acid)",
    "N02BE": "Analgesics (Anilides / Paracetamol)",
    "N05B": "Anxiolytics",
    "N05C": "Hypnotics & Sedatives",
    "R03": "Respiratory / Anti-asthmatic",
    "R06": "Antihistamines"
}
for code, desc in atc_info.items():
    st.sidebar.markdown(f"- **{code}:** {desc}")

st.title("💊 Drug Demand Forecaster")
st.markdown("#### Per-Category ARIMA vs Prophet Auto-Selection across 8 ATC Drug Categories")
st.markdown("---")

@st.cache_data
def get_data():
    return load_data()

df = get_data()

summary_path = os.path.join(BASE_DIR, "model_comparison.csv")
summary_df = pd.read_csv(summary_path)
drug_row = summary_df[summary_df["Drug"] == selected_drug].iloc[0]

winner = drug_row["Winner"]
arima_rmse = drug_row["ARIMA RMSE"]
prophet_rmse = drug_row["Prophet RMSE"]
best_rmse = drug_row["Best RMSE"]

col1, col2, col3 = st.columns(3)
col1.metric("🏆 Winning Model", winner)
col2.metric("ARIMA RMSE", arima_rmse)
col3.metric("Prophet RMSE", prophet_rmse)

st.markdown("---")

model_path = os.path.join(BASE_DIR, "Model", f"{selected_drug}_{winner.lower()}.pkl")
model = joblib.load(model_path)

TEST_SIZE = 30

if winner == "ARIMA":
    train, test = prepare_arima_series(df, selected_drug, TEST_SIZE)
    forecast_values = model.forecast(steps=TEST_SIZE)
    forecast_values.index = test.index
    test_actual = test.values
    test_dates = test.index
    forecast_array = forecast_values.values
else:
    train, test = prepare_prophet_df(df, selected_drug, TEST_SIZE)
    future = model.make_future_dataframe(periods=TEST_SIZE)
    prophet_pred = model.predict(future)
    forecast_array = prophet_pred["yhat"].tail(TEST_SIZE).values
    test_actual = test["y"].values
    test_dates = pd.to_datetime(test["ds"].values)

st.subheader(f"📈 {selected_drug} — {winner} Forecast vs Actual (Last 30 Days)")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_dates, test_actual, label="Actual", marker="o", alpha=0.7)
ax.plot(test_dates, forecast_array, label=f"{winner} Forecast", marker="x", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Sales (Units)")
ax.set_title(f"{selected_drug} — {winner} Forecast vs Actual (Last 30 Days)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.subheader("📊 All Categories — Model Comparison Summary")
st.dataframe(summary_df, use_container_width=True)

st.markdown("---")
st.subheader("⚠️ Limitations")
st.markdown("""
- Fixed `ARIMA(2,0,2)` parameters used across all categories — not auto-tuned per drug
- Random demand spikes driven by external factors are inherently unpredictable from history alone
- N05C showed a convergence warning during training — results should be interpreted cautiously
- MAPE excludes zero-actual days to avoid division by zero — reported on non-zero days only
- Dataset covers 2014–2019 only — model may not generalize to post-2019 demand patterns
""")

st.markdown("---")
st.caption("Built by Mukundan.D | B.E. Electronics & Communication Engineering | Drug Demand Forecasting using Time-Series ML")