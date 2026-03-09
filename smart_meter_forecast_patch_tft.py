from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")

# ------------------------------------------------
# Paths
# ------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

HIST_PATH = ROOT / "clean_50_meters_with_fft_clusters_named.parquet"

DEEPAR_FORECAST = RESULTS / "forecast_deepar.parquet"
PATCH_FORECAST = RESULTS / "forecast_patchtst.parquet"
TFT_FORECAST = RESULTS / "forecast_tft.parquet"

DEEPAR_METRICS = RESULTS / "meter_metrics_deepar.parquet"
PATCH_METRICS = RESULTS / "meter_metrics_patchtst.parquet"
TFT_METRICS = RESULTS / "meter_metrics_tft.parquet"

GLOBAL_DEEPAR = RESULTS / "global_stats_deepar.parquet"
GLOBAL_PATCH = RESULTS / "global_stats_patchtst.parquet"
GLOBAL_TFT = RESULTS / "global_stats_tft.parquet"

MODEL_COLORS = {
    "DeepAR": "seagreen",
    "PatchTST": "royalblue",
    "TFT": "darkorange",
}


# ------------------------------------------------
# Load Data
# ------------------------------------------------

@st.cache_data
def load_data():
    hist = pd.read_parquet(HIST_PATH)

    deepar = pd.read_parquet(DEEPAR_FORECAST)
    patch = pd.read_parquet(PATCH_FORECAST)
    tft = pd.read_parquet(TFT_FORECAST)

    metrics_deepar = pd.read_parquet(DEEPAR_METRICS)
    metrics_patch = pd.read_parquet(PATCH_METRICS)
    metrics_tft = pd.read_parquet(TFT_METRICS)

    global_deepar = pd.read_parquet(GLOBAL_DEEPAR)
    global_patch = pd.read_parquet(GLOBAL_PATCH)
    global_tft = pd.read_parquet(GLOBAL_TFT)

    return (
        hist,
        deepar,
        patch,
        tft,
        metrics_deepar,
        metrics_patch,
        metrics_tft,
        global_deepar,
        global_patch,
        global_tft,
    )


(
    hist,
    deepar,
    patch,
    tft,
    m_deepar,
    m_patch,
    m_tft,
    g_deepar,
    g_patch,
    g_tft,
) = load_data()

MODEL_FORECASTS = {
    "DeepAR": deepar,
    "PatchTST": patch,
    "TFT": tft,
}

MODEL_METRICS = {
    "DeepAR": m_deepar,
    "PatchTST": m_patch,
    "TFT": m_tft,
}

MODEL_GLOBAL = {
    "DeepAR": g_deepar,
    "PatchTST": g_patch,
    "TFT": g_tft,
}


# ------------------------------------------------
# Header
# ------------------------------------------------

st.title("Smart Meter Multi-Series Demand Forecasting Dashboard")

st.markdown(
    """
Compare **DeepAR**, **PatchTST**, and **TFT** on **50 residential smart meters**  
48-step horizon | 336-step encoder | Global multi-series training
"""
)

st.divider()


# ------------------------------------------------
# Sidebar
# ------------------------------------------------

st.sidebar.header("Controls")

meter_ids = sorted(hist["unique_id"].unique())

selected_meter = st.sidebar.selectbox("Select Household (Meter ID)", meter_ids)

model_choice = st.sidebar.selectbox("Model", ["PatchTST", "DeepAR", "TFT"], index=0)

comparison_mode = st.sidebar.checkbox("Enable Model Comparison")

if comparison_mode:
    aggregation_mode = st.sidebar.radio(
        "Aggregation Mode",
        ["Per Meter", "Average Across All 50 Meters"],
    )

    error_mode = st.sidebar.radio(
        "Error Breakdown",
        ["Overall", "By Hour of Day", "By Day of Week"],
    )


# ------------------------------------------------
# Filter Data
# ------------------------------------------------

hist_meter = hist[hist["unique_id"] == selected_meter]

deepar_meter = deepar[deepar["unique_id"] == selected_meter]
patch_meter = patch[patch["unique_id"] == selected_meter]
tft_meter = tft[tft["unique_id"] == selected_meter]

forecast_candidates = [
    df["timestamp"].min()
    for df in (patch_meter, deepar_meter, tft_meter)
    if not df.empty
]
forecast_start = min(forecast_candidates) if forecast_candidates else hist_meter["timestamp"].max()

hist_plot = hist_meter[hist_meter["timestamp"] < forecast_start]


# ------------------------------------------------
# Forecast Visualization
# ------------------------------------------------

st.subheader("Forecast Visualization")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=hist_plot["timestamp"],
        y=hist_plot["y"],
        name="Historical",
        line=dict(color="black"),
    )
)

if not comparison_mode:
    selected_forecast = MODEL_FORECASTS[model_choice]
    selected_meter_forecast = selected_forecast[selected_forecast["unique_id"] == selected_meter]

    fig.add_trace(
        go.Scatter(
            x=selected_meter_forecast["timestamp"],
            y=selected_meter_forecast["prediction"],
            name=f"{model_choice} Forecast",
            line=dict(color=MODEL_COLORS[model_choice]),
        )
    )
else:
    for model_name, meter_df in (
        ("DeepAR", deepar_meter),
        ("PatchTST", patch_meter),
        ("TFT", tft_meter),
    ):
        fig.add_trace(
            go.Scatter(
                x=meter_df["timestamp"],
                y=meter_df["prediction"],
                name=f"{model_name} Forecast",
                line=dict(color=MODEL_COLORS[model_name]),
            )
        )

fig.update_layout(
    height=500,
    xaxis_title="Time",
    yaxis_title="Electricity Demand",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()


# ------------------------------------------------
# Prediction Mode Metrics
# ------------------------------------------------

if not comparison_mode:
    st.subheader(f"Model Performance — {model_choice}")

    row = MODEL_METRICS[model_choice][MODEL_METRICS[model_choice]["unique_id"] == selected_meter].iloc[0]
    global_row = MODEL_GLOBAL[model_choice].iloc[0]

    st.markdown("### Meter Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("MAE", f"{row.mae:.2f}")
    c2.metric("RMSE", f"{row.rmse:.2f}")
    c3.metric("sMAPE", f"{row.smape:.2f}")
    c4.metric("MASE-48", f"{row.mase_48:.2f}")
    c5.metric("MASE-336", f"{row.mase_336:.2f}")

    st.markdown("### Global Performance (50 Meters)")

    g1, g2, g3, g4, g5 = st.columns(5)

    g1.metric("Mean MAE", f"{global_row.mae_mean:.2f}")
    g2.metric("Mean RMSE", f"{global_row.rmse_mean:.2f}")
    g3.metric("Mean sMAPE", f"{global_row.smape_mean:.2f}")
    g4.metric("Mean MAPE", f"{global_row.mape_mean:.2f}")
    g5.metric("Mean MASE", f"{global_row.mase_mean:.2f}")


# ------------------------------------------------
# Comparison Mode
# ------------------------------------------------

else:
    st.subheader("Model Comparison")

    if aggregation_mode == "Per Meter":
        table = pd.concat(
            [
                m_deepar[m_deepar["unique_id"] == selected_meter],
                m_patch[m_patch["unique_id"] == selected_meter],
                m_tft[m_tft["unique_id"] == selected_meter],
            ]
        )

        st.dataframe(table[["model", "mae", "rmse", "smape", "mase_48", "mase_336"]], use_container_width=True)
    else:
        global_df = pd.concat([g_deepar, g_patch, g_tft], ignore_index=True)
        st.dataframe(global_df, use_container_width=True)

    st.divider()

    st.subheader("Error Diagnostics")

    df_error = pd.concat([deepar_meter, patch_meter, tft_meter], ignore_index=True)
    df_error["abs_error"] = (df_error["actual"] - df_error["prediction"]).abs()
    df_error["hour"] = pd.to_datetime(df_error["timestamp"]).dt.hour
    df_error["day"] = pd.to_datetime(df_error["timestamp"]).dt.day_name()

    if error_mode == "By Hour of Day":
        err = df_error.groupby(["model", "hour"])["abs_error"].mean().reset_index()
        st.bar_chart(err.pivot(index="hour", columns="model", values="abs_error"))
    elif error_mode == "By Day of Week":
        err = df_error.groupby(["model", "day"])["abs_error"].mean().reset_index()
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        err["day"] = pd.Categorical(err["day"], categories=day_order, ordered=True)
        err = err.sort_values("day")
        st.bar_chart(err.pivot(index="day", columns="model", values="abs_error"))
    else:
        st.bar_chart(df_error.groupby("model")["abs_error"].mean())

    st.divider()

    st.subheader("Model Architecture Comparison")

    arch = pd.DataFrame(
        {
            "Feature": [
                "Architecture",
                "Handles Static Covariates",
                "Multi-horizon Forecasting",
                "Attention Mechanism",
                "Computational Complexity",
                "Interpretability",
            ],
            "DeepAR": [
                "Autoregressive RNN",
                "Yes",
                "Autoregressive",
                "No",
                "Low",
                "Low",
            ],
            "PatchTST": [
                "Transformer",
                "No",
                "Direct",
                "Yes",
                "High",
                "Medium",
            ],
            "TFT": [
                "Hybrid LSTM + Transformer",
                "Yes",
                "Direct",
                "Yes",
                "High",
                "High",
            ],
        }
    )

    st.table(arch)
