
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AquaGuard AI | Water Infrastructure Monitoring",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parents[1]

# ==================================================
# 2. LOAD DATA AND MODEL
# ==================================================

@st.cache_data
def load_data():
    path = BASE_DIR / "data" / "processed" / "water_data.csv"

    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource
def load_model():
    path = BASE_DIR / "models" / "aquaguard_v4.joblib"
    return joblib.load(path)


df = load_data()
model_package = load_model()

model = model_package["model"]
features = model_package["features"]
medians = model_package["medians"]
threshold = model_package["threshold"]

# ==================================================
# 3. PREPARE MODEL PREDICTIONS
# ==================================================

prediction_data = df.copy()

prediction_data["gw_missing"] = (
    prediction_data["gw_lvl_site2_anomaly_score"]
    .isna()
    .astype(int)
)

prediction_data["vigor_missing"] = (
    prediction_data["326416_ws_vigor"]
    .isna()
    .astype(int)
)

prediction_data["hour"] = prediction_data["timestamp"].dt.hour
prediction_data["day_of_week"] = prediction_data["timestamp"].dt.dayofweek
prediction_data["month"] = prediction_data["timestamp"].dt.month

X = prediction_data[features].fillna(medians)

prediction_data["Fault_Score"] = model.predict_proba(X)[:, 1]

prediction_data["Prediction"] = (
    prediction_data["Fault_Score"] >= threshold
).astype(int)

prediction_data["Risk_Status"] = np.where(
    prediction_data["Prediction"] == 1,
    "⚠️ Flagged",
    "✅ Not Flagged"
)

# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("💧 AquaGuard AI")

st.sidebar.markdown(
    "Intelligent Water Infrastructure Monitoring"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Infrastructure Overview",
        "📈 Sensor Monitoring",
        "🚨 AI Risk Analysis",
        "🤖 Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Research Prototype\n\n"
    "Model: Random Forest v4\n\n"
    f"Alert Threshold: {threshold:.0%}"
)

st.sidebar.caption("AquaGuard AI | Portfolio Project")

# ==================================================
# 5. DASHBOARD HEADER
# ==================================================

st.title("💧 AquaGuard AI")

st.caption(
    "AI-powered analysis of historical water infrastructure "
    "sensor data and potential fault windows."
)

st.divider()

# ==================================================
# PAGE 1: INFRASTRUCTURE OVERVIEW
# ==================================================

if page == "📊 Infrastructure Overview":

    st.header("📊 Infrastructure Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Observations",
        f"{len(df):,}"
    )

    col2.metric(
        "Labelled Fault Windows",
        f"{int(df['fault_d7'].sum()):,}"
    )

    col3.metric(
        "Normal Observations",
        f"{int((df['fault_d7'] == 0).sum()):,}"
    )

    col4.metric(
        "AI-Flagged Observations",
        f"{int(prediction_data['Prediction'].sum()):,}"
    )

    st.divider()

    st.subheader("Dataset Overview")

    st.write(
        "Explore historical sensor measurements used to "
        "study potential infrastructure fault windows."
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Fault-Window Distribution")

    distribution = df["fault_d7"].value_counts()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Normal", "Fault Window"],
            y=[
                int(distribution.get(0, 0)),
                int(distribution.get(1, 0))
            ],
            marker_color=["#38BDF8", "#FF4B4B"]
        )
    )

    fig.update_layout(
        title="Historical Label Distribution",
        yaxis_title="Number of Observations",
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="distribution_chart"
    )

    st.info(
        "Fault-window labels describe periods associated "
        "with faults, not necessarily exact fault onset times."
    )

# ==================================================
# PAGE 2: SENSOR MONITORING
# ==================================================

elif page == "📈 Sensor Monitoring":

    st.header("📈 Interactive Sensor Monitoring")

    sensor_columns = [
        col
        for col in df.select_dtypes(include="number").columns
        if col != "fault_d7"
    ]

    selected_sensor = st.selectbox(
        "Select a sensor",
        sensor_columns
    )

    show_faults = st.checkbox(
        "Highlight fault-window observations",
        value=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df[selected_sensor],
            mode="lines",
            name="Sensor Reading",
            line=dict(
                color="#38BDF8",
                width=1.5
            )
        )
    )

    if show_faults:

        fault_points = df[df["fault_d7"] == 1]

        fig.add_trace(
            go.Scatter(
                x=fault_points["timestamp"],
                y=fault_points[selected_sensor],
                mode="markers",
                name="Labelled Fault Window",
                marker=dict(
                    color="#FF4B4B",
                    size=5,
                    opacity=0.75
                )
            )
        )

    fig.update_layout(
        title=f"{selected_sensor} Over Time",
        xaxis_title="Date",
        yaxis_title=selected_sensor,
        template="plotly_dark",
        height=550,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="sensor_chart"
    )

    st.caption(
        "Red markers identify labelled fault-window observations."
    )

    st.divider()

    st.subheader("Sensor Statistics")

    st.dataframe(
        df[selected_sensor].describe().to_frame(),
        use_container_width=True
    )

    st.subheader("Missing Sensor Readings")

    missing_count = int(df[selected_sensor].isna().sum())

    st.metric(
        "Missing Observations",
        f"{missing_count:,}"
    )

# ==================================================
# PAGE 3: AI RISK ANALYSIS
# ==================================================

elif page == "🚨 AI Risk Analysis":

    st.header("🚨 AI Fault Risk Analysis")

    flagged = int(prediction_data["Prediction"].sum())
    not_flagged = len(prediction_data) - flagged

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "AI-Flagged Records",
        f"{flagged:,}"
    )

    col2.metric(
        "Not Flagged",
        f"{not_flagged:,}"
    )

    col3.metric(
        "Alert Threshold",
        f"{threshold:.0%}"
    )

    st.divider()

    st.subheader("Predicted Fault-Window Scores Over Time")

    risk_fig = go.Figure()

    risk_fig.add_trace(
        go.Scatter(
            x=prediction_data["timestamp"],
            y=prediction_data["Fault_Score"],
            mode="lines",
            name="Model Score",
            line=dict(
                color="#38BDF8",
                width=1.5
            )
        )
    )

    risk_fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#FF4B4B",
        annotation_text="Alert Threshold"
    )

    risk_fig.update_layout(
        title="Historical AI Risk Monitoring",
        xaxis_title="Date",
        yaxis_title="Model Score",
        yaxis_range=[0, 1],
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        risk_fig,
        use_container_width=True,
        key="risk_monitoring_chart"
    )

    st.divider()

    st.subheader("AI Prediction Results")

    display_data = prediction_data[
        [
            "timestamp",
            "Fault_Score",
            "Prediction",
            "Risk_Status"
        ]
    ].copy()

    show_only_flagged = st.checkbox(
        "Show only flagged observations"
    )

    if show_only_flagged:
        display_data = display_data[
            display_data["Prediction"] == 1
        ]

    st.dataframe(
        display_data.tail(100),
        use_container_width=True
    )

    st.download_button(
        label="📥 Download AI Predictions",
        data=display_data.to_csv(index=False).encode("utf-8"),
        file_name="aquaguard_predictions.csv",
        mime="text/csv"
    )

    st.warning(
        "Predictions are based on historical observations, "
        "including data used during model development. "
        "Scores are not necessarily calibrated probabilities. "
        "This dashboard does not provide live fault alerts."
    )

# ==================================================
# PAGE 4: MODEL PERFORMANCE
# ==================================================

elif page == "🤖 Model Performance":

    st.header("🤖 Machine Learning Model Performance")

    comparison_path = (
        BASE_DIR / "reports" / "model_comparison.csv"
    )

    comparison = pd.read_csv(comparison_path)

    st.subheader("Model Comparison")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.divider()

    st.subheader("Validation Balanced Accuracy")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison["Model"],
            y=comparison["Balanced Accuracy"],
            marker_color=["#38BDF8", "#22C55E"],
            text=comparison["Balanced Accuracy"].round(3),
            textposition="outside"
        )
    )

    fig.update_layout(
        yaxis_title="Balanced Accuracy",
        yaxis_range=[0, 1],
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="model_comparison_chart"
    )

    st.divider()

    st.subheader("Selected Model")

    st.success(
        "Random Forest v4 achieved 72.6% balanced accuracy "
        "on the development validation dataset."
    )

    st.write(
        "**Algorithm:** Random Forest Classifier"
    )

    st.write(
        f"**Decision Threshold:** {threshold:.2f}"
    )

    st.write(
        "**Additional Features:** Groundwater missingness "
        "and sensor missingness indicators"
    )

    st.warning(
        "The selected model is a research prototype. "
        "Performance on later chronological data remains "
        "a concern, and additional independent validation "
        "is needed before deployment."
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "AquaGuard AI | Intelligent Water Infrastructure Analytics"
)
