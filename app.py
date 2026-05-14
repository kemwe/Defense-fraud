# =============================================================================
# Fraud Detection — Full Analytics Dashboard
# Streamlit Standalone Version (No FastAPI Required)
#
# Pages:
#   1. Live Transaction Checker
#   2. Batch Upload & Results
#   3. Model Performance
#   4. SHAP Feature Importance
#
# Uses local ML model directly from Models/model_pipeline.pkl
#
# Kennedy Mwenda — Strathmore University, 2026
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import time

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# LOAD MODEL
# =============================================================================
MODEL_PATH = "Models/model_pipeline.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f1629;
        border-right: 1px solid #1e2d4a;
    }

    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f1629 0%, #1a2440 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #38bdf8;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
        margin-bottom: 20px;
        margin-top: 10px;
    }

    .page-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .page-subtitle {
        color: #64748b;
        margin-bottom: 30px;
    }

    .result-fraud {
        background: linear-gradient(135deg, #2d0a0a 0%, #3d1212 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-safe {
        background: linear-gradient(135deg, #0a2d1a 0%, #123d21 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 10px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0284c7, #0ea5e9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        width: 100%;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTS
# =============================================================================
BEST_MODEL_NAME = "Stacking-Diversity Ensemble"
BEST_PR_AUC = 1.0000
BEST_F1 = 0.9999

SHAP_FEATURES = [
    ("num__pagerank_orig", 0.0564),
    ("num__avg_rx_amount", 0.0525),
    ("num__community_orig", 0.0487),
    ("num__amount_dev_orig", 0.0462),
    ("cat__type_TRANSFER", 0.0413),
    ("num__total_amount", 0.0397),
    ("num__total_rx_amount", 0.0383),
    ("num__oldbalanceDest", 0.0366),
    ("num__avg_amount", 0.0310),
    ("num__tx_velocity", 0.0294),
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def gauge_chart(probability):

    color = (
        "#ef4444"
        if probability >= 0.75
        else "#f59e0b"
        if probability >= 0.40
        else "#22c55e"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Fraud Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#0a2d1a"},
                    {"range": [40, 75], "color": "#2d2000"},
                    {"range": [75, 100], "color": "#2d0a0a"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font_color="#e2e8f0",
        height=300,
    )

    return fig


def get_risk(prob):

    if prob >= 0.75:
        return "HIGH"

    elif prob >= 0.40:
        return "MEDIUM"

    return "LOW"


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:

    st.markdown(
        """
        <div style='text-align:center;padding:20px 0;'>
            <div style='font-size:3rem;'>🛡️</div>
            <div style='font-size:1.1rem;font-weight:700;color:#38bdf8;'>
                FraudShield
            </div>
            <div style='font-size:0.75rem;color:#64748b;'>
                Detection System v1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is not None:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Model file missing")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🔍 Live Transaction Checker",
            "📂 Batch Upload & Results",
            "📊 Model Performance",
            "🔬 SHAP Feature Importance",
        ],
    )

    st.markdown("---")

    st.markdown(
        f"""
        ### 🏆 Best Model
        
        **{BEST_MODEL_NAME}**
        
        - PR-AUC: **{BEST_PR_AUC}**
        - F1 Score: **{BEST_F1}**
        """,
    )

# =============================================================================
# PAGE 1 — LIVE CHECKER
# =============================================================================
if "Live" in page:

    st.markdown(
        '<div class="page-title">🔍 Live Transaction Checker</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Real-time fraud prediction using Stacking Ensemble model.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.markdown(
            '<div class="section-header">Transaction Details</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            tx_type = st.selectbox(
                "Transaction Type",
                ["TRANSFER", "PAYMENT", "DEPOSIT", "WITHDRAWAL", "DEBIT"],
            )

            amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=1000.0,
            )

            step = st.number_input(
                "Step",
                min_value=0,
                value=1,
            )

        with c2:
            oldbalanceOrg = st.number_input(
                "Sender Balance Before",
                min_value=0.0,
                value=5000.0,
            )

            newbalanceOrig = st.number_input(
                "Sender Balance After",
                min_value=0.0,
                value=4000.0,
            )

        c3, c4 = st.columns(2)

        with c3:
            oldbalanceDest = st.number_input(
                "Receiver Balance Before",
                min_value=0.0,
                value=1000.0,
            )

        with c4:
            newbalanceDest = st.number_input(
                "Receiver Balance After",
                min_value=0.0,
                value=2000.0,
            )

        st.markdown(
            '<div class="section-header">User Context</div>',
            unsafe_allow_html=True,
        )

        c5, c6 = st.columns(2)

        with c5:
            user_tx_count = st.number_input(
                "User Transaction Count",
                min_value=0.0,
                value=10.0,
            )

        with c6:
            user_avg_amount = st.number_input(
                "User Average Amount",
                min_value=0.0,
                value=500.0,
            )

        predict_btn = st.button("⚡ Run Fraud Detection")

    with col2:

        st.markdown(
            '<div class="section-header">Prediction Result</div>',
            unsafe_allow_html=True,
        )

        if predict_btn:

            if model is None:
                st.error("Model file not found.")
            else:

                payload = pd.DataFrame(
                    [
                        {
                            "step": step,
                            "type": tx_type,
                            "amount": amount,
                            "oldbalanceOrg": oldbalanceOrg,
                            "newbalanceOrig": newbalanceOrig,
                            "oldbalanceDest": oldbalanceDest,
                            "newbalanceDest": newbalanceDest,
                            "user_tx_count": user_tx_count,
                            "user_avg_amount": user_avg_amount,
                        }
                    ]
                )

                start = time.time()

                prediction = model.predict(payload)[0]

                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(payload)[0][1]
                else:
                    probability = 0.5

                latency = (time.time() - start) * 1000

                risk = get_risk(probability)

                if prediction == 1:

                    st.markdown(
                        f"""
                        <div class="result-fraud">
                            <div style='font-size:3rem;'>🚨</div>
                            <div class="result-title">FRAUD DETECTED</div>
                            <div>Risk Level: {risk}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="result-safe">
                            <div style='font-size:3rem;'>✅</div>
                            <div class="result-title">NO FRAUD</div>
                            <div>Risk Level: {risk}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.plotly_chart(
                    gauge_chart(probability),
                    use_container_width=True,
                )

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Fraud Probability",
                    f"{probability*100:.2f}%",
                )

                m2.metric(
                    "Risk",
                    risk,
                )

                m3.metric(
                    "Latency",
                    f"{latency:.2f} ms",
                )

# =============================================================================
# PAGE 2 — BATCH UPLOAD
# =============================================================================
elif "Batch" in page:

    st.markdown(
        '<div class="page-title">📂 Batch Upload & Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Upload CSV transactions for batch fraud detection.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
    )

    if uploaded:

        df = pd.read_csv(uploaded)

        st.markdown(
            '<div class="section-header">Uploaded Dataset</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(df.head())

        if st.button("⚡ Run Batch Prediction"):

            if model is None:
                st.error("Model file not found.")

            else:

                start = time.time()

                predictions = model.predict(df)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(df)[:, 1]
                else:
                    probabilities = np.zeros(len(df))

                latency = (time.time() - start) * 1000

                df["prediction"] = predictions
                df["fraud_probability"] = probabilities
                df["risk_level"] = [
                    get_risk(p)
                    for p in probabilities
                ]

                fraud_count = int(df["prediction"].sum())

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Total Records", len(df))
                c2.metric("Fraud Detected", fraud_count)
                c3.metric("Legitimate", len(df) - fraud_count)
                c4.metric("Latency", f"{latency:.2f} ms")

                fig = px.histogram(
                    df,
                    x="fraud_probability",
                    nbins=50,
                    title="Fraud Probability Distribution",
                )

                fig.update_layout(
                    paper_bgcolor="#0a0e1a",
                    plot_bgcolor="#0f1629",
                    font_color="#e2e8f0",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.markdown(
                    '<div class="section-header">Prediction Results</div>',
                    unsafe_allow_html=True,
                )

                st.dataframe(df)

                st.download_button(
                    "⬇️ Download Results",
                    data=df.to_csv(index=False),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

# =============================================================================
# PAGE 3 — MODEL PERFORMANCE
# =============================================================================
elif "Performance" in page:

    st.markdown(
        '<div class="page-title">📊 Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Evaluation metrics from thesis experiments.</div>',
        unsafe_allow_html=True,
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC",
            ],
            "Score": [
                0.9998,
                0.9997,
                1.0000,
                0.9999,
                1.0000,
                1.0000,
            ],
        }
    )

    st.dataframe(metrics_df, use_container_width=True)

    fig = px.bar(
        metrics_df,
        x="Metric",
        y="Score",
        title="Model Evaluation Metrics",
    )

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f1629",
        font_color="#e2e8f0",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    graph_paths = [
        "Graphs/dashboard_baseline_v3.png",
        "Graphs/dashboard_tuned_v3.png",
        "Graphs/baseline_vs_tuned_comparison_v3.png",
    ]

    for path in graph_paths:

        if os.path.exists(path):

            st.image(
                path,
                use_container_width=True,
            )

# =============================================================================
# PAGE 4 — SHAP
# =============================================================================
elif "SHAP" in page:

    st.markdown(
        '<div class="page-title">🔬 SHAP Feature Importance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Model explainability and feature importance analysis.</div>',
        unsafe_allow_html=True,
    )

    shap_df = pd.DataFrame(
        SHAP_FEATURES,
        columns=["Feature", "Importance"],
    )

    fig = px.bar(
        shap_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top SHAP Features",
    )

    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f1629",
        font_color="#e2e8f0",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    shap_images = [
        "shap_summary.png",
        "shap_bar.png",
        "shap_force.png",
    ]

    for img in shap_images:

        if os.path.exists(img):

            st.image(
                img,
                use_container_width=True,
            )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:#64748b;padding:20px;'>
        FraudShield Detection System • Kennedy Mwenda • Strathmore University • 2026
    </div>
    """,
    unsafe_allow_html=True,
)