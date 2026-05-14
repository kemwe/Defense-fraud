# =============================================================================
# Fraud Detection — Full Analytics Dashboard
# Streamlit app with 4 pages:
#   1. Live Transaction Checker
#   2. Batch Upload & Results
#   3. Model Performance
#   4. SHAP Feature Importance
# Connects to FastAPI backend at http://localhost:8000
# Kennedy Mwenda — Strathmore University, 2026
#
# Best Model: Stacking-Diversity Ensemble (PR-AUC: 1.0000, F1: 0.9999)
# Dataset: MoMTSim Version 2 (4,225,958 records)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import io

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS — dark financial theme
# =============================================================================
st.markdown(
    """
<style>
    /* ---- Base ---- */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #0f1629;
        border-right: 1px solid #1e2d4a;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: linear-gradient(135deg, #0f1629 0%, #1a2440 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-top: 4px;
    }

    /* ---- Result cards ---- */
    .result-fraud {
        background: linear-gradient(135deg, #2d0a0a 0%, #3d1212 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-no-fraud {
        background: linear-gradient(135deg, #0a2d1a 0%, #123d21 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-fraud .result-title   { color: #ef4444; }
    .result-no-fraud .result-title { color: #22c55e; }

    /* ---- Risk badges ---- */
    .badge-high   { background:#ef4444; color:#fff; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; letter-spacing:0.05em; }
    .badge-medium { background:#f59e0b; color:#000; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; letter-spacing:0.05em; }
    .badge-low    { background:#22c55e; color:#000; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; letter-spacing:0.05em; }

    /* ---- Section headers ---- */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #38bdf8;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    /* ---- Input styling ---- */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #0f1629 !important;
        border: 1px solid #1e3a5f !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #0369a1, #0ea5e9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.05em;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0ea5e9, #38bdf8);
        transform: translateY(-1px);
    }

    /* ---- Dataframe ---- */
    .dataframe { background-color: #0f1629 !important; }

    /* ---- Divider ---- */
    hr { border-color: #1e2d4a; }

    /* ---- Page title ---- */
    .page-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        margin-bottom: 32px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Constants
# =============================================================================
API_BASE = "http://localhost:8000"

# ── Baseline model results — confirmed from dashboard_baseline_v3.png ──
MODEL_RESULTS = {
    "LogisticRegression": {
        "Accuracy": 0.9440,
        "Precision": 0.9490,
        "Recall": 0.9680,
        "F1-Score": 0.9580,
        "ROC-AUC": 0.9890,
        "PR-AUC": 0.9930,
        "Brier Score": 0.0760,
        "Latency (ms)": 0.003737,
    },
    "RandomForest": {
        "Accuracy": 0.9900,
        "Precision": 0.9960,
        "Recall": 0.9890,
        "F1-Score": 0.9930,
        "ROC-AUC": 0.9990,
        "PR-AUC": 0.9990,
        "Brier Score": 0.0762,
        "Latency (ms)": 0.041937,
    },
    "XGBoost": {
        "Accuracy": 0.4410,
        "Precision": 0.8390,
        "Recall": 0.2090,
        "F1-Score": 0.3350,
        "ROC-AUC": 0.6340,
        "PR-AUC": 0.7790,
        "Brier Score": 0.3146,
        "Latency (ms)": 0.001394,
    },
    "LightGBM": {
        "Accuracy": 0.6140,
        "Precision": 0.6810,
        "Recall": 0.8030,
        "F1-Score": 0.7370,
        "ROC-AUC": 0.6330,
        "PR-AUC": 0.7790,
        "Brier Score": 0.2298,
        "Latency (ms)": 0.001799,
    },
    "CatBoost": {
        "Accuracy": 0.6320,
        "Precision": 0.6850,
        "Recall": 0.8380,
        "F1-Score": 0.7540,
        "ROC-AUC": 0.7330,
        "PR-AUC": 0.8790,
        "Brier Score": 0.2311,
        "Latency (ms)": 0.001908,
    },
    "Stacking-Diversity (Baseline) ★": {
        "Accuracy": 0.9998,
        "Precision": 0.9997,
        "Recall": 1.0000,
        "F1-Score": 0.9999,
        "ROC-AUC": 1.0000,
        "PR-AUC": 1.0000,
        "Brier Score": 0.0549,
        "Latency (ms)": 0.028498,
    },
}

# ── Tuned model results — confirmed from dashboard_tuned_v3.png ──
TUNED_RESULTS = {
    "LogisticRegression (Tuned)": {
        "Accuracy": 0.6730,
        "Precision": 0.6730,
        "Recall": 1.0000,
        "F1-Score": 0.8040,
        "ROC-AUC": 0.0490,
        "PR-AUC": 0.4640,
        "Brier Score": 0.2352,
        "Latency (ms)": 0.000952,
    },
    "RandomForest (Tuned)": {
        "Accuracy": 0.5680,
        "Precision": 0.6440,
        "Recall": 0.8020,
        "F1-Score": 0.7140,
        "ROC-AUC": 0.6300,
        "PR-AUC": 0.8430,
        "Brier Score": 0.2316,
        "Latency (ms)": 0.004274,
    },
    "XGBoost (Tuned)": {
        "Accuracy": 0.6530,
        "Precision": 0.6660,
        "Recall": 0.9690,
        "F1-Score": 0.7900,
        "ROC-AUC": 0.2620,
        "PR-AUC": 0.5410,
        "Brier Score": 0.2443,
        "Latency (ms)": 0.001947,
    },
    "LightGBM (Tuned)": {
        "Accuracy": 0.6200,
        "Precision": 0.6580,
        "Recall": 0.9050,
        "F1-Score": 0.7620,
        "ROC-AUC": 0.2370,
        "PR-AUC": 0.5160,
        "Brier Score": 0.2550,
        "Latency (ms)": 0.002835,
    },
    "CatBoost (Tuned)": {
        "Accuracy": 0.4550,
        "Precision": 0.5940,
        "Recall": 0.5990,
        "F1-Score": 0.5970,
        "ROC-AUC": 0.2660,
        "PR-AUC": 0.5310,
        "Brier Score": 0.3081,
        "Latency (ms)": 0.001908,
    },
    "Stacking-Diversity (Tuned)": {
        "Accuracy": 0.9147,
        "Precision": 0.9002,
        "Recall": 0.9821,
        "F1-Score": 0.9394,
        "ROC-AUC": 0.9802,
        "PR-AUC": 0.9844,
        "Brier Score": 0.0920,
        "Latency (ms)": 0.013783,
    },
}

BEST_MODEL_NAME = "Stacking-Diversity (Baseline) ★"
BEST_THRESHOLD = 0.5492
# Confusion matrix — Stacking-Diversity Baseline, validation set (318,160 records)
CM_TP, CM_FP, CM_TN, CM_FN = 214044, 54, 104055, 7

# SHAP top features — Stacking-Diversity (Baseline), PermutationExplainer, 2,000 val samples (notebook Cell [105])
SHAP_FEATURES = [
    {
        "Feature": "num__pagerank_orig",
        "Mean |SHAP|": 0.0564,
        "Influence": "🔴 High",
        "Interpretation": "Originator PageRank — high-centrality sender = network hub / fraud ring coordinator",
    },
    {
        "Feature": "num__avg_rx_amount",
        "Mean |SHAP|": 0.0525,
        "Influence": "🔴 High",
        "Interpretation": "Recipient average received amount — high volumes signal money mule aggregation",
    },
    {
        "Feature": "num__community_orig",
        "Mean |SHAP|": 0.0487,
        "Influence": "🔴 High",
        "Interpretation": "Originator graph community — fraud cluster membership signal",
    },
    {
        "Feature": "num__amount_dev_orig",
        "Mean |SHAP|": 0.0462,
        "Influence": "🔴 High",
        "Interpretation": "Deviation from sender's avg — unusual transaction amount = structuring signal",
    },
    {
        "Feature": "cat__type_TRANSFER",
        "Mean |SHAP|": 0.0413,
        "Influence": "🔴 High",
        "Interpretation": "TRANSFER type — 88.94% of all TRANSFER transactions are fraudulent",
    },
    {
        "Feature": "num__total_amount",
        "Mean |SHAP|": 0.0397,
        "Influence": "🟡 Medium",
        "Interpretation": "Cumulative sender total — high totals from low-frequency accounts are suspicious",
    },
    {
        "Feature": "num__total_rx_amount",
        "Mean |SHAP|": 0.0383,
        "Influence": "🟡 Medium",
        "Interpretation": "Total received by recipient — mule account accumulation signal",
    },
    {
        "Feature": "num__oldbalanceDest",
        "Mean |SHAP|": 0.0366,
        "Influence": "🟡 Medium",
        "Interpretation": "Recipient balance before tx — low balance + large receipt = new/mule account",
    },
    {
        "Feature": "num__avg_amount",
        "Mean |SHAP|": 0.0310,
        "Influence": "🟢 Low-Med",
        "Interpretation": "Sender historical average amount — baseline for structuring deviation detection",
    },
    {
        "Feature": "num__tx_velocity",
        "Mean |SHAP|": 0.0294,
        "Influence": "🟢 Low-Med",
        "Interpretation": "Transaction velocity — rapid successive transfers indicate account takeover",
    },
]


# =============================================================================
# Helpers
# =============================================================================
def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except:
        return False


def risk_badge(level: str) -> str:
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(
        level, "badge-low"
    )
    return f'<span class="{cls}">{level}</span>'


def gauge_chart(prob: float):
    colour = "#ef4444" if prob >= 0.75 else "#f59e0b" if prob >= 0.40 else "#22c55e"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 2),
            title={
                "text": "Fraud Probability (%)",
                "font": {"color": "#94a3b8", "size": 13},
            },
            number={
                "suffix": "%",
                "font": {"color": colour, "size": 32, "family": "IBM Plex Mono"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#334155",
                    "tickfont": {"color": "#64748b"},
                },
                "bar": {"color": colour},
                "bgcolor": "#0f1629",
                "steps": [
                    {"range": [0, 40], "color": "#0a2d1a"},
                    {"range": [40, 75], "color": "#2d2000"},
                    {"range": [75, 100], "color": "#2d0a0a"},
                ],
                "threshold": {
                    "line": {"color": colour, "width": 3},
                    "value": round(prob * 100, 2),
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        font_color="#94a3b8",
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


# =============================================================================
# Sidebar Navigation
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 20px 0 28px 0;'>
            <div style='font-size:2.5rem;'>🛡️</div>
            <div style='font-family: IBM Plex Mono; font-size:1rem; font-weight:700; color:#38bdf8;'>FraudShield</div>
            <div style='font-size:0.7rem; color:#475569; letter-spacing:0.1em; text-transform:uppercase;'>Detection System v1.0</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # API Status
    api_ok = check_api_health()
    status_color = "#22c55e" if api_ok else "#ef4444"
    status_text = "API Connected" if api_ok else "API Offline — run python main.py"
    st.markdown(
        f"""
        <div style='background:#0f1629; border:1px solid #1e3a5f; border-radius:8px;
                    padding:10px 14px; margin-bottom:16px; display:flex; align-items:center; gap:8px;'>
            <span style='color:{status_color}; font-size:0.7rem;'>●</span>
            <span style='font-size:0.8rem; color:#94a3b8;'>{status_text}</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Best model metrics in sidebar
    st.markdown(
        """
        <div style='background:#0f1629; border:1px solid #1e3a5f; border-radius:8px;
                    padding:12px 14px; margin-bottom:16px;'>
            <div style='font-family:IBM Plex Mono; font-size:0.65rem; color:#38bdf8;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>
                Best Model · Stacking Ensemble ★
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:0.78rem;'>
                <span style='color:#64748b;'>PR-AUC</span>
                <span style='color:#38bdf8; font-family:IBM Plex Mono;'>1.0000</span>
                <span style='color:#64748b;'>F1</span>
                <span style='color:#38bdf8; font-family:IBM Plex Mono;'>0.9999</span>
                <span style='color:#64748b;'>Threshold</span>
                <span style='color:#38bdf8; font-family:IBM Plex Mono;'>0.5492</span>
                <span style='color:#64748b;'>Latency</span>
                <span style='color:#38bdf8; font-family:IBM Plex Mono;'>0.028 ms</span>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🔍  Live Transaction Checker",
            "📂  Batch Upload & Results",
            "📊  Model Performance",
            "🔬  SHAP Feature Importance",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.7rem; color:#334155; text-align:center; padding-top:8px;'>
            Kennedy Mwenda · 193443<br>
            Strathmore University, 2026<br>
            <span style='color:#1e3a5f;'>MoMTSim v2 Dataset</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE 1 — Live Transaction Checker
# =============================================================================
if "Live" in page:
    st.markdown(
        '<div class="page-title">🔍 Live Transaction Checker</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Submit a single transaction for an instant Stacking-Diversity Ensemble fraud prediction (threshold: 0.5492).</div>',
        unsafe_allow_html=True,
    )

    if not api_ok:
        st.error(
            "⚠️ API is offline. Start the FastAPI server with `python main.py` before using this page."
        )
        st.stop()

    col_form, col_result = st.columns([1.2, 1], gap="large")

    with col_form:
        st.markdown(
            '<div class="section-header">Transaction Details</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "📌 Fraud occurs ONLY in TRANSFER transactions (88.94% fraud rate in dataset)"
        )

        c1, c2 = st.columns(2)
        with c1:
            tx_type = st.selectbox(
                "Transaction Type",
                ["TRANSFER", "PAYMENT", "DEBIT", "DEPOSIT", "WITHDRAWAL"],
            )
            amount = st.number_input(
                "Amount (KES)", min_value=0.01, value=39.03, format="%.2f"
            )
            step = st.number_input(
                "Step (0–192)",
                min_value=0,
                max_value=192,
                value=10,
                help="1 step = 1 hour. Dataset spans 8 days (0–192 steps).",
            )
        with c2:
            old_bal_orig = st.number_input(
                "Sender Balance (Before)", min_value=0.0, value=39.03, format="%.2f"
            )
            new_bal_orig = st.number_input(
                "Sender Balance (After)",
                min_value=0.0,
                value=0.0,
                format="%.2f",
                help="Fraud pattern: sender balance drained to 0",
            )

        c3, c4 = st.columns(2)
        with c3:
            old_bal_dest = st.number_input(
                "Recipient Balance (Before)", min_value=0.0, value=55.94, format="%.2f"
            )
        with c4:
            new_bal_dest = st.number_input(
                "Recipient Balance (After)", min_value=0.0, value=94.98, format="%.2f"
            )

        st.markdown(
            '<div class="section-header" style="margin-top:20px;">User Context</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Legitimate avg: 2,397 tx / 70,710 KES  |  Fraud avg: 535 tx / 37,236 KES"
        )

        c5, c6 = st.columns(2)
        with c5:
            user_tx_count = st.number_input(
                "User Tx Count",
                min_value=0.0,
                value=3.0,
                format="%.1f",
                help="Total historical transactions by this sender",
            )
        with c6:
            user_avg_amount = st.number_input(
                "User Avg Amount (KES)",
                min_value=0.0,
                value=45.67,
                format="%.2f",
                help="Sender's average transaction amount historically",
            )

        predict_btn = st.button("⚡ Run Fraud Check")

    with col_result:
        st.markdown(
            '<div class="section-header">Prediction Result</div>',
            unsafe_allow_html=True,
        )

        if predict_btn:
            payload = {
                "step": step,
                "type": tx_type,
                "amount": amount,
                "oldbalanceOrg": old_bal_orig,
                "newbalanceOrig": new_bal_orig,
                "oldbalanceDest": old_bal_dest,
                "newbalanceDest": new_bal_dest,
                "user_tx_count": user_tx_count,
                "user_avg_amount": user_avg_amount,
            }
            with st.spinner("Running Stacking Ensemble inference..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/predict", json=payload, timeout=10
                    )
                    data = resp.json()

                    if resp.status_code == 200:
                        pred = data["prediction"]
                        label = data["prediction_label"]
                        prob = data["fraud_probability"]
                        risk = data["risk_level"]
                        lat = data["latency_ms"]

                        # Result card
                        card_class = "result-fraud" if pred == 1 else "result-no-fraud"
                        icon = "🚨" if pred == 1 else "✅"
                        st.markdown(
                            f"""
                            <div class="{card_class}">
                                <div style='font-size:2.5rem;'>{icon}</div>
                                <div class="result-title">{label}</div>
                                <div style='color:#94a3b8; font-size:0.85rem; margin-top:4px;'>
                                    Risk Level: {risk_badge(risk)}
                                </div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        # Gauge
                        st.plotly_chart(gauge_chart(prob), use_container_width=True)

                        # Stats row
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Fraud Probability", f"{prob*100:.2f}%")
                        m2.metric("Latency", f"{lat:.2f} ms")
                        m3.metric("Model", "Stacking ★")

                        if pred == 1:
                            st.warning(
                                "⚠️ Fraud indicators: TRANSFER type + "
                                "sender balance drained to 0 + high amount deviation"
                            )
                    else:
                        st.error(f"API Error: {data.get('detail', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Request failed: {str(e)}")
        else:
            st.markdown(
                """
                <div style='text-align:center; padding:60px 20px; color:#334155;'>
                    <div style='font-size:3rem;'>🛡️</div>
                    <div style='font-family: IBM Plex Mono; font-size:0.85rem; margin-top:12px;'>
                        Fill in the transaction details<br>and click Run Fraud Check
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

# =============================================================================
# PAGE 2 — Batch Upload & Results
# =============================================================================
elif "Batch" in page:
    st.markdown(
        '<div class="page-title">📂 Batch Upload & Results</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Upload a CSV of transactions for bulk Stacking-Diversity Ensemble fraud prediction.</div>',
        unsafe_allow_html=True,
    )

    if not api_ok:
        st.error("⚠️ API is offline. Start the FastAPI server with `python main.py`.")
        st.stop()

    # Expected columns info
    with st.expander("📋 Expected CSV Columns (9 fields)"):
        st.code(
            """step, type, amount, oldbalanceOrg, newbalanceOrig,
oldbalanceDest, newbalanceDest, user_tx_count, user_avg_amount"""
        )
        # Sample uses real rows from the notebook dataset
        sample_csv = pd.DataFrame(
            [
                {
                    "step": 0,
                    "type": "TRANSFER",
                    "amount": 20674.33,
                    "oldbalanceOrg": 20674.33,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 87.70,
                    "newbalanceDest": 20762.03,
                    "user_tx_count": 1.0,
                    "user_avg_amount": 20674.33,
                },
                {
                    "step": 0,
                    "type": "PAYMENT",
                    "amount": 495.47,
                    "oldbalanceOrg": 243425.93,
                    "newbalanceOrig": 242930.46,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 495.47,
                    "user_tx_count": 8.0,
                    "user_avg_amount": 10604.77,
                },
                {
                    "step": 0,
                    "type": "TRANSFER",
                    "amount": 34461.78,
                    "oldbalanceOrg": 242930.46,
                    "newbalanceOrig": 208468.68,
                    "oldbalanceDest": 86.27,
                    "newbalanceDest": 34548.05,
                    "user_tx_count": 8.0,
                    "user_avg_amount": 10604.77,
                },
                {
                    "step": 0,
                    "type": "TRANSFER",
                    "amount": 503.98,
                    "oldbalanceOrg": 503.98,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 33.83,
                    "newbalanceDest": 537.81,
                    "user_tx_count": 1.0,
                    "user_avg_amount": 503.98,
                },
                {
                    "step": 0,
                    "type": "PAYMENT",
                    "amount": 575.87,
                    "oldbalanceOrg": 86417.85,
                    "newbalanceOrig": 85841.98,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 575.87,
                    "user_tx_count": 95.0,
                    "user_avg_amount": 10366.62,
                },
            ]
        )
        st.dataframe(sample_csv, use_container_width=True)
        st.download_button(
            "⬇️ Download Sample CSV",
            data=sample_csv.to_csv(index=False),
            file_name="sample_transactions.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.markdown(
            f'<div class="section-header">Preview — {len(df_upload):,} transactions loaded</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(df_upload.head(5), use_container_width=True)

        if st.button("⚡ Run Batch Prediction"):
            if len(df_upload) > 1000:
                st.warning("⚠️ Only the first 1,000 rows will be processed (API limit).")
                df_upload = df_upload.head(1000)

            with st.spinner(f"Processing {len(df_upload):,} transactions..."):
                try:
                    payload = df_upload.to_dict(orient="records")
                    resp = requests.post(
                        f"{API_BASE}/predict/batch", json=payload, timeout=60
                    )
                    data = resp.json()

                    if resp.status_code == 200:
                        summary = data["summary"]
                        results = pd.DataFrame(data["results"])

                        # Summary metrics
                        st.markdown(
                            '<div class="section-header" style="margin-top:24px;">Summary</div>',
                            unsafe_allow_html=True,
                        )
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric(
                            "Total Transactions", f"{summary['total_transactions']:,}"
                        )
                        m2.metric(
                            "🚨 Fraud Detected",
                            f"{summary['fraud_detected']:,}",
                            delta=f"{summary['fraud_detected']/summary['total_transactions']*100:.2f}%",
                            delta_color="inverse",
                        )
                        m3.metric("✅ Legitimate", f"{summary['no_fraud']:,}")
                        m4.metric(
                            "Fraud Rate",
                            f"{summary['fraud_detected']/summary['total_transactions']*100:.2f}%",
                        )
                        m5.metric(
                            "Total Latency", f"{summary['total_latency_ms']:.1f} ms"
                        )

                        # Charts
                        st.markdown(
                            '<div class="section-header" style="margin-top:24px;">Visual Breakdown</div>',
                            unsafe_allow_html=True,
                        )
                        ch1, ch2 = st.columns(2)

                        with ch1:
                            fig_pie = px.pie(
                                values=[summary["no_fraud"], summary["fraud_detected"]],
                                names=["0 - No Fraud", "1 - Fraud"],
                                color_discrete_sequence=["#22c55e", "#ef4444"],
                                title="Fraud vs Legitimate",
                            )
                            fig_pie.update_layout(
                                paper_bgcolor="#0a0e1a",
                                plot_bgcolor="#0a0e1a",
                                font_color="#94a3b8",
                                title_font_color="#e2e8f0",
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with ch2:
                            risk_counts = (
                                results["risk_level"].value_counts().reset_index()
                            )
                            risk_counts.columns = ["Risk Level", "Count"]
                            color_map = {
                                "HIGH": "#ef4444",
                                "MEDIUM": "#f59e0b",
                                "LOW": "#22c55e",
                            }
                            fig_risk = px.bar(
                                risk_counts,
                                x="Risk Level",
                                y="Count",
                                color="Risk Level",
                                color_discrete_map=color_map,
                                title="Transactions by Risk Level",
                            )
                            fig_risk.update_layout(
                                paper_bgcolor="#0a0e1a",
                                plot_bgcolor="#0f1629",
                                font_color="#94a3b8",
                                title_font_color="#e2e8f0",
                                showlegend=False,
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)

                        # Fraud probability distribution
                        fig_hist = px.histogram(
                            results,
                            x="fraud_probability",
                            nbins=50,
                            color_discrete_sequence=["#38bdf8"],
                            title="Distribution of Fraud Probability Scores",
                            labels={"fraud_probability": "Fraud Probability"},
                        )
                        fig_hist.update_layout(
                            paper_bgcolor="#0a0e1a",
                            plot_bgcolor="#0f1629",
                            font_color="#94a3b8",
                            title_font_color="#e2e8f0",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Results table
                        st.markdown(
                            '<div class="section-header">Full Results</div>',
                            unsafe_allow_html=True,
                        )
                        df_display = df_upload.copy().reset_index(drop=True)
                        df_display["prediction_label"] = results["prediction_label"]
                        df_display["fraud_probability"] = results["fraud_probability"]
                        df_display["risk_level"] = results["risk_level"]

                        def colour_rows(row):
                            if row["prediction_label"] == "1 - Fraud":
                                return ["background-color: #2d0a0a"] * len(row)
                            return ["background-color: #0a2d1a"] * len(row)

                        st.dataframe(
                            df_display.style.apply(colour_rows, axis=1),
                            use_container_width=True,
                        )

                        # Download
                        st.download_button(
                            "⬇️ Download Results CSV",
                            data=df_display.to_csv(index=False),
                            file_name="fraud_predictions.csv",
                            mime="text/csv",
                        )

                    else:
                        st.error(f"API Error: {data.get('detail', 'Unknown error')}")

                except Exception as e:
                    st.error(f"Request failed: {str(e)}")

# =============================================================================
# PAGE 3 — Model Performance
# =============================================================================
elif "Performance" in page:
    st.markdown(
        '<div class="page-title">📊 Model Performance</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="page-subtitle">Comparative metrics from notebook (baseline evaluation) — validation set (318,160 transactions).</div>',
        unsafe_allow_html=True,
    )

    # ── Selected model hero metrics ──
    st.markdown(
        '<div class="section-header">Selected Model — Stacking-Diversity (Baseline) ★  |  Best on all metrics</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("PR-AUC", "1.0000", "✓ Best")
    m2.metric("F1-Score", "0.9999", "✓ Best")
    m3.metric("Precision", "0.9997", "✓ Best")
    m4.metric("Recall", "1.0000", "✓ Best")
    m5.metric("ROC-AUC", "1.0000", "✓ Best")
    m6.metric("Brier Score", "0.0549", "✓ Lowest")

    st.info(
        "ℹ️ Stacking-Diversity Ensemble (Baseline) dominates every metric. "
        "Base learners: XGBoost · LightGBM · Random Forest · Logistic Regression. "
        "Meta-learner: Logistic Regression. Threshold: 0.5492. "
        "7 missed frauds out of 214,051 (FNR = 0.003%)."
    )

    # ── View toggle ──
    view_mode = st.radio(
        "Show results for:",
        ["Baseline Models", "Tuned Models", "Baseline vs Tuned Comparison"],
        horizontal=True,
    )

    if view_mode == "Baseline Models":
        comp_df = pd.DataFrame(MODEL_RESULTS).T
        title_str = "Baseline Model Performance — Validation Set (318,160 records)"
        highlight_label = "Stacking-Diversity (Baseline) ★"
    elif view_mode == "Tuned Models":
        comp_df = pd.DataFrame(TUNED_RESULTS).T
        title_str = "Tuned Model Performance — Validation Set (318,160 records)"
        highlight_label = "Stacking-Diversity (Tuned)"
    else:
        comp_df = pd.concat(
            [
                pd.DataFrame(MODEL_RESULTS).T,
                pd.DataFrame(TUNED_RESULTS).T,
            ]
        )
        title_str = "Baseline vs Tuned — All Models Comparison (Validation Set)"
        highlight_label = "Stacking-Diversity (Baseline) ★"

    # ── Metrics table ──
    st.markdown(
        f'<div class="section-header" style="margin-top:24px;">{title_str}</div>',
        unsafe_allow_html=True,
    )
    highlight_cols = [
        c
        for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
        if c in comp_df.columns
    ]
    low_cols = [c for c in ["Brier Score", "Latency (ms)"] if c in comp_df.columns]
    st.dataframe(
        comp_df.style.highlight_max(
            subset=highlight_cols, color="#0a2d1a"
        ).highlight_min(subset=low_cols, color="#0a2d1a"),
        use_container_width=True,
    )

    # ── Radar chart ──
    st.markdown(
        '<div class="section-header" style="margin-top:24px;">Model Comparison Radar</div>',
        unsafe_allow_html=True,
    )
    radar_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    colors = [
        "#38bdf8",
        "#22c55e",
        "#f59e0b",
        "#a78bfa",
        "#fb7185",
        "#34d399",
        "#60a5fa",
        "#4ade80",
        "#fbbf24",
        "#c084fc",
        "#f87171",
        "#2dd4bf",
    ]
    fig_radar = go.Figure()
    for idx, model_name in enumerate(comp_df.index):
        vals = comp_df.loc[model_name, radar_metrics].tolist()
        vals += [vals[0]]
        is_best = "Stacking" in model_name and "Baseline" in model_name
        fig_radar.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=radar_metrics + [radar_metrics[0]],
                name=model_name,
                line=dict(color=colors[idx % len(colors)], width=3 if is_best else 1.2),
                fill="toself",
                fillcolor=colors[idx % len(colors)],
                opacity=0.25 if is_best else 0.06,
            )
        )
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0f1629",
            radialaxis=dict(
                visible=True, range=[0, 1], color="#334155", gridcolor="#1e2d4a"
            ),
            angularaxis=dict(color="#64748b", gridcolor="#1e2d4a"),
        ),
        paper_bgcolor="#0a0e1a",
        font_color="#94a3b8",
        legend=dict(bgcolor="#0f1629", bordercolor="#1e3a5f", font_color="#94a3b8"),
        height=500,
        title=f"Radar — {title_str}",
        title_font_color="#e2e8f0",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── PR-AUC bar chart ──
    st.markdown(
        '<div class="section-header" style="margin-top:24px;">PR-AUC — Primary Metric Ranking</div>',
        unsafe_allow_html=True,
    )
    prauc_sorted = comp_df["PR-AUC"].sort_values(ascending=True)
    bar_colors = [
        "#ef4444" if "Stacking" in n and "Baseline" in n else "#38bdf8"
        for n in prauc_sorted.index
    ]
    fig_prauc = go.Figure(
        go.Bar(
            x=prauc_sorted.values,
            y=prauc_sorted.index,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in prauc_sorted.values],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=10),
        )
    )
    fig_prauc.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f1629",
        font_color="#94a3b8",
        title_font_color="#e2e8f0",
        height=380,
        xaxis=dict(range=[0, 1.05]),
        margin=dict(t=20, b=20, l=20, r=60),
    )
    st.plotly_chart(fig_prauc, use_container_width=True)

    # ── Confusion matrix (Stacking Baseline) ──
    st.markdown(
        '<div class="section-header" style="margin-top:24px;">Stacking-Diversity (Baseline) — Confusion Matrix</div>',
        unsafe_allow_html=True,
    )
    cm_data = [[CM_TN, CM_FP], [CM_FN, CM_TP]]
    fig_cm = go.Figure(
        go.Heatmap(
            z=cm_data,
            x=["Predicted: 0 — No Fraud", "Predicted: 1 — Fraud"],
            y=["Actual: 0 — No Fraud", "Actual: 1 — Fraud"],
            colorscale="Blues",
            text=[[f"{v:,}" for v in row] for row in cm_data],
            texttemplate="%{text}",
            textfont={"size": 16},
        )
    )
    fig_cm.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f1629",
        font_color="#94a3b8",
        title_font_color="#e2e8f0",
        height=360,
        title=f"Threshold = {BEST_THRESHOLD}  |  TP:{CM_TP:,}  FP:{CM_FP:,}  TN:{CM_TN:,}  FN:{CM_FN:,}  |  FNR = 0.003%",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Dashboard images from notebook ──
    col_imgs = {
        "Baseline Dashboard": "Graphs/dashboard_baseline_v3.png",
        "Tuned Dashboard": "Graphs/dashboard_tuned_v3.png",
        "Baseline vs Tuned": "Graphs/baseline_vs_tuned_comparison_v3.png",
    }
    any_img = any(os.path.exists(p) for p in col_imgs.values())
    if any_img:
        st.markdown(
            '<div class="section-header" style="margin-top:24px;">Notebook Output Charts</div>',
            unsafe_allow_html=True,
        )
        for label, path in col_imgs.items():
            if os.path.exists(path):
                st.markdown(f"**{label}**")
                st.image(path, use_column_width=True)

# =============================================================================
# PAGE 4 — SHAP Feature Importance
# =============================================================================
elif "SHAP" in page:
    st.markdown(
        '<div class="page-title">🔬 SHAP Feature Importance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Stacking-Diversity Ensemble explainability — what drives fraud predictions? (notebook Cell [105], PermutationExplainer, 2,000 val samples)</div>',
        unsafe_allow_html=True,
    )

    # Always show actual SHAP results from notebook
    st.markdown(
        '<div class="section-header">Top 10 Features — Actual Mean |SHAP| Values (Stacking-Diversity Ensemble, 2,000 val samples)</div>',
        unsafe_allow_html=True,
    )
    shap_df = pd.DataFrame(SHAP_FEATURES)

    fig_shap = px.bar(
        shap_df.sort_values("Mean |SHAP|"),
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        color="Mean |SHAP|",
        color_continuous_scale="Blues",
        text=[f"{v:.4f}" for v in shap_df.sort_values("Mean |SHAP|")["Mean |SHAP|"]],
        title="Stacking-Diversity Ensemble — SHAP Feature Importance",
    )
    fig_shap.update_traces(textposition="outside")
    fig_shap.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f1629",
        font_color="#94a3b8",
        title_font_color="#e2e8f0",
        coloraxis_showscale=False,
        height=420,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # SHAP PNG plots
    shap_files = {
        "Summary (Beeswarm)": os.path.join("shap_summary.png"),
        "Feature Importance (Bar)": os.path.join("shap_bar.png"),
        "Force Plot — Example Fraud": os.path.join("shap_force.png"),
    }

    any_found = any(os.path.exists(p) for p in shap_files.values())

    if any_found:
        for title, path in shap_files.items():
            if os.path.exists(path):
                st.markdown(
                    f'<div class="section-header">{title}</div>', unsafe_allow_html=True
                )
                st.image(path, use_column_width=True)

                with open(path, "rb") as f:
                    st.download_button(
                        f"⬇️ Download {title}",
                        data=f,
                        file_name=os.path.basename(path),
                        mime="image/png",
                    )
                st.markdown("---")
    else:
        st.info(
            "💡 SHAP plots not found. Run Phase 9 (SHAP) in your notebook to generate: "
            "shap_summary.png, shap_bar.png, shap_force.png"
        )

    # Top parameters table — updated with actual notebook results
    st.markdown(
        '<div class="section-header">Top Parameters Influencing the Model</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        shap_df[["Feature", "Mean |SHAP|", "Influence", "Interpretation"]],
        use_container_width=True,
        hide_index=True,
    )
