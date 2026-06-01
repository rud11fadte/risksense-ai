import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import predict, model_metrics
from visualisations import (gauge_chart, shap_waterfall,
                             feature_importance_chart, roc_placeholder)
from report import generate_pdf

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RiskSense AI",
    page_icon="🏦",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .risk-box {
        padding: 1.5rem; border-radius: 12px;
        text-align: center; margin: 1rem 0;
    }
    .metric-card {
        background: white; padding: 1rem;
        border-radius: 10px; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab"] { font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("# 🏦 RiskSense AI")
st.markdown("*Explainable ML for Credit Risk Assessment*")
st.divider()

# ── Navigation tabs ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Predict Risk",
    "📊 Model Dashboard",
    "ℹ️ About"
])

# ════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 40,
            help="Applicant's age in years")
        income = st.number_input("Monthly Income (USD)",
            0, 100000, 5000, step=500,
            help="Gross monthly income")
        debt_ratio = st.slider("Debt Ratio", 0.0, 2.0, 0.3,
            step=0.01,
            help="Monthly debt payments / Monthly income")
        revolving = st.slider(
            "Revolving Credit Utilization",
            0.0, 1.0, 0.3, step=0.01,
            help="Credit card usage as % of limit")
        open_credit = st.slider(
            "Open Credit Lines & Loans",
            0, 30, 5,
            help="Total number of open credit accounts")

    with col2:
        real_estate = st.slider("Real Estate Loans", 0, 10, 1,
            help="Number of mortgage/real estate loans")
        dependents = st.slider("Number of Dependents", 0, 10, 1,
            help="Number of financial dependents")
        late_30_59 = st.slider(
            "Times 30-59 Days Late", 0, 10, 0,
            help="Number of times 30-59 days past due")
        late_60_89 = st.slider(
            "Times 60-89 Days Late", 0, 10, 0,
            help="Number of times 60-89 days past due")
        late_90 = st.slider(
            "Times 90+ Days Late", 0, 10, 0,
            help="Number of times 90+ days past due")

    st.divider()
    predict_btn = st.button("🔍 Assess Risk", 
                             use_container_width=True,
                             type="primary")

    if predict_btn:
        with st.spinner("Analysing applicant profile..."):
            result = predict(
                revolving, age, late_30_59, debt_ratio,
                income, open_credit, late_90, real_estate,
                late_60_89, dependents
            )

        # ── Risk verdict ──────────────────────────────────
        st.divider()
        c1, c2, c3 = st.columns([1, 2, 1])

        with c2:
            st.markdown(
                f"<div class='risk-box' style='background:"
                f"{result['risk_color']}22; border: 2px solid "
                f"{result['risk_color']}'>"
                f"<h1 style='color:{result['risk_color']};margin:0'>"
                f"{result['risk_emoji']} {result['risk_label']}</h1>"
                f"<p style='font-size:18px;margin:8px 0 0'>Default "
                f"Probability: <b>{result['probability']}%</b></p>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ── Gauge ─────────────────────────────────────────
        col_g, col_s = st.columns(2)

        with col_g:
            st.plotly_chart(
                gauge_chart(result['probability'],
                            result['risk_color']),
                use_container_width=True
            )

        with col_s:
            st.plotly_chart(
                shap_waterfall(result['shap_df']),
                use_container_width=True
            )

        # ── Key metrics ───────────────────────────────────
        st.divider()
        st.subheader("📋 Applicant Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Age",            f"{age} yrs")
        m2.metric("Monthly Income", f"${income:,}")
        m3.metric("Debt Ratio",     f"{debt_ratio:.2f}")
        m4.metric("Total Late Payments",
                  int(late_30_59 + late_60_89 + late_90))

        # ── PDF download ──────────────────────────────────
        st.divider()
        inputs = {
            'Age': age, 'Monthly Income': income,
            'Debt Ratio': debt_ratio,
            'Revolving Utilization': revolving,
            'Open Credit Lines': open_credit,
            'Real Estate Loans': real_estate,
            'Dependents': dependents,
            'Times 30-59 Days Late': late_30_59,
            'Times 60-89 Days Late': late_60_89,
            'Times 90+ Days Late': late_90
        }

        pdf_buffer = generate_pdf(result, inputs)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"risksense_report_{age}yrs.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ════════════════════════════════════════════════════
# TAB 2 — MODEL DASHBOARD
# ════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Dashboard")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model",     "Random Forest")
    m2.metric("AUC-ROC",   model_metrics['auc'])
    m3.metric("F1 Score",  model_metrics['f1'])
    m4.metric("Training Samples", "223,956")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(roc_placeholder(),
                        use_container_width=True)
    with col_b:
        st.plotly_chart(feature_importance_chart(model_metrics),
                        use_container_width=True)

    st.divider()
    st.subheader("📐 Preprocessing Pipeline")
    st.markdown("""
    | Step | Method | Why |
    |------|--------|-----|
    | Missing Values | Median by age group | Income varies heavily by age |
    | Class Imbalance | SMOTE | 6.68% default rate — model needs balance |
    | Feature Scaling | StandardScaler | Required for distance-based models |
    | Feature Engineering | 4 new features | Capture risk signals better |
    | Train/Test Split | 80/20 stratified | Preserves class ratio in both sets |
    """)

# ════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════
with tab3:
    st.subheader("About RiskSense AI")
    st.markdown("""
    **RiskSense AI** is an end-to-end machine learning system for 
    credit risk assessment, built as part of an MSc Data Science portfolio.

    ### What It Does
    Predicts the probability that a loan applicant will default within 
    2 years, with full explainability showing *why* each decision was made.

    ### Tech Stack
    - **ML Model:** Random Forest (AUC-ROC: 0.8496)
    - **Explainability:** SHAP (SHapley Additive exPlanations)
    - **Data:** 150,000 real loan applicants (Kaggle — Give Me Some Credit)
    - **Framework:** Streamlit + Plotly
    - **Preprocessing:** SMOTE, feature engineering, StandardScaler

    ### Key Features
    - Real-time risk prediction with probability score
    - Animated gauge with risk categories (Low / Medium / High)
    - SHAP waterfall chart — explains every individual prediction
    - Downloadable PDF report per applicant
    - Model performance dashboard

    ### Dataset
    [Give Me Some Credit — Kaggle](https://kaggle.com/c/GiveMeSomeCredit)

    ### Built By
    MSc Data Science Student | BSc Mathematics  
    📧 your.email@gmail.com  
    🔗 [GitHub](https://github.com/yourusername)  
    💼 [LinkedIn](https://linkedin.com/in/yourusername)
    """)