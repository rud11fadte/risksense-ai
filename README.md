# 🏦 RiskSense AI — Explainable Credit Risk Prediction

> Predict loan default probability with full SHAP explainability.
> Built on 150,000 real applicants using Random Forest + SHAP + Streamlit.

🚀 **Live Demo:** [Add HuggingFace link here]
📁 **Dataset:** [Give Me Some Credit — Kaggle](https://kaggle.com/c/GiveMeSomeCredit)

---

## What It Does

RiskSense AI assesses the probability that a loan applicant will default
within 2 years — and explains *why*, using SHAP explainability.

- Input 10 applicant details via an interactive form
- Get an instant risk score (0–100%) with animated gauge
- See a SHAP waterfall chart showing which features drove the prediction
- Download a professional PDF report per applicant
- Explore model performance on the dashboard

---

## Demo

| Low Risk Applicant | High Risk Applicant |
|---|---|
| Age 52, Income $8000, No late payments | Age 35, Income $2500, 7 late payments |
| **Risk Score: 10.1%** ✅ | **Risk Score: 89.5%** 🚨 |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | Pandas, NumPy, Plotly |
| Preprocessing | Scikit-learn, imbalanced-learn (SMOTE) |
| Modelling | Random Forest, XGBoost, Logistic Regression |
| Explainability | SHAP (TreeExplainer) |
| App | Streamlit, Plotly |
| PDF Reports | ReportLab |
| Deployment | HuggingFace Spaces |

---

## Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.8496 |
| Default Recall | 68% |
| Safe Borrower Precision | 97% |
| Training Samples (after SMOTE) | 223,956 |

> Model chosen based on **recall** on the default class — in banking,
> missing a real defaulter costs more than a false alarm.

---

## Project Structure

risksense-ai/
├── app/
│   ├── main.py              ← Streamlit app (3 tabs)
│   ├── predictor.py         ← Model loading + prediction logic
│   ├── visualisations.py    ← Plotly gauge, SHAP waterfall, charts
│   └── report.py            ← PDF report generator (ReportLab)
├── notebooks/
│   ├── 01_eda.ipynb          ← Exploratory data analysis
│   ├── 02_preprocessing.ipynb ← SMOTE, imputation, feature engineering
│   ├── 03_modelling.ipynb    ← Train + compare 3 models
│   └── 04_shap_analysis.ipynb ← SHAP explainability
├── requirements.txt
└── README.md

---

## Setup

```bash
git clone https://github.com/rud11fadte/risksense-ai
cd risksense-ai
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Download dataset from [Kaggle](https://kaggle.com/c/GiveMeSomeCredit)
and place `cs-training.csv` in `data/`

Run notebooks 01 → 02 → 03 → 04 in order to generate model files, then:

```bash
streamlit run app/main.py
```

---

## Key Concepts Demonstrated

- **Class imbalance handling** — SMOTE oversampling on training data only
- **Feature engineering** — 4 engineered features, 3 in top 5 importance
- **Model selection reasoning** — recall prioritised over F1 for banking context
- **Explainable AI** — SHAP TreeExplainer for individual predictions
- **Production patterns** — model persistence with joblib, scaler saved separately
- **End-to-end deployment** — from raw CSV to live deployable app

---

## Skills Demonstrated

`Python` `Machine Learning` `Random Forest` `XGBoost` `SHAP` 
`Feature Engineering` `SMOTE` `Plotly` `Streamlit` `ReportLab`
`Pandas` `NumPy` `Scikit-learn` `Joblib` `HuggingFace`

---

*MSc Data Science Portfolio Project | BSc Mathematics*  
*[LinkedIn](www.linkedin.com/in/rudresh-fadate) · 
[GitHub](https://github.com/rud11fadte)*