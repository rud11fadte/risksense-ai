import numpy as np
import pandas as pd
import joblib
import os

# Load all models once when app starts
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, 'models')
REQUIRED_FILES = [
    'best_model.pkl',
    'scaler.pkl',
    'feature_names.pkl',
    'model_metrics.pkl',
    'shap_explainer.pkl'
]

missing_files = [
    name for name in REQUIRED_FILES
    if not os.path.exists(os.path.join(MODELS, name))
]

if missing_files:
    model = None
    scaler = None
    feature_names = None
    model_metrics = None
    explainer = None
    MISSING_MODEL_MESSAGE = (
        "Missing trained model artifacts in the 'models' folder. "
        "Please run the preprocessing, modelling, and SHAP notebooks first "
        "to generate: " + ", ".join(missing_files)
    )
else:
    model = joblib.load(os.path.join(MODELS, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(MODELS, 'feature_names.pkl'))
    model_metrics = joblib.load(os.path.join(MODELS, 'model_metrics.pkl'))
    explainer = joblib.load(os.path.join(MODELS, 'shap_explainer.pkl'))
    MISSING_MODEL_MESSAGE = None


def build_input(revolving, age, late_30_59, debt_ratio,
                income, open_credit, late_90, real_estate,
                late_60_89, dependents):
    """Convert raw user inputs into model-ready feature vector."""

    # Engineered features — same as Notebook 2
    total_late       = late_30_59 + late_60_89 + late_90
    income_debt      = income / (debt_ratio + 1)
    util_per_account = revolving / (open_credit + 1)
    ever_late        = 1 if late_90 > 0 else 0

    raw = [revolving, age, late_30_59, debt_ratio, income,
           open_credit, late_90, real_estate, late_60_89,
           dependents, total_late, income_debt,
           util_per_account, ever_late]

    df = pd.DataFrame([raw], columns=feature_names)
    scaled = scaler.transform(df)
    return df, scaled


def predict(revolving, age, late_30_59, debt_ratio,
            income, open_credit, late_90, real_estate,
            late_60_89, dependents):
    """Run prediction and return full result dict."""

    if model is None or scaler is None or feature_names is None or explainer is None:
        raise FileNotFoundError(MISSING_MODEL_MESSAGE)

    df_raw, df_scaled = build_input(
        revolving, age, late_30_59, debt_ratio,
        income, open_credit, late_90, real_estate,
        late_60_89, dependents
    )

    prob        = model.predict_proba(df_scaled)[0][1]
    prediction  = int(prob >= 0.5)

    # Risk category
    if prob < 0.3:
        risk_label = "LOW RISK"
        risk_color = "#2ECC71"
        risk_emoji = "✅"
    elif prob < 0.6:
        risk_label = "MEDIUM RISK"
        risk_color = "#F39C12"
        risk_emoji = "⚠️"
    else:
        risk_label = "HIGH RISK"
        risk_color = "#E74C3C"
        risk_emoji = "🚨"

    # SHAP values for this applicant
    shap_vals = explainer.shap_values(df_scaled)
    if isinstance(shap_vals, list):
        shap_default = shap_vals[1][0]
    elif len(np.array(shap_vals).shape) == 3:
        shap_default = np.array(shap_vals)[0, :, 1]
    else:
        shap_default = shap_vals[0]

    shap_df = pd.DataFrame({
        'Feature':    feature_names,
        'SHAP Value': shap_default,
        'Raw Value':  df_raw.values[0]
    }).sort_values('SHAP Value', key=abs, ascending=False)

    return {
        'probability':  round(float(prob) * 100, 1),
        'prediction':   prediction,
        'risk_label':   risk_label,
        'risk_color':   risk_color,
        'risk_emoji':   risk_emoji,
        'shap_df':      shap_df,
        'model_metrics': model_metrics
    }