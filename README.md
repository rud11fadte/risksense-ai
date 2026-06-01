# 🏦 RiskSense AI — Explainable Credit Risk Prediction

### Predict loan default risk with transparent AI-powered explanations

RiskSense AI is an end-to-end Machine Learning application that predicts the probability of a borrower defaulting on a loan within two years and explains the prediction using SHAP Explainable AI.

Built using real-world credit data from 150,000+ applicants, the project demonstrates the complete ML lifecycle—from data preprocessing and model development to explainability, reporting, and deployment.

---

## 🚀 Live Demo

🔗 **Application:** *Add Hugging Face URL Here*

📁 **Dataset:** [Give Me Some Credit — Kaggle](https://kaggle.com/c/GiveMeSomeCredit)

---

## 📌 Business Problem

Financial institutions lose millions due to loan defaults.

Traditional credit scoring systems often provide little transparency regarding why an applicant is considered high-risk, making decision-making difficult for both analysts and customers.

RiskSense AI addresses this challenge by:

* Predicting default probability in real time
* Highlighting key risk-driving factors
* Providing explainable and auditable decisions
* Generating professional downloadable reports

---

## ✨ Features

### 🔍 Credit Risk Prediction

Predict the likelihood of default within two years using a trained Random Forest model.

### 📈 Explainable AI (SHAP)

Understand exactly why a prediction was made through feature-level explanations.

### 🎯 Risk Score Dashboard

View borrower risk on an intuitive probability gauge ranging from 0–100%.

### 📄 PDF Report Generation

Generate professional applicant risk assessment reports instantly.

### 📊 Model Analytics

Explore performance metrics, feature importance, and evaluation visualizations.

---

## 🖥️ Application Workflow

1. Enter applicant information through the Streamlit interface.
2. Model calculates default probability.
3. Risk score is displayed on an interactive gauge.
4. SHAP analysis explains prediction drivers.
5. PDF report can be downloaded for record keeping.

---

## 📸 Example Predictions

| Scenario                                | Result               |
| --------------------------------------- | -------------------- |
| Age 52, Income $8,000, No late payments | ✅ Low Risk (10.1%)   |
| Age 35, Income $2,500, 7 late payments  | 🚨 High Risk (89.5%) |

---

## 🏗️ Machine Learning Pipeline

### Data Preparation

* Missing value treatment
* Outlier handling
* Feature engineering
* Feature scaling
* Train-test split

### Class Imbalance Handling

Applied **SMOTE** exclusively on training data to improve minority class detection.

### Models Evaluated

* Logistic Regression
* Random Forest
* XGBoost

### Model Selection Strategy

For credit risk applications, identifying true defaulters is more critical than maximizing overall accuracy.

Therefore, model selection prioritized:

* Recall on the default class
* ROC-AUC score
* Business impact of missed defaults

Random Forest delivered the best balance between predictive power and explainability.

---

## 📊 Model Performance

| Metric                       | Score       |
| ---------------------------- | ----------- |
| ROC-AUC                      | **0.8496**  |
| Default Recall               | **68%**     |
| Safe Borrower Precision      | **97%**     |
| Training Samples After SMOTE | **223,956** |

### Why Recall Matters

In lending, failing to identify a genuine defaulter can lead to substantial financial losses.

A false positive may reject a good applicant, but a false negative can result in a defaulted loan.

Therefore, recall was treated as the primary optimization metric.

---

## 🧠 Explainable AI

The project uses SHAP (SHapley Additive exPlanations) to provide local interpretability for every prediction.

For each applicant:

* Positive contributors to risk are highlighted
* Negative contributors to risk are highlighted
* Individual feature impacts are quantified
* Waterfall visualizations explain model decisions

This creates a transparent and auditable prediction process.

---

## 🛠️ Tech Stack

| Category            | Technologies          |
| ------------------- | --------------------- |
| Programming         | Python                |
| Data Analysis       | Pandas, NumPy         |
| Visualization       | Plotly                |
| Machine Learning    | Scikit-learn, XGBoost |
| Imbalanced Learning | SMOTE                 |
| Explainability      | SHAP                  |
| Web App             | Streamlit             |
| Reporting           | ReportLab             |
| Deployment          | Hugging Face Spaces   |

---

## 📂 Project Structure

```text
risksense-ai/
│
├── app/
│   ├── main.py
│   ├── predictor.py
│   ├── visualisations.py
│   └── report.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_shap_analysis.ipynb
│
├── data/
├── models/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rud11fadate/risksense-ai.git
cd risksense-ai
```

Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and place:

```text
cs-training.csv
```

inside:

```text
data/
```

Run notebooks in order:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_modelling.ipynb
04_shap_analysis.ipynb
```

Launch the application:

```bash
streamlit run app/main.py
```

---

## 🎯 Skills Demonstrated

* Machine Learning
* Credit Risk Analytics
* Feature Engineering
* Class Imbalance Handling (SMOTE)
* Model Evaluation & Selection
* Explainable AI (SHAP)
* Streamlit Development
* Interactive Data Visualization
* PDF Report Automation
* Model Deployment

---

## 📚 Key Learnings

* Building interpretable ML systems for high-stakes domains
* Balancing business objectives with model metrics
* Applying SHAP for model transparency
* Deploying production-ready ML applications
* Designing end-to-end data science solutions

---

## 👨‍💻 Author

**Rudresh Fadate**

MSc Data Science | BSc Mathematics

* GitHub: https://github.com/rud11fadate
* LinkedIn: https://www.linkedin.com/in/rudresh-fadate

---

⭐ If you found this project useful, consider giving the repository a star.
