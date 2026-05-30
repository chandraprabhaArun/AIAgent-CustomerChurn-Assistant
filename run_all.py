"""
run_all.py — Execute entire TeleConnect project in one command.

Usage:
    cd /Users/shubhamchaudhari/project_vault/EDA_PROJECT
    source venv/bin/activate
    python run_all.py
"""
import pandas as pd
import numpy as np
import joblib
import yaml
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

print("""
╔══════════════════════════════════════════════════════════╗
║   🔮 TeleConnect — Full Pipeline Execution              ║
║   Part 1: Churn Model + Part 2: Retention Agent         ║
╚══════════════════════════════════════════════════════════╝
""")

# ═══════════════════════════════════════════════════════
# PART 1.1 — DATA CLEANING
# ═══════════════════════════════════════════════════════
print("=" * 58)
print("  📊 PART 1.1 — Data Cleaning")
print("=" * 58)

df_raw = pd.read_csv(os.path.join(ROOT, 'test_datafile.csv'))
print(f"  Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

df = df_raw.copy()

# Gender: 9 variants → Male/Female/Other/Unknown
gender_map = {'male': 'Male', 'Male': 'Male', 'M': 'Male', 'm': 'Male', 'MALE': 'Male',
              'female': 'Female', 'Female': 'Female', 'F': 'Female', 'f': 'Female', 'Other': 'Other'}
df['gender'] = df['gender'].replace('', np.nan).map(gender_map).fillna('Unknown')

# Internet: lowercase + merge
internet_map = {'DSL': 'DSL', 'dsl': 'DSL', 'Fiber optic': 'Fiber optic', 'fiber': 'Fiber optic',
                'No': 'No', 'None': 'No', 'nan': np.nan}
df['internet_service'] = df['internet_service'].map(internet_map).fillna('No')

# Phone: normalize
phone_map = {'Yes': 'Yes', 'yes': 'Yes', 'Y': 'Yes', 'No': 'No', 'no': 'No', 'N': 'No'}
df['phone_service'] = df['phone_service'].map(phone_map)

# Payment: standardize
payment_map = {'Bank transfer': 'Bank transfer', 'bank transfer': 'Bank transfer', 'BT': 'Bank transfer',
               'Credit card': 'Credit card', 'credit card': 'Credit card', 'CC': 'Credit card',
               'Electronic check': 'Electronic check', 'Mailed check': 'Mailed check'}
df['payment_method'] = df['payment_method'].map(payment_map)

# Impossible values → NaN
df.loc[(df['age'] < 18) | (df['age'] > 100), 'age'] = np.nan
df.loc[(df['tenure_months'] < 0) | (df['tenure_months'] > 72), 'tenure_months'] = np.nan
df.loc[(df['satisfaction_score'] < 1) | (df['satisfaction_score'] > 10), 'satisfaction_score'] = np.nan

# Impute
for col in ['age', 'tenure_months', 'monthly_charges', 'total_charges',
            'avg_monthly_gb_used', 'avg_monthly_minutes', 'satisfaction_score']:
    df[col] = df[col].fillna(df[col].median())
for col in ['gender', 'internet_service', 'phone_service', 'payment_method']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Engineered features
df['charge_per_tenure'] = df['monthly_charges'] / df['tenure_months'].clip(lower=1)
df['support_rate'] = df['num_support_tickets'] / df['tenure_months'].clip(lower=1)
df['tenure_x_charges'] = df['tenure_months'] * df['monthly_charges']
df['satisfaction_x_tickets'] = df['satisfaction_score'] * df['num_support_tickets']
df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)
df['is_high_spender'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)
df['low_satisfaction'] = (df['satisfaction_score'] <= 4).astype(int)
df['charges_to_total_ratio'] = df['monthly_charges'] / df['total_charges'].clip(lower=1)
df['gb_per_charge'] = df['avg_monthly_gb_used'] / df['monthly_charges'].clip(lower=1)
df['minutes_per_charge'] = df['avg_monthly_minutes'] / df['monthly_charges'].clip(lower=1)

df.to_csv(os.path.join(ROOT, 'data', 'cleaned_data.csv'), index=False)
print(f"  ✅ Cleaned: {df.shape[0]} rows, {df.isnull().sum().sum()} nulls remaining")
print(f"  ✅ Saved: data/cleaned_data.csv")

# ═══════════════════════════════════════════════════════
# PART 1.2 — EDA SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 58}")
print("  📊 PART 1.2 — EDA Summary")
print("=" * 58)

churn_rate = df['churned'].mean()
print(f"  Churn Rate: {churn_rate:.1%} ({df['churned'].sum()} / {len(df)})")
print(f"  Top Features: contract_type, satisfaction_score, tenure_months, support_tickets, monthly_charges")

# ═══════════════════════════════════════════════════════
# PART 1.3 — MODEL TRAINING
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 58}")
print("  📊 PART 1.3 — Model Training (Decision Tree + Random Forest)")
print("=" * 58)

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

feature_cols_num = ['age', 'tenure_months', 'monthly_charges', 'total_charges',
    'avg_monthly_gb_used', 'num_support_tickets', 'avg_monthly_minutes',
    'satisfaction_score', 'num_additional_services', 'charge_per_tenure', 'support_rate',
    'tenure_x_charges', 'satisfaction_x_tickets', 'is_new_customer',
    'is_high_spender', 'low_satisfaction', 'charges_to_total_ratio',
    'gb_per_charge', 'minutes_per_charge']
feature_cols_cat = ['gender', 'contract_type', 'internet_service', 'phone_service', 'payment_method']

X = df[feature_cols_num + feature_cols_cat]
y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), feature_cols_num),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), feature_cols_cat)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree
print("  Training Decision Tree (5-fold CV, 40 configs)...")
dt_search = RandomizedSearchCV(
    Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))]),
    {'clf__max_depth': [5, 7, 10, 12], 'clf__min_samples_split': [10, 20, 30],
     'clf__min_samples_leaf': [5, 10, 15], 'clf__class_weight': ['balanced', {0:1, 1:2}],
     'clf__criterion': ['gini', 'entropy']},
    n_iter=40, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)
dt_search.fit(X_train, y_train)
dt_prob = dt_search.predict_proba(X_test)[:, 1]
dt_pred = dt_search.predict(X_test)
dt_auc = roc_auc_score(y_test, dt_prob)
dt_acc = accuracy_score(y_test, dt_pred)
dt_rec = recall_score(y_test, dt_pred)
print(f"  ✅ Decision Tree — Accuracy: {dt_acc:.4f} | AUC-ROC: {dt_auc:.4f} | Recall: {dt_rec:.2f}")

# Random Forest
print("  Training Random Forest (5-fold CV, 60 configs)...")
rf_search = RandomizedSearchCV(
    Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))]),
    {'clf__n_estimators': [300, 500, 700, 1000], 'clf__max_depth': [10, 15, 20, None],
     'clf__min_samples_split': [2, 5, 10], 'clf__min_samples_leaf': [1, 2, 4],
     'clf__max_features': ['sqrt', 0.5, 0.7], 'clf__class_weight': ['balanced', 'balanced_subsample']},
    n_iter=60, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
rf_prob = rf_search.predict_proba(X_test)[:, 1]
rf_pred = rf_search.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_prob)
rf_acc = accuracy_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
print(f"  ✅ Random Forest — Accuracy: {rf_acc:.4f} | AUC-ROC: {rf_auc:.4f} | Recall: {rf_rec:.2f}")

# Pick best
print(f"\n  {'Model':<18} {'Accuracy':<12} {'AUC-ROC':<12} {'Recall':<10}")
print(f"  {'-'*50}")
print(f"  {'Decision Tree':<18} {dt_acc:<12.4f} {dt_auc:<12.4f} {dt_rec:<10.2f}")
print(f"  {'Random Forest':<18} {rf_acc:<12.4f} {rf_auc:<12.4f} {rf_rec:<10.2f}")

if rf_auc >= dt_auc:
    best_name, best_model, best_prob, best_pred = 'Random Forest', rf_search.best_estimator_, rf_prob, rf_pred
else:
    best_name, best_model, best_prob, best_pred = 'Decision Tree', dt_search.best_estimator_, dt_prob, dt_pred

print(f"\n  🏆 Selected: {best_name}")

# ═══════════════════════════════════════════════════════
# PART 1.4 — RESULTS
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 58}")
print(f"  📊 PART 1.4 — Final Model Results ({best_name})")
print("=" * 58)
print(classification_report(y_test, best_pred, target_names=['Retained', 'Churned']))
print(f"  AUC-ROC:  {roc_auc_score(y_test, best_prob):.4f}")
print(f"  Accuracy: {accuracy_score(y_test, best_pred):.4f}")

# ═══════════════════════════════════════════════════════
# PART 1.5 — SAVE MODEL
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 58}")
print("  📊 PART 1.5 — Export Model")
print("=" * 58)

ohe = preprocessor.fit(X_train).transformers_[1][1].get_feature_names_out(feature_cols_cat)
all_features = feature_cols_num + list(ohe)

joblib.dump(best_model, os.path.join(ROOT, 'models', 'churn_model.joblib'))
joblib.dump({'feature_cols_num': feature_cols_num, 'feature_cols_cat': feature_cols_cat,
             'all_features': all_features, 'model_type': best_name},
            os.path.join(ROOT, 'models', 'model_metadata.joblib'))
print(f"  ✅ Saved: models/churn_model.joblib ({best_name})")
print(f"  ✅ Saved: models/model_metadata.joblib")

# ═══════════════════════════════════════════════════════
# PART 2.1 — AGENT TOOLS TEST
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 58}")
print("  🤖 PART 2.1 — Agent Tools Test")
print("=" * 58)

from part2.tools import lookup_customer, predict_churn, get_retention_offers, log_interaction, escalate_to_supervisor

profile = lookup_customer('TC-004711')
print(f"  ✅ lookup_customer('TC-004711') → {profile['gender']}, {profile['contract_type']}, tenure={profile['tenure_months']:.0f}mo")

features = {k: v for k, v in profile.items() if k not in ['customer_id', 'last_interaction_date']}
pred = predict_churn(features)
print(f"  ✅ predict_churn() → {pred['risk_tier']} risk ({pred['churn_probability']:.0%})")

offers = get_retention_offers(pred['risk_tier'], profile['contract_type'])
print(f"  ✅ get_retention_offers() → {len(offers['offers'])} offers available")

log = log_interaction('TC-004711', 'REP-001', 'retained', ['H-MTM-01'], 'H-MTM-01', 'Customer accepted')
print(f"  ✅ log_interaction() → {log['interaction_id']}")

esc = escalate_to_supervisor('TC-004711', 'legal_threat', 'Customer threatening lawsuit')
print(f"  ✅ escalate_to_supervisor() → {esc['ticket_id']}")

# ═══════════════════════════════════════════════════════
# PART 2.2 — EVALUATION METRICS TEST
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 58}")
print("  🧪 PART 2.2 — Evaluation Framework Test")
print("=" * 58)

from part2.evaluation.metrics import evaluate_test_case
with open(os.path.join(ROOT, 'part2', 'evaluation', 'test_cases.json')) as f:
    test_cases = json.load(f)
print(f"  ✅ Loaded {len(test_cases)} test cases")

# Quick metric test
mock_output = {
    'response': 'Customer TC-004711 is high risk at 74% churn probability. Recommend offering 25% discount.',
    'tool_calls': [
        {'tool': 'lookup_customer', 'arguments': {'customer_id': 'TC-004711'}, 'result': profile},
        {'tool': 'predict_churn', 'arguments': {'customer_data': {}}, 'result': pred},
        {'tool': 'get_retention_offers', 'arguments': {'risk_tier': 'high', 'contract_type': 'Month-to-month'}, 'result': offers}
    ]
}
metrics = evaluate_test_case(test_cases[1], mock_output)
print(f"  ✅ Metrics working — Tool Accuracy: {metrics['tool_selection_accuracy']['score']:.2f} | Completeness: {metrics['response_completeness']['score']:.2f} | Hallucination: {metrics['hallucination_check']['score']:.2f}")

# ═══════════════════════════════════════════════════════
# PART 2.3 — STREAMLIT APP CHECK
# ═══════════════════════════════════════════════════════
print(f"\n{'=' * 58}")
print("  🖥️  PART 2.3 — Streamlit App")
print("=" * 58)

import py_compile
py_compile.compile(os.path.join(ROOT, 'part2', 'app', 'main.py'), doraise=True)
print("  ✅ Streamlit app compiles OK")
print("  📌 To launch: streamlit run part2/app/main.py")
print("  📌 Open: http://localhost:8501")

# ═══════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════
print(f"""
{'=' * 58}
  ✅ ALL DONE — FULL PIPELINE EXECUTED SUCCESSFULLY!
{'=' * 58}

  📊 Part 1 Results:
     Model:    {best_name}
     Accuracy: {accuracy_score(y_test, best_pred):.4f}
     AUC-ROC:  {roc_auc_score(y_test, best_prob):.4f}
     Recall:   {recall_score(y_test, best_pred):.2f}

  🤖 Part 2 Results:
     Tools:      5/5 working
     Test Cases: {len(test_cases)} loaded
     Metrics:    3/3 working
     App:        Ready at localhost:8501

  📌 Next Steps:
     1. streamlit run part2/app/main.py  (Demo Mode ON)
     2. Open http://localhost:8501
     3. Try: "TC-004711 wants to cancel. What should I do?"
""")
