"""predict_churn.py — Callable churn prediction function for the retention agent."""
import pandas as pd
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def predict_churn(customer_data: dict) -> dict:
    """
    Accepts a dictionary of customer features. Returns:
    {
        "churn_probability": float,   # 0.0 to 1.0
        "risk_tier": str,              # "high", "medium", or "low"
        "top_risk_factors": list       # top 3 features driving this prediction
    }
    """
    model = joblib.load(os.path.join(MODEL_DIR, 'churn_model.joblib'))
    metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.joblib'))

    input_df = pd.DataFrame([customer_data])

    # Add engineered features if not present
    if 'charge_per_tenure' not in input_df.columns:
        input_df['charge_per_tenure'] = input_df['monthly_charges'] / input_df['tenure_months'].clip(lower=1)
    if 'support_rate' not in input_df.columns:
        input_df['support_rate'] = input_df['num_support_tickets'] / input_df['tenure_months'].clip(lower=1)
    if 'tenure_x_satisfaction' not in input_df.columns:
        input_df['tenure_x_satisfaction'] = input_df['tenure_months'] * input_df['satisfaction_score']
    if 'charges_per_service' not in input_df.columns:
        input_df['charges_per_service'] = input_df['monthly_charges'] / input_df['num_additional_services'].clip(lower=1)
    if 'is_new_customer' not in input_df.columns:
        input_df['is_new_customer'] = (input_df['tenure_months'] <= 6).astype(int)
    # Default new features to 0 if not provided
    for feat in ['billing_complaint_count', 'competitor_offer_received', 'network_outage_hours', 'last_plan_change_days']:
        if feat not in input_df.columns:
            input_df[feat] = 0

    input_df = input_df[metadata['feature_cols_num'] + metadata['feature_cols_cat']]

    prob = model.predict_proba(input_df)[0, 1]

    if prob >= 0.7:
        risk_tier = 'high'
    elif prob >= 0.4:
        risk_tier = 'medium'
    else:
        risk_tier = 'low'

    xgb_model = model.named_steps['clf']
    importances = pd.Series(xgb_model.feature_importances_, index=metadata['all_features'])
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return {
        'churn_probability': round(float(prob), 4),
        'risk_tier': risk_tier,
        'top_risk_factors': top_factors
    }
