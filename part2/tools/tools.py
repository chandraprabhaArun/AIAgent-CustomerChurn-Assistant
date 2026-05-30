"""Agent tools for the TeleConnect Retention Agent."""
import pandas as pd
import joblib
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'cleaned_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'churn_model.joblib')
METADATA_PATH = os.path.join(BASE_DIR, '..', 'models', 'model_metadata.joblib')

# Load dataset once for lookups
_df = None
def _get_data():
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
    return _df


# --- Tool 1: lookup_customer ---
def lookup_customer(customer_id: str) -> dict:
    """Retrieves a customer profile by ID. Returns demographics, contract, tenure, charges, satisfaction."""
    df = _get_data()
    match = df[df['customer_id'] == customer_id]
    if match.empty:
        return {"error": f"Customer '{customer_id}' not found."}
    row = match.iloc[0].to_dict()
    # Remove target column from profile
    row.pop('churned', None)
    return row


# --- Tool 2: predict_churn ---
def predict_churn(customer_data: dict) -> dict:
    """
    Accepts customer features dict. Returns:
    - churn_probability (float 0-1)
    - risk_tier (high/medium/low)
    - top_risk_factors (list of 3)
    """
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    input_df = pd.DataFrame([customer_data])

    # Engineered features
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


# --- Tool 3: get_retention_offers ---
OFFER_CATALOG = {
    "high": {
        "Month-to-month": [
            {"offer_id": "H-MTM-01", "description": "Free upgrade to 1-year contract with 25% discount for 6 months", "value": "$150"},
            {"offer_id": "H-MTM-02", "description": "Waive next 2 months + dedicated account manager", "value": "$200"},
            {"offer_id": "H-MTM-03", "description": "Free premium channel bundle for 12 months", "value": "$180"},
        ],
        "One year": [
            {"offer_id": "H-1Y-01", "description": "20% discount on renewal + free speed upgrade", "value": "$120"},
            {"offer_id": "H-1Y-02", "description": "Loyalty credit of $100 applied immediately", "value": "$100"},
        ],
        "Two year": [
            {"offer_id": "H-2Y-01", "description": "15% discount + priority support tier", "value": "$90"},
            {"offer_id": "H-2Y-02", "description": "Free equipment upgrade", "value": "$150"},
        ],
    },
    "medium": {
        "Month-to-month": [
            {"offer_id": "M-MTM-01", "description": "10% discount if switching to 1-year contract", "value": "$80"},
            {"offer_id": "M-MTM-02", "description": "Free additional service for 6 months", "value": "$60"},
        ],
        "One year": [
            {"offer_id": "M-1Y-01", "description": "Loyalty bonus: 1 month free on renewal", "value": "$70"},
        ],
        "Two year": [
            {"offer_id": "M-2Y-01", "description": "Speed upgrade at no extra cost", "value": "$50"},
        ],
    },
    "low": {
        "Month-to-month": [
            {"offer_id": "L-MTM-01", "description": "Thank-you discount: 5% off next 3 months", "value": "$30"},
        ],
        "One year": [
            {"offer_id": "L-1Y-01", "description": "Early renewal bonus: lock in current rate", "value": "$20"},
        ],
        "Two year": [
            {"offer_id": "L-2Y-01", "description": "Referral bonus program enrollment", "value": "$25"},
        ],
    },
}

def get_retention_offers(risk_tier: str, contract_type: str) -> dict:
    """Returns retention offers filtered by risk tier and contract type."""
    tier = risk_tier.lower()
    if tier not in OFFER_CATALOG:
        return {"error": f"Invalid risk tier: '{risk_tier}'. Must be high/medium/low."}
    tier_offers = OFFER_CATALOG[tier]
    offers = tier_offers.get(contract_type, tier_offers.get("Month-to-month", []))
    return {"risk_tier": tier, "contract_type": contract_type, "offers": offers}


# --- Tool 4: log_interaction ---
_interaction_log = []

def log_interaction(customer_id: str, agent_id: str, outcome: str,
                    offers_presented: list, offer_accepted: str = None,
                    notes: str = "") -> dict:
    """Records the outcome of a retention conversation."""
    record = {
        "interaction_id": f"INT-{len(_interaction_log)+1:05d}",
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "agent_id": agent_id,
        "outcome": outcome,  # "retained", "churned", "escalated", "callback_scheduled"
        "offers_presented": offers_presented,
        "offer_accepted": offer_accepted,
        "notes": notes,
    }
    _interaction_log.append(record)
    return {"status": "logged", "interaction_id": record["interaction_id"]}


# --- Tool 5: escalate_to_supervisor ---
def escalate_to_supervisor(customer_id: str, reason: str, context_summary: str,
                           urgency: str = "normal") -> dict:
    """Transfers the case to a human supervisor with context summary.
    Use for: legal threats, complex disputes, abusive language, or scenarios outside agent toolset."""
    return {
        "status": "escalated",
        "ticket_id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "customer_id": customer_id,
        "reason": reason,
        "urgency": urgency,  # "normal", "high", "critical"
        "context_summary": context_summary,
        "message": "Case transferred to supervisor queue. Rep should inform customer that a specialist will follow up within 2 hours."
    }
