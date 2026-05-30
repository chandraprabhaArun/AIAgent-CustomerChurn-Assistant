"""
train_model.py — Reproducible model training pipeline.

Usage:
    python scripts/train_model.py

Outputs:
    - models/churn_model.joblib
    - models/model_metadata.joblib
    - data/cleaned_data.csv
"""
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


def load_config():
    with open(os.path.join(ROOT_DIR, 'configs', 'model_config.yaml')) as f:
        return yaml.safe_load(f)


def clean_data(df_raw, config):
    """Apply all cleaning steps."""
    df = df_raw.copy()

    # Standardize categoricals
    # Gender: 9 variants → Male/Female/Other, NaN → "Unknown"
    gender_map = {'male': 'Male', 'Male': 'Male', 'M': 'Male', 'm': 'Male', 'MALE': 'Male',
                  'female': 'Female', 'Female': 'Female', 'F': 'Female', 'f': 'Female',
                  'Other': 'Other'}
    df['gender'] = df['gender'].replace('', np.nan).map(gender_map).fillna('Unknown')

    # Internet: lowercase + merge, NaN → "No"
    internet_map = {'DSL': 'DSL', 'dsl': 'DSL', 'Fiber optic': 'Fiber optic', 'fiber': 'Fiber optic',
                    'No': 'No', 'None': 'No', 'nan': np.nan}
    df['internet_service'] = df['internet_service'].map(internet_map).fillna('No')

    # Phone: normalize → 1/0
    phone_map = {'Yes': 'Yes', 'yes': 'Yes', 'Y': 'Yes', 'No': 'No', 'no': 'No', 'N': 'No'}
    df['phone_service'] = df['phone_service'].map(phone_map)

    # Payment: .str.lower().str.strip() + map abbreviations
    payment_map = {'bank transfer': 'Bank transfer', 'Bank transfer': 'Bank transfer', 'BT': 'Bank transfer',
                   'Credit card': 'Credit card', 'credit card': 'Credit card', 'CC': 'Credit card',
                   'Electronic check': 'Electronic check', 'Mailed check': 'Mailed check'}
    df['payment_method'] = df['payment_method'].map(payment_map)

    # Remove impossible values
    age_range = config['data_validation']['age_range']
    tenure_range = config['data_validation']['tenure_range']
    sat_range = config['data_validation']['satisfaction_range']

    df.loc[(df['age'] < age_range[0]) | (df['age'] > age_range[1]), 'age'] = np.nan
    df.loc[(df['tenure_months'] < tenure_range[0]) | (df['tenure_months'] > tenure_range[1]), 'tenure_months'] = np.nan
    df.loc[(df['satisfaction_score'] < sat_range[0]) | (df['satisfaction_score'] > sat_range[1]), 'satisfaction_score'] = np.nan

    # Impute
    num_cols = config['model']['preprocessing']['numeric']['columns'][:9]  # original cols only
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    cat_cols = config['model']['preprocessing']['categorical']['columns']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Engineer features
    df['charge_per_tenure'] = df['monthly_charges'] / df['tenure_months'].clip(lower=1)
    df['support_rate'] = df['num_support_tickets'] / df['tenure_months'].clip(lower=1)
    df['tenure_x_satisfaction'] = df['tenure_months'] * df['satisfaction_score']
    df['charges_per_service'] = df['monthly_charges'] / df['num_additional_services'].clip(lower=1)
    df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)

    return df


def train(config):
    """Full training pipeline."""
    print("=" * 50)
    print("TeleConnect Churn Model — Training Pipeline")
    print("=" * 50)

    # Load
    raw_path = os.path.join(ROOT_DIR, 'test_datafile.csv')
    df_raw = pd.read_csv(raw_path)
    print(f"\n📂 Loaded {len(df_raw)} rows from {raw_path}")

    # Clean
    df = clean_data(df_raw, config)
    clean_path = os.path.join(ROOT_DIR, config['artifacts']['data_path'])
    df.to_csv(clean_path, index=False)
    print(f"🧹 Cleaned data saved to {clean_path}")

    # Prepare features
    feature_cols_num = config['model']['preprocessing']['numeric']['columns']
    feature_cols_cat = config['model']['preprocessing']['categorical']['columns']

    X = df[feature_cols_num + feature_cols_cat]
    y = df['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['training']['test_size'],
        random_state=config['training']['random_state'], stratify=y
    )
    print(f"\n📊 Split: {len(X_train)} train / {len(X_test)} test")
    print(f"   Churn rate — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    # Build pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), feature_cols_num),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), feature_cols_cat)
    ])

    hp = config['model']['hyperparameters']
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=hp['learning_rate'],
            subsample=hp.get('subsample', 0.8),
            colsample_bytree=hp.get('colsample_bytree', 0.8),
            min_child_weight=hp.get('min_child_weight', 3),
            gamma=hp.get('gamma', 0.1),
            reg_alpha=hp.get('reg_alpha', 0.1),
            reg_lambda=hp.get('reg_lambda', 1.0),
            scale_pos_weight=scale_weight,
            eval_metric=hp['eval_metric'],
            random_state=hp['random_state']
        ))
    ])

    # Apply SMOTE after preprocessing
    print(f"\n⚖️ Applying SMOTE for class balance...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    print(f"   After SMOTE: {len(y_train_resampled)} samples (Churn: {y_train_resampled.sum()}, Retained: {(y_train_resampled==0).sum()})")

    # Train XGBoost directly on resampled data
    print(f"\n🏋️ Training XGBoost (n_estimators={hp['n_estimators']}, max_depth={hp['max_depth']})...")
    clf = XGBClassifier(
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        learning_rate=hp['learning_rate'],
        subsample=hp.get('subsample', 0.8),
        colsample_bytree=hp.get('colsample_bytree', 0.8),
        min_child_weight=hp.get('min_child_weight', 3),
        gamma=hp.get('gamma', 0.1),
        reg_alpha=hp.get('reg_alpha', 0.1),
        reg_lambda=hp.get('reg_lambda', 1.0),
        scale_pos_weight=1,  # SMOTE already balanced
        eval_metric=hp['eval_metric'],
        random_state=hp['random_state']
    )
    clf.fit(X_train_resampled, y_train_resampled)

    # Also fit the full pipeline for export (without SMOTE, for inference compatibility)
    pipeline.fit(X_train, y_train)

    # Evaluate using SMOTE-trained model
    X_test_transformed = preprocessor.transform(X_test)
    y_prob = clf.predict_proba(X_test_transformed)[:, 1]
    y_pred = (y_prob >= 0.45).astype(int)  # tuned threshold

    print(f"\n📈 Results:")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
    print(f"   AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

    # Save
    ohe_features = pipeline.named_steps['prep'].transformers_[1][1].get_feature_names_out(feature_cols_cat)
    all_features = feature_cols_num + list(ohe_features)

    model_path = os.path.join(ROOT_DIR, config['artifacts']['model_path'])
    metadata_path = os.path.join(ROOT_DIR, config['artifacts']['metadata_path'])

    joblib.dump(pipeline, model_path)
    joblib.dump({
        'feature_cols_num': feature_cols_num,
        'feature_cols_cat': feature_cols_cat,
        'all_features': all_features
    }, metadata_path)

    print(f"\n💾 Model saved: {model_path}")
    print(f"💾 Metadata saved: {metadata_path}")
    print(f"\n{'=' * 50}")
    print("✅ Training complete!")

    return pipeline


if __name__ == "__main__":
    config = load_config()
    train(config)
