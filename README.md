# 🔮 TeleConnect — Churn Prediction & AI Retention Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-412991.svg)](https://openai.com)

> End-to-end ML pipeline for customer churn prediction paired with an AI-powered retention agent that helps representatives handle at-risk customers in real time.

---

## 🎯 Live Demo

🔗 **[Launch Retention Agent →](https://teleconnect-retention.streamlit.app)**

---

## 📐 Architecture

```
TeleConnect/
│
├── 📊 part1/                          # Churn Prediction Model
│   ├── churn_model.ipynb              # Full ML pipeline (EDA → Model → Export)
│   └── predict_churn.py               # Standalone inference function
│
├── 🤖 part2/                          # AI Retention Agent
│   ├── agent.py                       # LLM orchestration (OpenAI tool-calling)
│   ├── tools/                         # 5 modular agent tools
│   │   ├── __init__.py
│   │   └── tools.py                   # lookup, predict, offers, log, escalate
│   ├── evaluation/                    # Evaluation framework
│   │   ├── test_cases.json            # 14 structured test cases
│   │   ├── metrics.py                 # 3 automated metrics
│   │   ├── llm_judge.py              # LLM-as-Judge (anchored rubrics)
│   │   ├── run_eval.py               # Evaluation runner
│   │   └── results/                   # Scorecard & analysis
│   │       └── RESULTS.md
│   └── app/                           # Streamlit deployment
│       └── main.py                    # UI with visible tool traces
│
├── 📦 models/                         # Exported model artifacts
│   ├── churn_model.joblib             # XGBoost pipeline (preprocessor + model)
│   └── model_metadata.joblib          # Feature lists & config
│
├── 📁 data/                           # Processed datasets
│   └── cleaned_data.csv               # Post-cleaning dataset
│
├── 📝 docs/                           # Documentation
│   ├── DESIGN_DECISIONS.md            # Architecture & trade-off rationale
│   └── EVALUATION_REPORT.md           # Full evaluation analysis
│
├── ⚙️  configs/                        # Configuration files
│   └── model_config.yaml              # Model hyperparameters & thresholds
│
├── 🔧 scripts/                        # Utility scripts
│   ├── train_model.py                 # Reproducible model training
│   └── run_evaluation.py              # Run full eval pipeline
│
├── .streamlit/                        # Streamlit Cloud config
│   └── config.toml
├── .gitignore
├── requirements.txt                   # Python dependencies
├── test_datafile.csv                  # Raw dataset (5,050 customers)
└── README.md                          # ← You are here
```

---

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/<username>/teleconnect-churn.git
cd teleconnect-churn
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train model (or use pre-trained artifacts in models/)
python scripts/train_model.py

# 3. Run agent locally
export OPENAI_API_KEY="sk.."
streamlit run part2/app/main.py

# 4. Run evaluation suite
python scripts/run_evaluation.py
```

---

## 📊 Part 1 — Churn Model

### Pipeline Summary

| Stage | What | Key Output |
|-------|------|------------|
| **1.1 Data Cleaning** | 12 issues fixed (encodings, sentinels, impossible values) | Before/after summary table |
| **1.2 EDA** | Churn rate 36%, top 5 features, 3 visualizations | 2 engineered features |
| **1.3 Modeling** | XGBoost vs Logistic Regression | XGBoost selected (AUC 0.68) |
| **1.4 Visualization** | Confusion matrix, ROC curve, feature importance | Publication-ready plots |
| **1.5 Export** | `predict_churn()` function + joblib artifacts | Ready for agent integration |

### Model Performance

| Metric | XGBoost | Logistic Regression |
|--------|---------|-------------------|
| AUC-ROC | **0.68** | 0.63 |
| Recall (Churn) | **0.59** | 0.54 |
| Precision (Churn) | 0.51 | 0.47 |
| F1 (Churn) | **0.55** | 0.50 |

### `predict_churn` API

```python
from part1.predict_churn import predict_churn

result = predict_churn({
    "age": 32, "tenure_months": 3, "monthly_charges": 89.50,
    "contract_type": "Month-to-month", "satisfaction_score": 4.2, ...
})
# → {"churn_probability": 0.81, "risk_tier": "high", "top_risk_factors": [...]}
```

---

## 🤖 Part 2 — Retention Agent

### Tool Architecture

| Tool | Type | Purpose |
|------|------|---------|
| `lookup_customer` | Data retrieval | Fetch customer profile by ID |
| `predict_churn` | ML inference | Run trained model, return risk assessment |
| `get_retention_offers` | Business logic | Filter offers by risk tier + contract |
| `log_interaction` | Audit trail | Record conversation outcomes |
| `escalate_to_supervisor` | Safety valve | Legal threats, abuse, complex disputes |

### Agent Flow

```
User Message → LLM (GPT-4o-mini)
                    ↓
            Tool Selection (auto)
                    ↓
    ┌─────────────────────────────────┐
    │  lookup_customer(id)            │
    │       ↓                         │
    │  predict_churn(features)        │
    │       ↓                         │
    │  get_retention_offers(tier)     │
    │       ↓                         │
    │  Synthesize Recommendation      │
    └─────────────────────────────────┘
                    ↓
        Final Response to Rep
```

### Evaluation Results

| Dimension | Score | Scale |
|-----------|-------|-------|
| Tool Selection Accuracy | 0.82 | 0-1 |
| Response Completeness | 0.79 | 0-1 |
| Hallucination Check | 0.93 | 0-1 |
| **LLM Judge Overall** | **4.1** | 1-5 |

---

## 🧪 Evaluation Framework

- **14 test cases** across 8 categories (happy path, multi-step, ambiguous, escalation, adversarial...)
- **3 automated metrics** (tool accuracy, completeness, hallucination detection)
- **LLM-as-Judge** with anchored rubrics (not 1-10 scale — descriptive anchors per level)
- **4 scoring dimensions**: factual correctness, tool use, actionability, hallucination

---

## 🏗️ Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary model | XGBoost | Best on tabular data with mixed types; captures non-linear interactions |
| Key metric | Recall | Missing a churner costs ~$1000 LTV; false alarm costs one phone call |
| Agent framework | OpenAI function calling | Native tool-calling, no heavy framework dependency |
| Evaluation | LLM-as-Judge + automated | Combines deterministic checks with nuanced quality assessment |
| Deployment | Streamlit | Fast iteration, visible tool traces, free cloud hosting |

---

## ⚠️ Limitations & Future Work

- **Model**: No hyperparameter tuning (Optuna), no cross-validation, no SHAP explanations
- **Agent**: Single-turn only; production would need conversation memory + session management
- **Evaluation**: Judge reliability not validated against human labels (would need calibration study)
- **Data**: Synthetic-looking dataset; real deployment would need temporal train/test split

---

## 👤 Author

Built as a demonstration of end-to-end ML engineering: from raw data → production-ready AI agent.
