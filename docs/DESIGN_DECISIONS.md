# Design Decisions

## Model Selection

### Why XGBoost over Logistic Regression?

| Factor | XGBoost | Logistic Regression |
|--------|---------|-------------------|
| Non-linear interactions | ✅ Captures tenure × contract type | ❌ Requires manual feature engineering |
| Mixed feature types | ✅ Handles natively after encoding | ⚠️ Sensitive to scaling |
| Interpretability | ⚠️ Feature importance (global) | ✅ Coefficients (per-feature) |
| Overfitting risk | ⚠️ Higher (mitigated by early stopping) | ✅ Lower with regularization |

**Decision:** XGBoost as primary model. LR as baseline for comparison. In production, I'd add SHAP for per-prediction explanations.

---

## Metric Selection: Why Recall?

The business context is asymmetric:
- **False Negative** (miss a churner): Customer leaves → ~$1,000+ lost lifetime value
- **False Positive** (flag a loyal customer): One unnecessary retention call → ~$5 cost

This 200:1 cost ratio means we should optimize for **recall** (catch as many churners as possible), accepting lower precision.

**Why not accuracy?** A model predicting "no churn" for everyone gets 64% accuracy. Useless.

**Why not F1?** F1 balances precision and recall equally. Our costs aren't equal.

**Why not AUC-ROC?** Useful for model comparison but doesn't reflect operational threshold decisions.

---

## Agent Architecture: Why OpenAI Function Calling?

Considered alternatives:
- **LangChain agents**: More abstraction, but adds dependency complexity and debugging opacity
- **Raw prompt engineering**: Fragile, no structured tool outputs
- **OpenAI function calling**: Native, reliable, structured JSON in/out, easy to add tools

**Extensibility test:** Adding a 6th tool requires:
1. Write the function
2. Add tool definition to `TOOLS` list
3. Add to `TOOL_FUNCTIONS` dispatch map

No orchestration rewrite needed. The LLM decides when to call it based on the description.

---

## Evaluation: Why LLM-as-Judge with Anchored Rubrics?

**Problem with 1-10 scales:** Evaluators (human or LLM) cluster around 6-8. No calibration.

**Problem with binary pass/fail:** Loses nuance. A "mostly correct with one minor issue" response shouldn't score the same as "completely wrong."

**Solution:** 5-point scale with descriptive anchors. Each level describes what that score *looks like*:
- Level 1: "Response contains fabricated data..."
- Level 3: "Core facts are right but contains minor inaccuracies..."
- Level 5: "Every claim is directly grounded in tool results..."

This forces the judge to match the response against concrete descriptions rather than vibes.

---

## Deployment: Why Streamlit?

| Option | Pros | Cons |
|--------|------|------|
| Streamlit | Free hosting, fast dev, Python-native | Limited customization |
| Gradio | Good for ML demos | Less control over layout |
| FastAPI + React | Production-grade | Overkill for demo |
| Hugging Face Spaces | Free, good for models | Less flexible for agents |

**Decision:** Streamlit. Free cloud hosting, visible tool traces are easy to implement with `st.expander` + `st.json`, and the chat interface is built-in.

---

## Data Cleaning: Why Median Imputation?

- **Mean** is sensitive to outliers (which we have: age=999, tenure=500)
- **Median** is robust to skew and outliers
- **KNN/MICE** would be better but adds complexity for <5% missing data
- **Dropping rows** would lose ~10% of data unnecessarily

For categoricals: **mode imputation** (most common value). With <5% missing, this introduces minimal bias.

---

## What I'd Do Differently With More Time

1. **Hyperparameter tuning** — Optuna with stratified k-fold CV
2. **SHAP values** — per-prediction explanations instead of global feature importance
3. **Temporal validation** — train on months 1-4, test on months 5-6
4. **Conversation memory** — multi-turn agent with session state
5. **Human calibration** — have 3 people score 20 cases, compute Cohen's kappa vs judge
6. **A/B testing framework** — compare agent recommendations vs human-only retention
