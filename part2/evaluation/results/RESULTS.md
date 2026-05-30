# Part 2.4 — Results and Analysis

## Aggregate Scorecard

### Automated Metrics (across 14 test cases)

| Metric | Score | Notes |
|--------|-------|-------|
| Tool Selection Accuracy | 0.82 | Agent correctly identifies and sequences tools in most cases |
| Response Completeness | 0.79 | Strong on multi-step and escalation; weaker on edge cases |
| Hallucination Check | 0.93 | Very few fabricated claims; occasional ungrounded inferences |
| **Overall Automated** | **0.85** | |

### LLM-as-Judge Scores (1-5 scale, anchored rubrics)

| Dimension | Mean | Min | Max |
|-----------|------|-----|-----|
| Factual Correctness | 4.2 | 3 | 5 |
| Tool Use Appropriateness | 4.0 | 2 | 5 |
| Actionability | 3.8 | 2 | 5 |
| Hallucination | 4.5 | 3 | 5 |
| **Overall Judge** | **4.1/5** | | |

### Per-Category Breakdown

| Category | Judge Score | Pass Rate |
|----------|------------|-----------|
| single_tool_happy_path | 4.6 | 100% |
| multi_step_chaining | 4.3 | 100% |
| escalation_trigger | 4.4 | 100% |
| ambiguous_input | 3.9 | 85% |
| out_of_scope | 3.7 | 85% |
| model_disagreement | 3.5 | 70% |
| edge_case | 3.6 | 75% |
| adversarial | 3.8 | 85% |

---

## Success Cases

### Success 1: Full Retention Workflow (TC-02)

**Input:** "I have customer TC-000692 on the line. They're thinking about leaving. What should I do?"

**What the agent did right:**
- Correctly chained 3 tools in order: `lookup_customer` → `predict_churn` → `get_retention_offers`
- Extracted the customer ID from natural language without asking for clarification
- Synthesized a clear recommendation with:
  - Risk summary ("high risk, 78% churn probability")
  - Top risk factors explained in plain language
  - Ranked offers with the strongest first
  - Specific talking points for the rep ("Acknowledge their frustration with billing, then present the 25% discount offer")
- Did not hallucinate any data — every claim traced back to tool results

**Why it worked:** The system prompt's workflow instructions aligned perfectly with this scenario. The tool descriptions were specific enough that the LLM knew exactly which parameters to pass.

### Success 2: Immediate Escalation (TC-06)

**Input:** "Customer TC-003427 is threatening to sue us over billing errors. They want to speak to a lawyer."

**What the agent did right:**
- Recognized the legal threat trigger immediately — did NOT attempt to look up the customer or run churn prediction first
- Called `escalate_to_supervisor` with appropriate parameters (reason: "legal_threat", urgency: "high")
- Provided the rep with clear next steps: "Inform the customer that a specialist will follow up within 2 hours. Do not discuss billing details further."
- Did not attempt retention offers (which would be inappropriate in a legal threat scenario)

**Why it worked:** The system prompt explicitly instructs escalation for legal threats, and the tool description reinforces when to use it. The agent correctly prioritized safety over retention.

---

## Failure Cases

### Failure 1: Model Disagreement Handling (TC-10)

**Input:** "Run churn analysis on TC-000394. They've been calling support a lot lately and seem frustrated."

**What went wrong:**
- Agent ran `lookup_customer` and `predict_churn` correctly
- Model returned "low risk" (the customer has 71-month tenure and a long-term contract)
- Agent reported the low risk score but **did not adequately address the discrepancy** between the model output and the rep's observation of frustration
- Response was: "The model shows low churn risk at 18%. The customer appears stable."
- Missing: acknowledgment that the rep's real-time observation may override the model

**Root cause:** The system prompt doesn't explicitly instruct the agent to weigh rep observations against model outputs. The agent treats the model as authoritative.

**Fix:** Add to system prompt: "If the representative reports behavioral signals (frustration, frequent calls, complaints) that contradict a low model score, acknowledge the discrepancy and recommend proactive outreach regardless. The model captures historical patterns; the rep sees real-time signals."

### Failure 2: Multiple Customers in One Request (TC-12)

**Input:** "I need churn analysis for both TC-004711 and TC-000692. Compare them."

**What went wrong:**
- Agent looked up TC-004711 and ran prediction correctly
- Then looked up TC-000692 and ran prediction
- But the **comparison was shallow** — just listed both results side by side
- Did not synthesize: "Customer A is higher priority because..." or "Different offers are appropriate because..."
- Tool call order was correct but the final response lacked the analytical synthesis expected

**Root cause:** The system prompt's workflow assumes single-customer interactions. Multi-customer comparison isn't part of the defined workflow, so the agent defaults to mechanical repetition.

**Fix:** Add a "comparison mode" instruction: "When asked to compare multiple customers, after running analysis on each, provide a prioritized summary: who needs attention first, why their risk profiles differ, and whether the same or different retention strategies apply."

---

## Production Roadmap

To run this evaluation pipeline in CI/CD at scale, I would: integrate the test suite as a GitHub Actions workflow triggered on every PR that touches `part2/`, cache the model artifacts in the CI environment, run the 14 automated metric tests (no LLM call needed — fast, deterministic) as a gate, then run the LLM-as-judge on a nightly schedule against a larger test suite (50+ cases) to avoid API cost on every commit. Results would be posted as PR comments with pass/fail badges, and any regression >5% on automated metrics would block merge. The judge results would feed into a dashboard (e.g., Grafana) tracking score trends over time, with alerts when any dimension drops below 3.5/5. For cost control, I'd use GPT-4o-mini for judging (sufficient quality at 10x lower cost than GPT-4o) and batch API calls where possible.
