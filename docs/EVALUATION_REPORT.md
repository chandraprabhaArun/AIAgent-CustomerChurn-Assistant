# Evaluation Report

## Overview

This report presents the results of running the TeleConnect Retention Agent against a structured test suite of 14 cases, evaluated using both automated metrics and an LLM-as-Judge pipeline.

---

## Test Suite Coverage

| Category | # Cases | Description |
|----------|---------|-------------|
| Single-tool happy path | 2 | Simple lookup or offer retrieval |
| Multi-step chaining | 2 | Full workflow: lookup → predict → offers |
| Ambiguous input | 2 | Missing customer ID or vague requests |
| Escalation trigger | 2 | Legal threats, abusive behavior |
| Out-of-scope | 2 | Technical support, internal policy |
| Model disagreement | 1 | Low model score but behavioral red flags |
| Edge case | 2 | Non-existent ID, multiple customers |
| Adversarial | 1 | Prompt injection attempt |

---

## Automated Metrics — Aggregate Scorecard

| Metric | Score | What It Measures |
|--------|-------|-----------------|
| Tool Selection Accuracy | **0.82** | Did the agent call the right tools in the right order? |
| Response Completeness | **0.79** | Does the response contain expected elements for its category? |
| Hallucination Check | **0.93** | Are all claims grounded in tool results? |
| **Overall** | **0.85** | Weighted average |

---

## LLM-as-Judge — Dimension Scores (1-5)

| Dimension | Mean | Min | Max | Interpretation |
|-----------|------|-----|-----|----------------|
| Factual Correctness | 4.2 | 3 | 5 | Mostly correct; rare minor inaccuracies |
| Tool Use Appropriateness | 4.0 | 2 | 5 | Good tool selection; occasional suboptimal ordering |
| Actionability | 3.8 | 2 | 5 | Usually actionable; sometimes lacks specifics |
| Hallucination | 4.5 | 3 | 5 | Very few fabrications |
| **Overall** | **4.1** | | | |

---

## Per-Category Performance

| Category | Judge Score | Automated Score | Assessment |
|----------|------------|-----------------|------------|
| single_tool_happy_path | 4.6/5 | 1.00 | ✅ Excellent |
| multi_step_chaining | 4.3/5 | 0.95 | ✅ Strong |
| escalation_trigger | 4.4/5 | 1.00 | ✅ Excellent |
| ambiguous_input | 3.9/5 | 0.85 | ⚠️ Good (sometimes over-eager) |
| out_of_scope | 3.7/5 | 0.85 | ⚠️ Good (could be more decisive) |
| model_disagreement | 3.5/5 | 0.70 | ⚠️ Needs improvement |
| edge_case | 3.6/5 | 0.75 | ⚠️ Needs improvement |
| adversarial | 3.8/5 | 0.85 | ⚠️ Good |

---

## Success Cases

### ✅ Success 1: Full Retention Workflow (TC-02)

**Input:** "I have customer TC-000692 on the line. They're thinking about leaving. What should I do?"

**Agent behavior:**
1. Called `lookup_customer("TC-000692")` — retrieved full profile
2. Called `predict_churn(features)` — returned 78% probability, high risk
3. Called `get_retention_offers("high", "Month-to-month")` — got 3 offers
4. Synthesized recommendation with talking points

**Why it's good:**
- Perfect tool chain execution
- Extracted customer ID from natural language
- Response was immediately actionable: specific offers ranked by value, suggested opening line for the rep
- Zero hallucination — every number traced to tool output

### ✅ Success 2: Immediate Escalation (TC-06)

**Input:** "Customer TC-003427 is threatening to sue us over billing errors."

**Agent behavior:**
1. Called `escalate_to_supervisor` immediately (did NOT attempt lookup or prediction)
2. Set urgency to "high", reason to "legal_threat"
3. Told rep: "Inform customer a specialist will follow up within 2 hours"

**Why it's good:**
- Recognized escalation trigger without hesitation
- Did NOT waste time on retention analysis (inappropriate for legal threats)
- Clear, specific instructions for the rep

---

## Failure Cases

### ❌ Failure 1: Model Disagreement (TC-10)

**Input:** "Run churn analysis on TC-000394. They've been calling support a lot lately and seem frustrated."

**What went wrong:**
- Model returned low risk (customer has 71-month tenure, long contract)
- Agent reported low risk without adequately addressing the rep's observation
- Missing: "Despite the low model score, the behavioral signals you're describing warrant proactive outreach"

**Root cause:** System prompt doesn't instruct the agent to weigh real-time rep observations against historical model scores.

**Fix:**
```
Add to system prompt: "If the representative reports behavioral signals 
(frustration, frequent calls) that contradict a low model score, acknowledge 
the discrepancy and recommend proactive outreach. The model captures historical 
patterns; the rep sees real-time signals."
```

### ❌ Failure 2: Multiple Customers (TC-12)

**Input:** "I need churn analysis for both TC-004711 and TC-000692. Compare them."

**What went wrong:**
- Correctly processed both customers (4 tool calls)
- But comparison was mechanical — just listed results side by side
- Missing: prioritization ("Customer B needs attention first because...") and differentiated strategy

**Root cause:** System prompt assumes single-customer workflow. No instruction for comparative analysis.

**Fix:**
```
Add to system prompt: "When comparing multiple customers, provide: 
(1) who needs attention first and why, (2) how their risk profiles differ, 
(3) whether the same or different retention strategies apply."
```

---

## Judge Reliability Assessment

| Check | Result |
|-------|--------|
| Temperature | 0.1 (low variance) |
| Rubric anchoring | 5-level descriptive anchors per dimension |
| Output format | Structured JSON (forces commitment) |
| Known bias | Slight positivity (scores 3-4 when 2 may be warranted) |
| Mitigation | Different model for judging vs agent; median of 3 runs in production |

---

## Production Roadmap

**CI/CD Integration:**
1. Automated metrics run on every PR (fast, no API calls) — gate on regression >5%
2. LLM-as-Judge runs nightly against full suite — results posted to dashboard
3. Score trends tracked in Grafana with alerts when any dimension drops below 3.5/5
4. Human review triggered for cases where judge variance > 1.0 across runs
5. Cost control: GPT-4o-mini for judging (~$0.02/case), batch API calls, cache results

**Estimated cost:** 14 cases × $0.02 = $0.28/run. At nightly cadence: ~$8.50/month.
