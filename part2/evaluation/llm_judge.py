"""LLM-as-Judge evaluator with anchored scoring rubrics."""
import json
from openai import OpenAI

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a customer retention AI agent. You will be given:
1. A test case (user input + expected behavior)
2. The agent's actual output (response + tool calls)

Score the agent's performance on 4 dimensions using the anchored rubrics below.
Return ONLY valid JSON with your scores and brief justifications.

---

## Dimension 1: Factual Correctness (1-5)

1 - WRONG: Response contains fabricated data, incorrect customer details, or wrong tool results. Claims things not supported by tool outputs.
2 - MOSTLY WRONG: Multiple factual errors. Some data is correct but key claims are unsupported.
3 - PARTIALLY CORRECT: Core facts are right but contains minor inaccuracies or unsupported inferences.
4 - MOSTLY CORRECT: All major facts align with tool results. At most one trivial imprecision.
5 - FULLY CORRECT: Every claim is directly grounded in tool results or reasonable inference from them. No fabrication.

## Dimension 2: Tool Use Appropriateness (1-5)

1 - WRONG TOOLS: Called completely irrelevant tools, or failed to call any tools when clearly needed.
2 - PARTIALLY WRONG: Called some correct tools but missed critical ones, or called unnecessary tools that add confusion.
3 - ACCEPTABLE: Called the right tools but in suboptimal order, or with minor parameter issues.
4 - GOOD: Correct tools in correct order with proper parameters. Minor optimization possible.
5 - OPTIMAL: Perfect tool selection, ordering, and parameterization. Exactly what an expert would do.

## Dimension 3: Actionability for Representative (1-5)

1 - USELESS: Raw data dump or vague platitudes. Rep cannot act on this response.
2 - BARELY USEFUL: Contains some relevant info but no clear recommendation or next steps.
3 - SOMEWHAT ACTIONABLE: Has a recommendation but lacks specifics (which offer, what to say, when to escalate).
4 - ACTIONABLE: Clear recommendation with specific offers and talking points. Rep knows what to do next.
5 - HIGHLY ACTIONABLE: Specific, prioritized recommendations with talking points, fallback options, and clear next steps. Rep can immediately act.

## Dimension 4: Hallucination (1-5)

1 - SEVERE HALLUCINATION: Invents customer data, fabricates offers not in catalog, or claims tool results that didn't happen.
2 - MODERATE HALLUCINATION: Makes specific claims (numbers, dates, IDs) not grounded in tool results.
3 - MINOR HALLUCINATION: Mostly grounded but makes one unsupported inference presented as fact.
4 - NEAR PERFECT: All claims grounded. At most one vague statement that could be interpreted as slight overreach.
5 - NO HALLUCINATION: Every statement is directly supported by tool results, user input, or clearly marked as a suggestion/recommendation.

---

Respond with this exact JSON structure:
{
  "factual_correctness": {"score": <1-5>, "justification": "<1 sentence>"},
  "tool_use_appropriateness": {"score": <1-5>, "justification": "<1 sentence>"},
  "actionability": {"score": <1-5>, "justification": "<1 sentence>"},
  "hallucination": {"score": <1-5>, "justification": "<1 sentence>"},
  "overall_assessment": "<1 sentence summary>"
}
"""


def judge_response(test_case: dict, agent_output: dict, client: OpenAI = None,
                   model: str = "gpt-4o-mini") -> dict:
    """
    Uses a separate LLM call to evaluate the agent's response against the test case.
    Returns structured rubric-based scores.
    """
    if client is None:
        client = OpenAI()

    # Build the evaluation prompt
    eval_input = {
        "test_case": {
            "id": test_case["id"],
            "category": test_case["category"],
            "user_input": test_case["user_input"],
            "expected_tool_calls": test_case.get("expected_tool_calls", []),
            "quality_criteria": test_case["quality_criteria"],
        },
        "agent_output": {
            "response": agent_output.get("response", ""),
            "tool_calls": [
                {"tool": tc["tool"], "arguments": tc["arguments"], "result": tc["result"]}
                for tc in agent_output.get("tool_calls", [])
            ],
        },
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(eval_input, default=str)},
        ],
        temperature=0.1,  # Low temp for consistent judging
        response_format={"type": "json_object"},
    )

    try:
        scores = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        scores = {
            "factual_correctness": {"score": 0, "justification": "Judge failed to return valid JSON"},
            "tool_use_appropriateness": {"score": 0, "justification": "Judge failed to return valid JSON"},
            "actionability": {"score": 0, "justification": "Judge failed to return valid JSON"},
            "hallucination": {"score": 0, "justification": "Judge failed to return valid JSON"},
            "overall_assessment": "Evaluation failed",
        }

    return scores


def run_judge_evaluation(test_cases: list, agent_outputs: list,
                         client: OpenAI = None, model: str = "gpt-4o-mini") -> dict:
    """
    Run LLM-as-judge across all test cases. Returns aggregate scorecard.
    """
    if client is None:
        client = OpenAI()

    results = []
    for tc, output in zip(test_cases, agent_outputs):
        scores = judge_response(tc, output, client=client, model=model)
        scores["test_id"] = tc["id"]
        scores["category"] = tc["category"]
        results.append(scores)

    # Aggregate
    dimensions = ["factual_correctness", "tool_use_appropriateness", "actionability", "hallucination"]
    aggregate = {}
    for dim in dimensions:
        dim_scores = [r[dim]["score"] for r in results if isinstance(r.get(dim, {}).get("score"), (int, float))]
        aggregate[dim] = {
            "mean": round(sum(dim_scores) / len(dim_scores), 2) if dim_scores else 0,
            "min": min(dim_scores) if dim_scores else 0,
            "max": max(dim_scores) if dim_scores else 0,
        }

    # Per-category breakdown
    categories = set(tc["category"] for tc in test_cases)
    per_category = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_scores = []
        for r in cat_results:
            avg = sum(r[d]["score"] for d in dimensions if isinstance(r.get(d, {}).get("score"), (int, float))) / len(dimensions)
            cat_scores.append(avg)
        per_category[cat] = round(sum(cat_scores) / len(cat_scores), 2) if cat_scores else 0

    return {
        "individual_results": results,
        "aggregate_scores": aggregate,
        "per_category": per_category,
        "overall_mean": round(
            sum(aggregate[d]["mean"] for d in dimensions) / len(dimensions), 2
        ),
    }


# --- Judge Reliability Discussion ---
RELIABILITY_NOTES = """
## How Do We Know the Judge Is Reliable?

1. **Low temperature (0.1)**: Reduces randomness in scoring, improving inter-run consistency.

2. **Anchored rubrics**: Each score level has a concrete description of what it looks like.
   This reduces positional bias and the "everything is a 4" problem common with unanchored scales.

3. **Structured JSON output**: Forces the judge to commit to specific scores rather than
   hedging with prose. The justification field creates accountability.

4. **Calibration approach**: To validate, we would:
   - Run the same test cases 5x and measure score variance (expect σ < 0.5 per dimension)
   - Have 2-3 humans score a subset and compute Cohen's kappa against the LLM judge
   - Check for positivity bias by including deliberately bad agent outputs

5. **Known limitations**:
   - LLM judges tend toward positivity bias (scoring 3-4 when 1-2 is warranted)
   - Prompt sensitivity: small rubric wording changes can shift scores by 0.5-1.0
   - Self-serving bias if the same model family judges its own outputs
   - Mitigation: use a different model for judging than for the agent (e.g., GPT-4o judges GPT-4o-mini)

6. **Production approach**: Run judge with 3 different temperature seeds, take median score.
   Flag cases where variance > 1.0 for human review.
"""
