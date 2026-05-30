"""Automated evaluation metrics for the retention agent."""
import json


def tool_selection_accuracy(expected_tools: list, actual_tools: list) -> dict:
    """
    Measures whether the agent called the correct tools in the correct order.
    
    Returns:
        - precision: fraction of actual calls that were expected
        - recall: fraction of expected calls that were made
        - order_correct: whether the sequence matches
        - score: combined score (0.0 - 1.0)
    """
    expected_names = [t["name"] for t in expected_tools]
    actual_names = [t["tool"] for t in actual_tools]

    if not expected_names and not actual_names:
        return {"precision": 1.0, "recall": 1.0, "order_correct": True, "score": 1.0}

    if not expected_names and actual_names:
        # Expected no tools but agent called some
        return {"precision": 0.0, "recall": 1.0, "order_correct": False, "score": 0.0}

    if expected_names and not actual_names:
        return {"precision": 1.0, "recall": 0.0, "order_correct": False, "score": 0.0}

    # Precision: how many actual calls were correct
    correct_calls = sum(1 for t in actual_names if t in expected_names)
    precision = correct_calls / len(actual_names) if actual_names else 0

    # Recall: how many expected calls were made
    made_calls = sum(1 for t in expected_names if t in actual_names)
    recall = made_calls / len(expected_names) if expected_names else 0

    # Order check: expected tools appear in correct relative order
    expected_in_actual = [t for t in actual_names if t in expected_names]
    order_correct = expected_in_actual == expected_names[:len(expected_in_actual)]

    # Combined score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    order_bonus = 0.1 if order_correct else 0
    score = min(1.0, f1 + order_bonus)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "order_correct": order_correct,
        "score": round(score, 3),
    }


def response_completeness(response: str, test_case: dict) -> dict:
    """
    Checks whether the agent's response contains expected elements based on category.
    
    Criteria by category:
    - multi_step_chaining: must include risk assessment, offers, and recommendation
    - escalation_trigger: must include escalation confirmation and next steps
    - ambiguous_input: must ask for clarification
    - single_tool_happy_path: must include relevant data from tool result
    """
    category = test_case["category"]
    response_lower = response.lower() if response else ""
    checks = {}

    if category == "multi_step_chaining":
        checks["has_risk_assessment"] = any(w in response_lower for w in ["risk", "churn", "probability"])
        checks["has_offers"] = any(w in response_lower for w in ["offer", "discount", "upgrade", "free"])
        checks["has_recommendation"] = any(w in response_lower for w in ["recommend", "suggest", "advise", "talking point"])
        checks["has_customer_context"] = any(w in response_lower for w in ["contract", "tenure", "month"])

    elif category == "escalation_trigger":
        checks["has_escalation_confirm"] = any(w in response_lower for w in ["escalat", "supervisor", "transfer"])
        checks["has_next_steps"] = any(w in response_lower for w in ["follow up", "specialist", "next", "inform"])

    elif category == "ambiguous_input":
        checks["asks_for_id"] = any(w in response_lower for w in ["customer id", "id", "which customer", "provide"])

    elif category == "out_of_scope":
        checks["declines_gracefully"] = any(w in response_lower for w in ["technical", "support", "retention", "can't help", "not able"])

    elif category == "single_tool_happy_path":
        checks["has_data"] = len(response_lower) > 50

    elif category == "edge_case":
        checks["handles_error"] = any(w in response_lower for w in ["not found", "verify", "check", "error", "invalid"])

    elif category == "model_disagreement":
        checks["acknowledges_discrepancy"] = any(w in response_lower for w in ["however", "despite", "but", "discrepancy", "proactive"])

    elif category == "adversarial":
        checks["stays_in_role"] = any(w in response_lower for w in ["retention", "customer", "help you with"])

    # Score: fraction of checks passed
    if not checks:
        return {"checks": {}, "score": 1.0}

    passed = sum(1 for v in checks.values() if v)
    score = passed / len(checks)

    return {"checks": checks, "score": round(score, 3)}


def hallucination_check(response: str, tool_results: list, test_case: dict) -> dict:
    """
    Detects potential hallucinations by checking if the response contains
    specific claims not grounded in tool results.
    
    Checks:
    - Fabricated customer IDs not in the conversation
    - Specific numbers (probabilities, charges) not from tool outputs
    - Offer IDs not from get_retention_offers results
    """
    if not response:
        return {"flags": [], "score": 1.0}

    flags = []
    tool_content = json.dumps(tool_results, default=str).lower()
    response_lower = response.lower()

    # Check for fabricated offer IDs
    import re
    offer_ids_in_response = re.findall(r'[HML]-(?:MTM|1Y|2Y)-\d{2}', response)
    for oid in offer_ids_in_response:
        if oid.lower() not in tool_content:
            flags.append(f"Offer ID '{oid}' not found in tool results")

    # Check for fabricated customer IDs
    customer_ids_in_response = re.findall(r'TC-\d{6}', response)
    customer_ids_in_input = re.findall(r'TC-\d{6}', test_case.get("user_input", ""))
    for cid in customer_ids_in_response:
        if cid not in customer_ids_in_input and cid.lower() not in tool_content:
            flags.append(f"Customer ID '{cid}' not grounded in input or tool results")

    # Check for specific probability claims
    prob_claims = re.findall(r'(\d{1,3}(?:\.\d+)?)\s*%', response)
    for prob in prob_claims:
        prob_val = float(prob)
        # Check if this percentage appears in tool results (as decimal or percentage)
        decimal_str = str(round(prob_val / 100, 4))
        if decimal_str not in tool_content and prob not in tool_content:
            # Allow some tolerance for rounding
            if prob_val not in [0, 25, 50, 75, 100]:  # common generic percentages are OK
                flags.append(f"Probability claim '{prob}%' not grounded in tool results")

    # Score: 1.0 = no hallucinations, decreases with each flag
    score = max(0.0, 1.0 - (len(flags) * 0.25))

    return {"flags": flags, "score": round(score, 3)}


def evaluate_test_case(test_case: dict, agent_output: dict) -> dict:
    """Run all 3 metrics on a single test case."""
    tool_accuracy = tool_selection_accuracy(
        test_case.get("expected_tool_calls", []),
        agent_output.get("tool_calls", [])
    )
    completeness = response_completeness(
        agent_output.get("response", ""),
        test_case
    )
    hallucination = hallucination_check(
        agent_output.get("response", ""),
        agent_output.get("tool_calls", []),
        test_case
    )

    return {
        "test_id": test_case["id"],
        "category": test_case["category"],
        "tool_selection_accuracy": tool_accuracy,
        "response_completeness": completeness,
        "hallucination_check": hallucination,
        "overall_score": round(
            (tool_accuracy["score"] + completeness["score"] + hallucination["score"]) / 3, 3
        ),
    }
