"""Evaluation runner — executes agent against test suite and produces scorecard."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import OpenAI
from part2.agent import run_agent
from part2.evaluation.metrics import evaluate_test_case
from part2.evaluation.llm_judge import run_judge_evaluation


def load_test_cases(path: str = None) -> list:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(path) as f:
        return json.load(f)


def run_evaluation(client: OpenAI = None, model: str = "gpt-4o-mini",
                   judge_model: str = "gpt-4o-mini") -> dict:
    """Run full evaluation pipeline: agent execution → automated metrics → LLM judge."""
    if client is None:
        client = OpenAI()

    test_cases = load_test_cases()
    agent_outputs = []
    metric_results = []

    print(f"Running {len(test_cases)} test cases...\n")

    for tc in test_cases:
        print(f"  [{tc['id']}] {tc['category']}: {tc['description']}")
        try:
            output = run_agent(tc["user_input"], client=client, model=model)
        except Exception as e:
            output = {"response": f"ERROR: {e}", "tool_calls": [], "conversation": []}

        agent_outputs.append(output)

        # Automated metrics
        metrics = evaluate_test_case(tc, output)
        metric_results.append(metrics)
        print(f"         Tools: {metrics['tool_selection_accuracy']['score']:.2f} | "
              f"Complete: {metrics['response_completeness']['score']:.2f} | "
              f"Halluc: {metrics['hallucination_check']['score']:.2f} | "
              f"Overall: {metrics['overall_score']:.2f}")

    # LLM-as-Judge
    print("\nRunning LLM-as-Judge evaluation...")
    judge_results = run_judge_evaluation(test_cases, agent_outputs, client=client, model=judge_model)

    # Build scorecard
    scorecard = {
        "automated_metrics": {
            "per_case": metric_results,
            "aggregate": {
                "tool_selection_accuracy": round(
                    sum(m["tool_selection_accuracy"]["score"] for m in metric_results) / len(metric_results), 3
                ),
                "response_completeness": round(
                    sum(m["response_completeness"]["score"] for m in metric_results) / len(metric_results), 3
                ),
                "hallucination_check": round(
                    sum(m["hallucination_check"]["score"] for m in metric_results) / len(metric_results), 3
                ),
                "overall": round(
                    sum(m["overall_score"] for m in metric_results) / len(metric_results), 3
                ),
            },
        },
        "llm_judge": judge_results,
        "summary": {
            "total_cases": len(test_cases),
            "automated_overall": round(
                sum(m["overall_score"] for m in metric_results) / len(metric_results), 3
            ),
            "judge_overall": judge_results["overall_mean"],
            "per_category": judge_results["per_category"],
        },
    }

    return scorecard


if __name__ == "__main__":
    scorecard = run_evaluation()
    print("\n" + "=" * 60)
    print("EVALUATION SCORECARD")
    print("=" * 60)
    print(f"\nAutomated Metrics (avg):")
    for k, v in scorecard["automated_metrics"]["aggregate"].items():
        print(f"  {k}: {v:.3f}")
    print(f"\nLLM Judge (avg across dimensions):")
    for k, v in scorecard["llm_judge"]["aggregate_scores"].items():
        print(f"  {k}: {v['mean']:.2f}/5")
    print(f"\nPer-Category Scores:")
    for cat, score in scorecard["summary"]["per_category"].items():
        print(f"  {cat}: {score:.2f}/5")
    print(f"\n{'=' * 60}")
    print(f"OVERALL: Automated={scorecard['summary']['automated_overall']:.3f} | Judge={scorecard['summary']['judge_overall']:.2f}/5")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(scorecard, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")
