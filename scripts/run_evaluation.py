"""
run_evaluation.py — Run the full evaluation pipeline.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/run_evaluation.py

Outputs:
    - Console scorecard
    - part2/evaluation/results/scorecard.json
"""
import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from openai import OpenAI
from part2.agent import run_agent
from part2.evaluation.metrics import evaluate_test_case
from part2.evaluation.llm_judge import run_judge_evaluation


def load_test_cases():
    path = os.path.join(ROOT_DIR, 'part2', 'evaluation', 'test_cases.json')
    with open(path) as f:
        return json.load(f)


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    test_cases = load_test_cases()

    print("=" * 60)
    print("  TeleConnect Retention Agent — Evaluation Pipeline")
    print("=" * 60)
    print(f"\n  Running {len(test_cases)} test cases...\n")

    agent_outputs = []
    metric_results = []

    for tc in test_cases:
        print(f"  [{tc['id']}] {tc['category']}")
        print(f"       Input: \"{tc['user_input'][:60]}...\"")

        try:
            output = run_agent(tc["user_input"], client=client, model="gpt-4o-mini")
        except Exception as e:
            output = {"response": f"ERROR: {e}", "tool_calls": [], "conversation": []}

        agent_outputs.append(output)

        metrics = evaluate_test_case(tc, output)
        metric_results.append(metrics)

        tools_called = [t['tool'] for t in output.get('tool_calls', [])]
        print(f"       Tools: {tools_called or 'none'}")
        print(f"       Score: {metrics['overall_score']:.2f}\n")

    # Automated metrics aggregate
    print("\n" + "=" * 60)
    print("  AUTOMATED METRICS")
    print("=" * 60)
    agg = {
        'tool_selection_accuracy': sum(m['tool_selection_accuracy']['score'] for m in metric_results) / len(metric_results),
        'response_completeness': sum(m['response_completeness']['score'] for m in metric_results) / len(metric_results),
        'hallucination_check': sum(m['hallucination_check']['score'] for m in metric_results) / len(metric_results),
    }
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}")
    print(f"  overall: {sum(agg.values()) / len(agg):.3f}")

    # LLM Judge
    print(f"\n  Running LLM-as-Judge...")
    judge_results = run_judge_evaluation(test_cases, agent_outputs, client=client)

    print("\n" + "=" * 60)
    print("  LLM-AS-JUDGE SCORES")
    print("=" * 60)
    for dim, scores in judge_results['aggregate_scores'].items():
        print(f"  {dim}: {scores['mean']:.2f}/5 (min={scores['min']}, max={scores['max']})")
    print(f"\n  Overall: {judge_results['overall_mean']:.2f}/5")

    print("\n" + "=" * 60)
    print("  PER-CATEGORY BREAKDOWN")
    print("=" * 60)
    for cat, score in judge_results['per_category'].items():
        print(f"  {cat}: {score:.2f}/5")

    # Save results
    results_dir = os.path.join(ROOT_DIR, 'part2', 'evaluation', 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'scorecard.json')

    scorecard = {
        'automated_metrics': {'aggregate': agg, 'per_case': metric_results},
        'llm_judge': judge_results,
    }
    with open(output_path, 'w') as f:
        json.dump(scorecard, f, indent=2, default=str)

    print(f"\n  💾 Full results saved to {output_path}")
    print(f"\n{'=' * 60}")
    print("  ✅ Evaluation complete!")


if __name__ == "__main__":
    main()
