"""
Show the exact prompts the Foundry SDK sent to the judge LLM
and the exact responses it got back, for any test case.

Usage:
  python scripts/show_judge_prompts.py              # Show TC1, all evaluators
  python scripts/show_judge_prompts.py 3             # Show TC3, all evaluators
  python scripts/show_judge_prompts.py 1 relevance   # Show TC1, relevance only
"""

import json
import sys


def load_raw_output(path="results/foundry_eval_output"):
    return json.loads(open(path, "r", encoding="utf-8").read())


EVALUATORS = [
    "relevance", "coherence", "groundedness",
    "fluency", "task_adherence", "intent_resolution",
]


def show_prompts(tc_index=1, evaluator_filter=None):
    data = load_raw_output()
    rows = data["rows"]

    if tc_index < 1 or tc_index > len(rows):
        print(f"ERROR: TC index must be 1-{len(rows)}, got {tc_index}")
        return

    row = rows[tc_index - 1]

    # Show the input data first
    print("=" * 80)
    print(f"  TEST CASE {tc_index} — JUDGE LLM PROMPTS & RESPONSES")
    print("=" * 80)
    print(f"\n  Query: {row.get('inputs.query', 'N/A')[:100]}...")
    print(f"  Response: {row.get('inputs.response', 'N/A')[:100]}...")
    print()

    evaluators = [evaluator_filter] if evaluator_filter else EVALUATORS

    for ev in evaluators:
        score = row.get(f"outputs.{ev}.{ev}", "N/A")
        result = row.get(f"outputs.{ev}.{ev}_result", "N/A")
        reason = row.get(f"outputs.{ev}.{ev}_reason", "N/A")
        model = row.get(f"outputs.{ev}.{ev}_model", "N/A")
        prompt_tokens = row.get(f"outputs.{ev}.{ev}_prompt_tokens", "?")
        completion_tokens = row.get(f"outputs.{ev}.{ev}_completion_tokens", "?")

        # The actual prompt sent to the judge LLM
        sample_input = row.get(f"outputs.{ev}.{ev}_sample_input", "")
        # The actual response from the judge LLM
        sample_output = row.get(f"outputs.{ev}.{ev}_sample_output", "")

        print("-" * 80)
        print(f"  EVALUATOR: {ev.upper()}")
        print(f"  Score: {score}  |  Result: {result}  |  Model: {model}")
        print(f"  Tokens: {prompt_tokens} prompt + {completion_tokens} completion")
        print("-" * 80)

        # Parse and pretty-print the judge prompt
        print(f"\n  >>> PROMPT SENT TO JUDGE LLM:")
        if sample_input:
            try:
                messages = json.loads(sample_input)
                for msg in messages:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    # Try to parse nested JSON for readability
                    try:
                        parsed = json.loads(content)
                        content = json.dumps(parsed, indent=4, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    print(f"\n  [{role}]:")
                    # Indent each line
                    for line in content.split("\n"):
                        print(f"    {line}")
                print()
            except json.JSONDecodeError:
                print(f"    {sample_input}")
        else:
            print("    (not captured)")

        # Parse and pretty-print the judge response
        print(f"  <<< JUDGE LLM RESPONSE:")
        if sample_output:
            try:
                messages = json.loads(sample_output)
                for msg in messages:
                    content = msg.get("content", "")
                    try:
                        parsed = json.loads(content)
                        content = json.dumps(parsed, indent=4, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    for line in content.split("\n"):
                        print(f"    {line}")
            except json.JSONDecodeError:
                print(f"    {sample_output}")
        else:
            print("    (not captured)")

        print(f"\n  EXTRACTED: score={score}, reason={reason}")
        print()


if __name__ == "__main__":
    tc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ev = sys.argv[2] if len(sys.argv) > 2 else None
    show_prompts(tc, ev)
