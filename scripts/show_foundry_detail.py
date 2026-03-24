"""Show detailed Foundry SDK evaluation results."""
import json

data = json.loads(open("results/foundry_eval_output", "r").read())

print("TC  Relev  Coher  Ground  Fluency  TaskAdh  Intent")
print("--  -----  -----  ------  -------  -------  ------")
for i, row in enumerate(data["rows"]):
    r = row.get("outputs.relevance.relevance", "?")
    c = row.get("outputs.coherence.coherence", "?")
    g = row.get("outputs.groundedness.groundedness", "?")
    f = row.get("outputs.fluency.fluency", "?")
    t = row.get("outputs.task_adherence.task_adherence", "?")
    ir = row.get("outputs.intent_resolution.intent_resolution", "?")
    print(f"{i+1:2}  {r:>5}  {c:>5}  {g:>6}  {f:>7}  {t:>7}  {ir:>6}")

m = data["metrics"]
print()
print("Averages:")
for key in ["relevance.relevance", "coherence.coherence", "groundedness.groundedness",
            "fluency.fluency", "task_adherence.task_adherence", "intent_resolution.intent_resolution"]:
    dim = key.split(".")[0]
    print(f"  {dim:<20} {m[key]}")

# Show task_adherence reasons for all TCs (it's the interesting one)
print()
print("=" * 70)
print("TASK ADHERENCE REASONS (the strictest evaluator):")
print("=" * 70)
for i, row in enumerate(data["rows"]):
    score = row.get("outputs.task_adherence.task_adherence", "?")
    result = row.get("outputs.task_adherence.task_adherence_result", "?")
    reason = row.get("outputs.task_adherence.task_adherence_reason", "N/A")
    print(f"\nTC {i+1} — Score: {score}, Result: {result}")
    print(f"  {reason}")
