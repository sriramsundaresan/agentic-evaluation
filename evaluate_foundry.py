"""
Tier 2: Azure AI Foundry SDK Evaluation Pipeline.

Uses Microsoft's built-in evaluators from azure-ai-evaluation SDK
instead of our custom judge.py rubrics. Same agent, same test data,
but scored by Foundry's production-grade LLM-as-a-Judge evaluators.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

from azure.identity import DefaultAzureCredential
from azure.ai.evaluation import (
    RelevanceEvaluator,
    CoherenceEvaluator,
    GroundednessEvaluator,
    FluencyEvaluator,
    TaskAdherenceEvaluator,
    IntentResolutionEvaluator,
    ToolCallAccuracyEvaluator,
    ResponseCompletenessEvaluator,
    evaluate,
    EvaluatorConfig,
)

from agent import run_agent, create_client, SYSTEM_PROMPT
from tools import TOOL_DEFINITIONS


DETAILED_ANALYSIS_PROMPT = """You are an expert AI evaluation analyst. Given an AI agent's interaction, produce a structured claim-by-claim analysis.

## User Query
{query}

## Agent Response
{response}

## Tool Calls and Results (Context)
{context}

## Foundry Evaluation Scores
{scores_summary}

## Instructions
Analyze the agent's response and produce a JSON object with detailed analysis for each dimension.
For each dimension, provide specific evidence from the response and tool output.

Return ONLY valid JSON in this exact structure:
{{
  "groundedness": {{
    "claims": [
      {{"claim": "<exact text from response>", "evidence": "<matching tool output or 'No evidence in tool output'>", "grounded": true/false}}
    ]
  }},
  "relevance": {{
    "addressed_aspects": ["<aspect of query that was addressed>"],
    "missed_aspects": ["<aspect of query that was missed, if any>"]
  }},
  "task_adherence": {{
    "rules_followed": ["<rule the agent followed, e.g. 'Looked up order status before answering'>"],
    "rules_violated": ["<rule the agent violated, if any>"]
  }},
  "coherence": {{
    "strengths": ["<what was well-structured>"],
    "weaknesses": ["<structural issues, if any>"]
  }},
  "fluency": {{
    "strengths": ["<what was natural/clear>"],
    "weaknesses": ["<language issues, if any>"]
  }},
  "intent_resolution": {{
    "resolved": ["<user intent that was resolved>"],
    "unresolved": ["<user intent left unresolved, if any>"]
  }}
}}"""


def generate_detailed_analysis(
    query: str,
    response: str,
    tool_calls: list[dict],
    scores: dict,
    reasons: dict,
) -> dict:
    """
    Call the judge LLM to produce a structured claim-by-claim analysis
    for each evaluation dimension.
    """
    from agent import create_client as create_agent_client

    context = "\n".join(
        f"Tool: {tc['tool']}({json.dumps(tc['arguments'])})\n"
        f"Result: {json.dumps(tc['result'])}"
        for tc in tool_calls
    ) if tool_calls else "No tools were called."

    scores_summary = "\n".join(
        f"- {dim}: {score}" + (f" — {reasons.get(dim, '')}" if reasons.get(dim) else "")
        for dim, score in scores.items()
    )

    prompt = DETAILED_ANALYSIS_PROMPT.format(
        query=query,
        response=response,
        context=context,
        scores_summary=scores_summary,
    )

    client = create_agent_client()
    judge_model = os.environ.get(
        "AZURE_OPENAI_JUDGE_DEPLOYMENT_NAME",
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    )

    completion = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": "You are an evaluation analyst. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except (json.JSONDecodeError, IndexError):
        return {}


def messages_to_conversation(messages: list) -> tuple[list[dict], list[dict]]:
    """Convert OpenAI-format messages to the Foundry SDK conversation format
    expected by TaskAdherenceEvaluator.

    Returns (query_messages, response_messages) where tool calls are
    embedded in the response messages as content blocks.
    """
    query_msgs = []
    response_msgs = []

    for msg in messages:
        # Handle both plain dicts and OpenAI Pydantic objects
        role = msg["role"] if isinstance(msg, dict) else msg.role

        if role == "system":
            content = msg["content"] if isinstance(msg, dict) else msg.content
            query_msgs.append({"role": "system", "content": content})

        elif role == "user":
            content = msg["content"] if isinstance(msg, dict) else msg.content
            query_msgs.append({
                "role": "user",
                "content": [{"type": "text", "text": content}],
            })

        elif role == "assistant":
            # Check for tool_calls (Pydantic object or dict)
            tool_calls = None
            if isinstance(msg, dict):
                tool_calls = msg.get("tool_calls")
            elif hasattr(msg, "tool_calls"):
                tool_calls = msg.tool_calls

            if tool_calls:
                content_blocks = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn_name = tc["function"]["name"]
                        fn_args = tc["function"]["arguments"]
                        tc_id = tc["id"]
                    else:
                        fn_name = tc.function.name
                        fn_args = tc.function.arguments
                        tc_id = tc.id
                    # Parse arguments if they're a JSON string
                    if isinstance(fn_args, str):
                        fn_args = json.loads(fn_args)
                    content_blocks.append({
                        "type": "tool_call",
                        "name": fn_name,
                        "arguments": fn_args,
                        "tool_call_id": tc_id,
                    })
                response_msgs.append({"role": "assistant", "content": content_blocks})
            else:
                text = msg["content"] if isinstance(msg, dict) else msg.content
                if text:
                    response_msgs.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    })

        elif role == "tool":
            tool_call_id = msg["tool_call_id"] if isinstance(msg, dict) else msg.tool_call_id
            content = msg["content"] if isinstance(msg, dict) else msg.content
            response_msgs.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": [{"type": "tool_result", "tool_result": content}],
            })

    return query_msgs, response_msgs


def build_model_config() -> dict:
    """Build the model config dict for Foundry SDK evaluators."""
    return {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "azure_deployment": os.environ.get(
            "AZURE_OPENAI_JUDGE_DEPLOYMENT_NAME",
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        ),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    }


def load_test_cases(path: str = "data/test_cases.jsonl") -> list[dict]:
    """Load test cases from JSONL file."""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping line {line_num}: {e}")
    return cases


def run_agent_and_collect(test_cases: list[dict]) -> list[dict]:
    """
    Run our agent on each test case and collect results
    in a format the Foundry SDK evaluators expect.
    """
    agent_client = create_client()
    collected = []

    for i, tc in enumerate(test_cases, 1):
        query = tc["query"]
        expected_behavior = tc.get("expected_behavior", "")
        print(f"\n  --- Running Agent on Test Case {i}/{len(test_cases)} ---")
        print(f"  Query: {query[:80]}...")

        start = time.time()
        result = run_agent(query, client=agent_client)
        elapsed = time.time() - start
        print(f"  Agent responded in {elapsed:.1f}s | Tools used: {len(result['tool_calls'])}")

        # Build context from tool call results (for groundedness evaluation)
        context = "\n".join(
            f"Tool: {tc_item['tool']}({json.dumps(tc_item['arguments'])})\n"
            f"Result: {json.dumps(tc_item['result'])}"
            for tc_item in result["tool_calls"]
        ) if result["tool_calls"] else "No tools were called."

        # Build conversation-format data for TaskAdherenceEvaluator
        conv_query, conv_response = messages_to_conversation(result["messages"])

        collected.append({
            "query": query,
            "response": result["response"],
            "context": context,
            "expected_behavior": expected_behavior,
            "tool_calls": result["tool_calls"],
            "conversation_query": conv_query,
            "conversation_response": conv_response,
            "agent_latency_sec": round(elapsed, 2),
        })

    return collected


def run_foundry_evaluation(
    test_cases_path: str = "data/test_cases.jsonl",
    output_dir: str = "results",
) -> dict:
    """
    Run the Tier 2 evaluation pipeline:
    1. Load test cases
    2. Run agent on each query, collect responses + tool context
    3. Save agent outputs to a JSONL for the Foundry SDK
    4. Run Foundry SDK evaluate() with built-in evaluators
    5. Aggregate and save report
    """
    print("=" * 70)
    print("  TIER 2: AZURE AI FOUNDRY SDK EVALUATION")
    print("=" * 70)

    # Step 1: Load test data
    print(f"\n[1/5] Loading test cases from {test_cases_path}...")
    test_cases = load_test_cases(test_cases_path)
    print(f"       Loaded {len(test_cases)} test cases")

    # Step 2: Run agent and collect outputs
    model_config = build_model_config()
    judge_deployment = model_config["azure_deployment"]
    print(f"\n[2/5] Running agent on test cases...")
    print(f"       Agent model: {os.environ['AZURE_OPENAI_DEPLOYMENT_NAME']}")
    print(f"       Judge model: {judge_deployment}")
    collected = run_agent_and_collect(test_cases)

    # Step 3: Save agent outputs as JSONL for Foundry SDK
    print(f"\n[3/5] Preparing data for Foundry SDK evaluators...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    eval_data_path = os.path.join(output_dir, "foundry_eval_input.jsonl")
    with open(eval_data_path, "w", encoding="utf-8") as f:
        for row in collected:
            f.write(json.dumps({
                "query": row["query"],
                "response": row["response"],
                "context": row["context"],
                "conversation_query": row["conversation_query"],
                "conversation_response": row["conversation_response"],
            }, ensure_ascii=False) + "\n")

    # Step 4: Run Foundry SDK evaluate()
    print(f"\n[4/5] Running Foundry SDK evaluators (this calls the judge LLM)...")
    start_time = time.time()

    credential = DefaultAzureCredential()

    eval_result = evaluate(
        data=eval_data_path,
        evaluators={
            "relevance": RelevanceEvaluator(model_config=model_config, credential=credential),
            "coherence": CoherenceEvaluator(model_config=model_config, credential=credential),
            "groundedness": GroundednessEvaluator(model_config=model_config, credential=credential),
            "fluency": FluencyEvaluator(model_config=model_config, credential=credential),
            "task_adherence": TaskAdherenceEvaluator(model_config=model_config, credential=credential),
            "intent_resolution": IntentResolutionEvaluator(model_config=model_config, credential=credential),
        },
        evaluator_config={
            "task_adherence": EvaluatorConfig(
                column_mapping={
                    "query": "${data.conversation_query}",
                    "response": "${data.conversation_response}",
                }
            ),
        },
        output_path=os.path.join(output_dir, "foundry_eval_output"),
    )

    eval_time = time.time() - start_time
    print(f"       Foundry SDK evaluation completed in {eval_time:.1f}s")

    # Step 5: Build report
    print(f"\n[5/5] Generating evaluation report...")
    report = build_report(eval_result, collected, output_dir)

    return report


def build_report(eval_result, collected: list[dict], output_dir: str) -> dict:
    """Build a structured report from Foundry SDK evaluation results."""
    metrics = eval_result.get("metrics", {})
    rows = eval_result.get("rows", [])

    # The evaluator names we registered
    EVALUATORS = [
        "relevance", "coherence", "groundedness",
        "fluency", "task_adherence", "intent_resolution",
    ]

    # Extract dimension averages from metrics.
    # Score keys follow the pattern "<evaluator>.<evaluator>" e.g. "relevance.relevance"
    dimension_scores = {}
    for ev in EVALUATORS:
        score_key = f"{ev}.{ev}"
        if score_key in metrics and isinstance(metrics[score_key], (int, float)):
            dimension_scores[ev] = round(metrics[score_key], 2)

    # Per-row details — extract only the actual score (not token counts, thresholds, etc.)
    # Score fields: "outputs.<evaluator>.<evaluator>" e.g. "outputs.relevance.relevance"
    # Reason fields: "outputs.<evaluator>.<evaluator>_reason"
    detailed_results = []
    for i, (row, agent_data) in enumerate(zip(rows, collected)):
        row_scores = {}
        row_reasons = {}
        for ev in EVALUATORS:
            score_key = f"outputs.{ev}.{ev}"
            reason_key = f"outputs.{ev}.{ev}_reason"
            if score_key in row and isinstance(row[score_key], (int, float)):
                row_scores[ev] = row[score_key]
            if reason_key in row and isinstance(row[reason_key], str):
                row_reasons[ev] = row[reason_key]

        # task_adherence uses 0/1 scale; others use 1-5.
        # For the per-row average, rescale task_adherence to 1-5 range.
        scores_for_avg = {}
        for dim, score in row_scores.items():
            if dim == "task_adherence":
                scores_for_avg[dim] = 1.0 + score * 4.0  # 0→1, 1→5
            else:
                scores_for_avg[dim] = score

        avg_score = round(
            sum(scores_for_avg.values()) / len(scores_for_avg), 2
        ) if scores_for_avg else 0

        detailed_results.append({
            "test_case_index": i + 1,
            "query": agent_data["query"],
            "expected_behavior": agent_data["expected_behavior"],
            "agent_response": agent_data["response"],
            "tool_calls": agent_data["tool_calls"],
            "foundry_scores": row_scores,
            "foundry_reasons": row_reasons,
            "average_score": avg_score,
            "pass": avg_score >= 4.0,
            "agent_latency_sec": agent_data["agent_latency_sec"],
        })

    total = len(detailed_results)
    passed = sum(1 for r in detailed_results if r["pass"])
    overall_avg = round(sum(r["average_score"] for r in detailed_results) / total, 2) if total else 0

    report = {
        "evaluation_id": datetime.now().strftime("foundry-eval-%Y%m%d-%H%M%S"),
        "evaluation_type": "Tier 2 — Azure AI Foundry SDK",
        "timestamp": datetime.now().isoformat(),
        "total_test_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{passed}/{total} ({100 * passed / total:.0f}%)" if total else "0/0",
        "overall_average_score": overall_avg,
        "dimension_averages": dimension_scores,
        "detailed_results": detailed_results,
    }

    # Save report
    report_path = os.path.join(output_dir, f"{report['evaluation_id']}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("  TIER 2 EVALUATION REPORT — Azure AI Foundry SDK")
    print("=" * 70)
    print(f"  Evaluation ID:    {report['evaluation_id']}")
    print(f"  Test Cases:       {total}")
    print(f"  Passed:           {passed} | Failed: {total - passed}")
    print(f"  Pass Rate:        {report['pass_rate']}")
    print(f"  Overall Average:  {overall_avg}/5")
    print()
    print("  Foundry SDK Dimension Averages:")
    print(f"  {'Dimension':<25} {'Avg Score':>10} {'Scale':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*8}")
    for dim, avg in sorted(dimension_scores.items()):
        scale = "0-1" if dim == "task_adherence" else "1-5"
        print(f"  {dim:<25} {avg:>10} {scale:>8}")

    # Per test case summary
    print()
    for r in detailed_results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  TC {r['test_case_index']}: {status} (avg {r['average_score']}/5) — {r['query'][:55]}...")
        for dim, score in r["foundry_scores"].items():
            if dim == "task_adherence":
                label = "1" if score >= 1.0 else "0"
                icon = "✓" if score >= 1.0 else "✗"
                print(f"    {icon} {dim}: {label} (binary)")
            else:
                icon = "✓" if score >= 4 else "✗"
                print(f"    {icon} {dim}: {score}/5")

    print(f"\n  Report saved to: {report_path}")
    print("=" * 70)

    return report


def evaluate_single(query: str, response: str, tool_calls: list[dict], messages: list) -> dict:
    """
    Run Foundry SDK evaluators on a single agent interaction.

    Returns a dict with per-dimension scores, reasons, and pass/fail.
    """
    # Look up expected_behavior from test cases
    expected_behavior = ""
    try:
        test_cases = load_test_cases()
        for tc in test_cases:
            if tc["query"] == query:
                expected_behavior = tc.get("expected_behavior", "")
                break
    except Exception:
        pass

    # Build context from tool call results (for groundedness)
    context = "\n".join(
        f"Tool: {tc_item['tool']}({json.dumps(tc_item['arguments'])})\n"
        f"Result: {json.dumps(tc_item['result'])}"
        for tc_item in tool_calls
    ) if tool_calls else "No tools were called."

    # Build conversation format for TaskAdherenceEvaluator
    conv_query, conv_response = messages_to_conversation(messages)

    # Write single-row JSONL for the SDK
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    eval_data_path = os.path.join(tmp_dir, "eval_input.jsonl")
    with open(eval_data_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "query": query,
            "response": response,
            "context": context,
            "conversation_query": conv_query,
            "conversation_response": conv_response,
        }, ensure_ascii=False) + "\n")

    # Run Foundry SDK evaluators
    model_config = build_model_config()
    credential = DefaultAzureCredential()

    eval_result = evaluate(
        data=eval_data_path,
        evaluators={
            "relevance": RelevanceEvaluator(model_config=model_config, credential=credential),
            "coherence": CoherenceEvaluator(model_config=model_config, credential=credential),
            "groundedness": GroundednessEvaluator(model_config=model_config, credential=credential),
            "fluency": FluencyEvaluator(model_config=model_config, credential=credential),
            "task_adherence": TaskAdherenceEvaluator(model_config=model_config, credential=credential),
            "intent_resolution": IntentResolutionEvaluator(model_config=model_config, credential=credential),
        },
        evaluator_config={
            "task_adherence": EvaluatorConfig(
                column_mapping={
                    "query": "${data.conversation_query}",
                    "response": "${data.conversation_response}",
                }
            ),
        },
        output_path=os.path.join(tmp_dir, "eval_output"),
    )

    # Extract scores and reasons from the single row
    EVALUATORS = [
        "relevance", "coherence", "groundedness",
        "fluency", "task_adherence", "intent_resolution",
    ]
    rows = eval_result.get("rows", [])
    row = rows[0] if rows else {}

    scores = {}
    reasons = {}
    for ev in EVALUATORS:
        score_key = f"outputs.{ev}.{ev}"
        reason_key = f"outputs.{ev}.{ev}_reason"
        if score_key in row and isinstance(row[score_key], (int, float)):
            scores[ev] = row[score_key]
        if reason_key in row and isinstance(row[reason_key], str):
            reasons[ev] = row[reason_key]

    # --- Tool Call Accuracy (called directly — needs tool_definitions) ---
    try:
        tca = ToolCallAccuracyEvaluator(model_config=model_config, credential=credential)
        tca_result = tca(
            query=conv_query,
            tool_definitions=TOOL_DEFINITIONS,
            response=conv_response,
        )
        if "tool_call_accuracy" in tca_result:
            scores["tool_call_accuracy"] = tca_result["tool_call_accuracy"]
        if "tool_call_accuracy_reason" in tca_result:
            reasons["tool_call_accuracy"] = tca_result["tool_call_accuracy_reason"]
    except Exception as e:
        print(f"  ToolCallAccuracy evaluator error: {e}")

    # --- Response Completeness (needs ground_truth / expected_behavior) ---
    if expected_behavior:
        try:
            rc = ResponseCompletenessEvaluator(model_config=model_config, credential=credential)
            rc_result = rc(response=response, ground_truth=expected_behavior)
            if "response_completeness" in rc_result:
                scores["response_completeness"] = rc_result["response_completeness"]
            if "response_completeness_reason" in rc_result:
                reasons["response_completeness"] = rc_result["response_completeness_reason"]
        except Exception as e:
            print(f"  ResponseCompleteness evaluator error: {e}")

    # Compute average (rescale task_adherence from 0-1 to 1-5 for averaging)
    scores_for_avg = {}
    for dim, score in scores.items():
        if dim == "task_adherence":
            scores_for_avg[dim] = 1.0 + score * 4.0
        else:
            scores_for_avg[dim] = score
    avg_score = round(
        sum(scores_for_avg.values()) / len(scores_for_avg), 2
    ) if scores_for_avg else 0

    # Generate detailed claim-by-claim analysis
    detailed_analysis = {}
    try:
        detailed_analysis = generate_detailed_analysis(
            query=query,
            response=response,
            tool_calls=tool_calls,
            scores=scores,
            reasons=reasons,
        )
    except Exception as e:
        detailed_analysis = {"error": str(e)}

    return {
        "query": query,
        "expected_behavior": expected_behavior,
        "scores": scores,
        "reasons": reasons,
        "average_score": avg_score,
        "pass": avg_score >= 4.0,
        "detailed_analysis": detailed_analysis,
    }


def run_model_only(query: str) -> dict:
    """Run the LLM on the query WITHOUT tools — pure model response."""
    client = create_client()
    model = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    return {
        "response": completion.choices[0].message.content or "",
    }


def evaluate_model_only(query: str) -> dict:
    """
    Run the same query through the model WITHOUT tools, then evaluate.
    Shows what happens when the model has no tool access — useful for
    demonstrating the value of agentic tool use.
    """
    model_result = run_model_only(query)
    model_response = model_result["response"]

    # Look up expected_behavior from test cases
    expected_behavior = ""
    try:
        test_cases = load_test_cases()
        for tc in test_cases:
            if tc["query"] == query:
                expected_behavior = tc.get("expected_behavior", "")
                break
    except Exception:
        pass

    # Write single-row JSONL
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    eval_data_path = os.path.join(tmp_dir, "model_eval_input.jsonl")
    with open(eval_data_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "query": query,
            "response": model_response,
            "context": "No tools or external data sources were used. "
                       "The model answered from its training knowledge only.",
        }, ensure_ascii=False) + "\n")

    model_config = build_model_config()
    credential = DefaultAzureCredential()

    eval_result = evaluate(
        data=eval_data_path,
        evaluators={
            "relevance": RelevanceEvaluator(model_config=model_config, credential=credential),
            "coherence": CoherenceEvaluator(model_config=model_config, credential=credential),
            "groundedness": GroundednessEvaluator(model_config=model_config, credential=credential),
            "fluency": FluencyEvaluator(model_config=model_config, credential=credential),
            "intent_resolution": IntentResolutionEvaluator(model_config=model_config, credential=credential),
        },
        output_path=os.path.join(tmp_dir, "model_eval_output"),
    )

    # Extract scores
    MODEL_EVALUATORS = ["relevance", "coherence", "groundedness", "fluency", "intent_resolution"]
    rows = eval_result.get("rows", [])
    row = rows[0] if rows else {}

    scores = {}
    reasons = {}
    for ev in MODEL_EVALUATORS:
        score_key = f"outputs.{ev}.{ev}"
        reason_key = f"outputs.{ev}.{ev}_reason"
        if score_key in row and isinstance(row[score_key], (int, float)):
            scores[ev] = row[score_key]
        if reason_key in row and isinstance(row[reason_key], str):
            reasons[ev] = row[reason_key]

    # Response Completeness (if expected_behavior available)
    if expected_behavior:
        try:
            rc = ResponseCompletenessEvaluator(model_config=model_config, credential=credential)
            rc_result = rc(response=model_response, ground_truth=expected_behavior)
            if "response_completeness" in rc_result:
                scores["response_completeness"] = rc_result["response_completeness"]
            if "response_completeness_reason" in rc_result:
                reasons["response_completeness"] = rc_result["response_completeness_reason"]
        except Exception as e:
            print(f"  ResponseCompleteness evaluator error: {e}")

    avg_score = round(
        sum(scores.values()) / len(scores), 2
    ) if scores else 0

    return {
        "query": query,
        "model_response": model_response,
        "expected_behavior": expected_behavior,
        "scores": scores,
        "reasons": reasons,
        "average_score": avg_score,
        "pass": avg_score >= 4.0,
    }


if __name__ == "__main__":
    report = run_foundry_evaluation()
