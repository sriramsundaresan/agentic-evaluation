"""
Unit tests for evaluate_foundry.py — model evaluation pipeline.

These tests mock all Azure API calls (OpenAI SDK and AI Evaluation SDK)
so no real credentials are required.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Set required env vars before importing the module under test
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://mock.openai.azure.com/")
os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mock")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

from evaluate_foundry import (
    load_test_cases,
    messages_to_conversation,
    build_model_config,
    build_model_report,
    run_model_only,
    run_model_evaluation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_foundry_eval_result(n_rows: int = 2, score: float = 4.0) -> dict:
    """Build a minimal Foundry SDK evaluate() return value."""
    evaluator_names = ["relevance", "coherence", "groundedness", "fluency", "intent_resolution"]
    rows = []
    for _ in range(n_rows):
        row = {}
        for ev in evaluator_names:
            row[f"outputs.{ev}.{ev}"] = score
            row[f"outputs.{ev}.{ev}_reason"] = f"Mock reason for {ev}."
        rows.append(row)

    metrics = {f"{ev}.{ev}": score for ev in evaluator_names}
    return {"rows": rows, "metrics": metrics}


# ---------------------------------------------------------------------------
# load_test_cases
# ---------------------------------------------------------------------------

class TestLoadTestCases:
    def test_loads_bundled_test_cases(self, tmp_path):
        cases = [
            {"query": "q1", "expected_behavior": "b1"},
            {"query": "q2", "expected_behavior": "b2"},
        ]
        path = tmp_path / "test_cases.jsonl"
        path.write_text("\n".join(json.dumps(c) for c in cases))
        result = load_test_cases(str(path))
        assert len(result) == 2
        assert result[0]["query"] == "q1"
        assert result[1]["expected_behavior"] == "b2"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "cases.jsonl"
        path.write_text('{"query": "q"}\n\n{"query": "q2"}\n')
        result = load_test_cases(str(path))
        assert len(result) == 2

    def test_real_test_cases_file(self):
        """The bundled data/test_cases.jsonl must load without errors."""
        result = load_test_cases("data/test_cases.jsonl")
        assert len(result) >= 1
        for tc in result:
            assert "query" in tc


# ---------------------------------------------------------------------------
# messages_to_conversation
# ---------------------------------------------------------------------------

class TestMessagesToConversation:
    def test_simple_exchange(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        query_msgs, response_msgs = messages_to_conversation(messages)
        roles_q = [m["role"] for m in query_msgs]
        assert "system" in roles_q
        assert "user" in roles_q
        roles_r = [m["role"] for m in response_msgs]
        assert "assistant" in roles_r

    def test_tool_call_appears_in_response(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Check my order"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "function": {
                            "name": "get_order_status",
                            "arguments": '{"order_id": "12345"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"status": "shipped"}',
            },
            {"role": "assistant", "content": "Your order is shipped."},
        ]
        query_msgs, response_msgs = messages_to_conversation(messages)
        # At least one response message should represent the tool call
        content_blocks = [
            block
            for msg in response_msgs
            for block in (msg.get("content") or [])
            if isinstance(block, dict)
        ]
        types = [b.get("type") for b in content_blocks]
        assert "tool_call" in types or "tool_result" in types

    def test_empty_messages_returns_empty_lists(self):
        q, r = messages_to_conversation([])
        assert q == []
        assert r == []


# ---------------------------------------------------------------------------
# build_model_config
# ---------------------------------------------------------------------------

class TestBuildModelConfig:
    def test_returns_required_keys(self):
        config = build_model_config()
        assert "azure_endpoint" in config
        assert "azure_deployment" in config
        assert "api_version" in config

    def test_endpoint_comes_from_env(self):
        config = build_model_config()
        assert config["azure_endpoint"] == os.environ["AZURE_OPENAI_ENDPOINT"]


# ---------------------------------------------------------------------------
# build_model_report
# ---------------------------------------------------------------------------

class TestBuildModelReport:
    EVALUATORS = ["relevance", "coherence", "groundedness", "fluency", "intent_resolution"]

    def _make_model_results(self, n: int = 2) -> list[dict]:
        return [
            {
                "query": f"query {i}",
                "response": f"response {i}",
                "expected_behavior": f"behavior {i}",
                "model_latency_sec": 1.0,
            }
            for i in range(n)
        ]

    def test_report_has_required_keys(self, tmp_path):
        eval_result = _make_foundry_eval_result(n_rows=2)
        collected = self._make_model_results(2)
        report = build_model_report(eval_result, collected, self.EVALUATORS, str(tmp_path))
        for key in ["evaluation_id", "evaluation_type", "total_test_cases",
                    "passed", "failed", "pass_rate", "overall_average_score",
                    "dimension_averages", "detailed_results"]:
            assert key in report, f"Missing key: {key}"

    def test_report_counts_match_input(self, tmp_path):
        n = 3
        eval_result = _make_foundry_eval_result(n_rows=n, score=4.5)
        collected = self._make_model_results(n)
        report = build_model_report(eval_result, collected, self.EVALUATORS, str(tmp_path))
        assert report["total_test_cases"] == n
        assert report["passed"] + report["failed"] == n

    def test_report_is_saved_to_disk(self, tmp_path):
        eval_result = _make_foundry_eval_result(n_rows=1)
        collected = self._make_model_results(1)
        report = build_model_report(eval_result, collected, self.EVALUATORS, str(tmp_path))
        report_id = report["evaluation_id"]
        saved = tmp_path / f"{report_id}.json"
        assert saved.exists()
        loaded = json.loads(saved.read_text())
        assert loaded["evaluation_id"] == report_id

    def test_pass_requires_average_score_4(self, tmp_path):
        # Score 3.0 → should fail
        eval_result = _make_foundry_eval_result(n_rows=1, score=3.0)
        collected = self._make_model_results(1)
        report = build_model_report(eval_result, collected, self.EVALUATORS, str(tmp_path))
        assert report["passed"] == 0
        assert report["failed"] == 1

    def test_high_score_passes(self, tmp_path):
        eval_result = _make_foundry_eval_result(n_rows=1, score=5.0)
        collected = self._make_model_results(1)
        report = build_model_report(eval_result, collected, self.EVALUATORS, str(tmp_path))
        assert report["passed"] == 1

    def test_evaluation_type_is_model_only(self, tmp_path):
        eval_result = _make_foundry_eval_result(n_rows=1)
        collected = self._make_model_results(1)
        report = build_model_report(eval_result, collected, self.EVALUATORS, str(tmp_path))
        assert "Model-Only" in report["evaluation_type"] or "model" in report["evaluation_type"].lower()

    def test_dimension_averages_present(self, tmp_path):
        eval_result = _make_foundry_eval_result(n_rows=2, score=4.2)
        collected = self._make_model_results(2)
        report = build_model_report(eval_result, collected, self.EVALUATORS, str(tmp_path))
        for ev in self.EVALUATORS:
            assert ev in report["dimension_averages"]


# ---------------------------------------------------------------------------
# run_model_only — mocked Azure OpenAI client
# ---------------------------------------------------------------------------

class TestRunModelOnly:
    def _mock_openai_response(self, text: str) -> MagicMock:
        """Return a minimal mock that matches OpenAI chat completion shape."""
        choice = MagicMock()
        choice.message.content = text
        completion = MagicMock()
        completion.choices = [choice]
        return completion

    @patch("evaluate_foundry.create_client")
    def test_returns_response_string(self, mock_create_client):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            "Your order is on the way."
        )
        result = run_model_only("Where is my order?")
        assert "response" in result
        assert result["response"] == "Your order is on the way."

    @patch("evaluate_foundry.create_client")
    def test_no_tool_definitions_passed(self, mock_create_client):
        """Model-only call must NOT pass any tools."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response("ok")
        run_model_only("Test query")
        call_kwargs = mock_client.chat.completions.create.call_args
        # tools param should not be present in keyword args
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        assert "tools" not in kwargs
        # tools param should not be passed as positional arg either
        positional = call_kwargs.args if call_kwargs.args else ()
        assert "tools" not in str(positional)


# ---------------------------------------------------------------------------
# run_model_evaluation — mocked Azure clients + Foundry SDK
# ---------------------------------------------------------------------------

class TestRunModelEvaluation:
    def _mock_openai_response(self, text: str = "Mock model response") -> MagicMock:
        choice = MagicMock()
        choice.message.content = text
        completion = MagicMock()
        completion.choices = [choice]
        return completion

    @patch("evaluate_foundry.DefaultAzureCredential")
    @patch("evaluate_foundry.evaluate")
    @patch("evaluate_foundry.create_client")
    def test_run_model_evaluation_returns_report(
        self, mock_create_client, mock_evaluate, mock_credential, tmp_path
    ):
        # Mock the Azure OpenAI client
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response()

        # Mock Foundry SDK evaluate()
        n_cases = 2
        mock_evaluate.return_value = _make_foundry_eval_result(n_rows=n_cases, score=4.0)

        # Use a tiny JSONL with 2 test cases
        test_jsonl = tmp_path / "cases.jsonl"
        cases = [
            {"query": "q1", "expected_behavior": "b1"},
            {"query": "q2", "expected_behavior": "b2"},
        ]
        test_jsonl.write_text("\n".join(json.dumps(c) for c in cases))

        report = run_model_evaluation(
            test_cases_path=str(test_jsonl),
            output_dir=str(tmp_path),
        )

        assert report["total_test_cases"] == n_cases
        assert "dimension_averages" in report
        assert "detailed_results" in report
        assert len(report["detailed_results"]) == n_cases

    @patch("evaluate_foundry.DefaultAzureCredential")
    @patch("evaluate_foundry.evaluate")
    @patch("evaluate_foundry.create_client")
    def test_model_eval_input_jsonl_written(
        self, mock_create_client, mock_evaluate, mock_credential, tmp_path
    ):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response("Answer")

        mock_evaluate.return_value = _make_foundry_eval_result(n_rows=1, score=4.0)

        test_jsonl = tmp_path / "cases.jsonl"
        test_jsonl.write_text(json.dumps({"query": "q1", "expected_behavior": "b1"}))

        run_model_evaluation(
            test_cases_path=str(test_jsonl),
            output_dir=str(tmp_path),
        )

        input_file = tmp_path / "model_eval_input.jsonl"
        assert input_file.exists()
        row = json.loads(input_file.read_text().strip())
        assert row["query"] == "q1"
        assert row["response"] == "Answer"
        assert "No tools" in row["context"]

    @patch("evaluate_foundry.DefaultAzureCredential")
    @patch("evaluate_foundry.evaluate")
    @patch("evaluate_foundry.create_client")
    def test_model_eval_report_saved_to_disk(
        self, mock_create_client, mock_evaluate, mock_credential, tmp_path
    ):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response()

        mock_evaluate.return_value = _make_foundry_eval_result(n_rows=1, score=4.0)

        test_jsonl = tmp_path / "cases.jsonl"
        test_jsonl.write_text(json.dumps({"query": "q1", "expected_behavior": "b1"}))

        report = run_model_evaluation(
            test_cases_path=str(test_jsonl),
            output_dir=str(tmp_path),
        )

        report_path = tmp_path / f"{report['evaluation_id']}.json"
        assert report_path.exists()
        loaded = json.loads(report_path.read_text())
        assert loaded["evaluation_id"] == report["evaluation_id"]

    @patch("evaluate_foundry.DefaultAzureCredential")
    @patch("evaluate_foundry.evaluate")
    @patch("evaluate_foundry.create_client")
    def test_foundry_evaluate_called_once(
        self, mock_create_client, mock_evaluate, mock_credential, tmp_path
    ):
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response()

        mock_evaluate.return_value = _make_foundry_eval_result(n_rows=2, score=4.0)

        test_jsonl = tmp_path / "cases.jsonl"
        cases = [
            {"query": "q1", "expected_behavior": "b1"},
            {"query": "q2", "expected_behavior": "b2"},
        ]
        test_jsonl.write_text("\n".join(json.dumps(c) for c in cases))

        run_model_evaluation(
            test_cases_path=str(test_jsonl),
            output_dir=str(tmp_path),
        )

        assert mock_evaluate.call_count == 1
