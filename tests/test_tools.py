"""
Unit tests for tools.py — simulated backend tools used by the agent.
These tests require no external dependencies (no Azure credentials).
"""

import json
import pytest

from tools import get_order_status, search_kb, process_refund, REFUND_LOG


class TestGetOrderStatus:
    def test_known_order_returns_data(self):
        result = json.loads(get_order_status("12345"))
        assert result["order_id"] == "12345"
        assert result["product"] == "Laptop - Dell XPS 15"
        assert result["status"] == "shipped"
        assert result["carrier"] == "FedEx"
        assert result["tracking_number"] == "FX789456123"

    def test_delivered_order(self):
        result = json.loads(get_order_status("67890"))
        assert result["status"] == "delivered"
        assert "delivered_date" in result

    def test_processing_order(self):
        result = json.loads(get_order_status("11111"))
        assert result["status"] == "processing"
        assert result["carrier"] is None
        assert result["tracking_number"] is None

    def test_cancelled_order(self):
        result = json.loads(get_order_status("99999"))
        assert result["status"] == "cancelled"
        assert "cancellation_reason" in result

    def test_unknown_order_returns_error(self):
        result = json.loads(get_order_status("00000"))
        assert "error" in result
        assert "00000" in result["error"]

    def test_returns_valid_json_string(self):
        raw = get_order_status("12345")
        assert isinstance(raw, str)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


class TestSearchKb:
    def test_late_delivery_query(self):
        result = json.loads(search_kb("late delivery"))
        assert isinstance(result, list)
        assert len(result) > 0
        topics = [r["topic"] for r in result]
        assert any("late delivery" in t for t in topics)

    def test_refund_policy_query(self):
        result = json.loads(search_kb("refund policy"))
        assert isinstance(result, list)
        topics = [r["topic"] for r in result]
        assert any("refund" in t for t in topics)

    def test_shipping_policy_query(self):
        result = json.loads(search_kb("shipping"))
        assert isinstance(result, list)
        topics = [r["topic"] for r in result]
        assert any("shipping" in t for t in topics)

    def test_cancellation_policy_query(self):
        result = json.loads(search_kb("cancellation policy"))
        assert isinstance(result, list)
        topics = [r["topic"] for r in result]
        assert any("cancellation" in t for t in topics)

    def test_return_policy_query(self):
        result = json.loads(search_kb("return policy"))
        assert isinstance(result, list)
        topics = [r["topic"] for r in result]
        assert any("return" in t for t in topics)

    def test_unknown_topic_returns_no_results_message(self):
        result = json.loads(search_kb("xyznonexistent12345"))
        assert "message" in result
        assert "No matching" in result["message"]

    def test_case_insensitive_search(self):
        lower = json.loads(search_kb("REFUND"))
        assert isinstance(lower, list)

    def test_result_includes_policy_text(self):
        result = json.loads(search_kb("late delivery"))
        assert isinstance(result, list)
        assert "policy" in result[0]


class TestProcessRefund:
    def setup_method(self):
        """Clear the global refund log before each test."""
        REFUND_LOG.clear()

    def test_refund_for_delivered_order(self):
        result = json.loads(process_refund("67890", "damaged product"))
        assert "refund_id" in result
        assert result["order_id"] == "67890"
        assert result["status"] == "initiated"
        assert result["amount"] == 79.99
        assert "estimated_completion" in result

    def test_refund_id_is_unique(self):
        r1 = json.loads(process_refund("67890", "reason one"))
        r2 = json.loads(process_refund("67890", "reason two"))
        assert r1["refund_id"] != r2["refund_id"]

    def test_refund_logged_to_global_list(self):
        assert len(REFUND_LOG) == 0
        process_refund("67890", "damaged")
        assert len(REFUND_LOG) == 1

    def test_refund_for_unknown_order_returns_error(self):
        result = json.loads(process_refund("00000", "test"))
        assert "error" in result
        assert "not found" in result["error"]

    def test_refund_for_cancelled_order_returns_error(self):
        result = json.loads(process_refund("99999", "test"))
        assert "error" in result
        assert "cancelled" in result["error"]

    def test_reason_stored_in_refund_record(self):
        reason = "product arrived broken"
        result = json.loads(process_refund("67890", reason))
        assert result["reason"] == reason
