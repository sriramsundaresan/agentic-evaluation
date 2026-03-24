"""
Simulated tools for the Customer Support Agent.
These mock real backend systems (order DB, knowledge base, refund service).
In production, these would call actual APIs/databases.
"""

import json
from datetime import datetime, timedelta

# --- Simulated Data ---

ORDERS_DB = {
    "12345": {
        "order_id": "12345",
        "product": "Laptop - Dell XPS 15",
        "status": "shipped",
        "carrier": "FedEx",
        "tracking_number": "FX789456123",
        "order_date": "2026-03-09",
        "estimated_delivery": "2026-03-20",
        "amount": 1299.99,
        "customer_name": "Alice Johnson",
    },
    "67890": {
        "order_id": "67890",
        "product": "Wireless Mouse - Logitech MX",
        "status": "delivered",
        "carrier": "UPS",
        "tracking_number": "UP123789456",
        "order_date": "2026-03-01",
        "estimated_delivery": "2026-03-08",
        "delivered_date": "2026-03-07",
        "amount": 79.99,
        "customer_name": "Bob Smith",
    },
    "11111": {
        "order_id": "11111",
        "product": "Monitor - Samsung 27 inch",
        "status": "processing",
        "carrier": None,
        "tracking_number": None,
        "order_date": "2026-03-22",
        "estimated_delivery": "2026-03-30",
        "amount": 449.99,
        "customer_name": "Carol Davis",
    },
    "99999": {
        "order_id": "99999",
        "product": "Keyboard - Mechanical RGB",
        "status": "cancelled",
        "carrier": None,
        "tracking_number": None,
        "order_date": "2026-03-15",
        "estimated_delivery": None,
        "amount": 129.99,
        "customer_name": "Dave Wilson",
        "cancellation_reason": "Customer requested cancellation",
    },
}

KNOWLEDGE_BASE = {
    "late delivery policy": {
        "policy": "If delivery is 3 or more days past the estimated delivery date, "
        "the customer is eligible for a $10 account credit or free reshipment of the item. "
        "Customer must contact support to initiate either option.",
        "last_updated": "2026-01-15",
    },
    "refund policy": {
        "policy": "Refunds are available within 30 days of delivery for undamaged items. "
        "For damaged items, refunds or replacements are available within 60 days of delivery. "
        "Refund is processed to the original payment method within 5-7 business days.",
        "last_updated": "2026-02-01",
    },
    "cancellation policy": {
        "policy": "Orders can be cancelled before they are shipped. Once shipped, "
        "the customer must wait for delivery and then initiate a return. "
        "Cancellation refunds are processed within 2-3 business days.",
        "last_updated": "2026-01-20",
    },
    "return policy": {
        "policy": "Items can be returned within 30 days of delivery in original packaging. "
        "Customer receives a prepaid return label. Refund is issued after item inspection, "
        "typically within 5-7 business days of receiving the return.",
        "last_updated": "2026-02-10",
    },
    "shipping policy": {
        "policy": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days "
        "at an additional cost of $15. Free shipping on orders over $50. "
        "Tracking information is emailed once the order ships.",
        "last_updated": "2026-01-10",
    },
}

REFUND_LOG: list[dict] = []


# --- Tool Functions ---

def get_order_status(order_id: str) -> str:
    """Look up the current status of an order by order ID."""
    order = ORDERS_DB.get(order_id)
    if order is None:
        return json.dumps({"error": f"Order {order_id} not found"})
    return json.dumps(order)


def search_kb(query: str) -> str:
    """Search the knowledge base for policy information."""
    query_lower = query.lower()
    results = []
    for topic, content in KNOWLEDGE_BASE.items():
        if any(word in query_lower for word in topic.split()):
            results.append({"topic": topic, **content})
    if not results:
        return json.dumps({"message": "No matching knowledge base articles found", "query": query})
    return json.dumps(results)


def process_refund(order_id: str, reason: str) -> str:
    """Process a refund for a given order."""
    order = ORDERS_DB.get(order_id)
    if order is None:
        return json.dumps({"error": f"Order {order_id} not found"})
    if order["status"] == "cancelled":
        return json.dumps({"error": f"Order {order_id} is already cancelled"})

    refund_record = {
        "refund_id": f"REF-{order_id}-{len(REFUND_LOG) + 1}",
        "order_id": order_id,
        "amount": order["amount"],
        "reason": reason,
        "status": "initiated",
        "estimated_completion": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
    }
    REFUND_LOG.append(refund_record)
    return json.dumps(refund_record)


# --- Tool Definitions (OpenAI function-calling schema) ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current status of a customer order by order ID. Returns order details including status, tracking, and delivery information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up (e.g., '12345')",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the company knowledge base for policy information on topics like refunds, returns, shipping, cancellations, and late deliveries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for the knowledge base (e.g., 'refund policy', 'late delivery')",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Initiate a refund for a customer order. Use only after verifying eligibility through order status and policy lookup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to refund",
                    },
                    "reason": {
                        "type": "string",
                        "description": "The reason for the refund (e.g., 'damaged product', 'late delivery')",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]

# Map function names to callables
TOOL_MAP = {
    "get_order_status": get_order_status,
    "search_kb": search_kb,
    "process_refund": process_refund,
}
