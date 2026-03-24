"""
Customer Support Agent built with Microsoft Agent Framework.
Uses AzureAIClient with tool-calling for order lookup, KB search, and refund processing.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv(override=False)

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tools import TOOL_DEFINITIONS, TOOL_MAP

SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce company.

Your capabilities:
1. Look up order status using the get_order_status tool
2. Search the company knowledge base for policies using the search_kb tool
3. Process refunds using the process_refund tool (only after verifying eligibility)

Guidelines:
- Always look up the order status before answering questions about an order
- Always check the knowledge base for relevant policies before making claims about policies
- Be empathetic and professional
- Provide specific details (tracking numbers, dates, amounts) when available
- Never make up information - only use data from tool results
- If an order is not found, ask the customer to verify the order ID
- When offering options, clearly explain each one and ask the customer's preference
- For refunds, always verify eligibility through order status and policy before processing
- For undelivered or late orders: FIRST check the late delivery policy before discussing refunds.
  If the order is past its estimated delivery date, present late delivery remedies (credit, reshipment)
  as the primary options. Only discuss full refunds if the customer specifically insists after hearing
  the late delivery options, and then also look up the refund policy.
"""


def create_client() -> AzureOpenAI:
    """Create an Azure OpenAI client using Entra ID (Azure AD) authentication."""
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


def run_agent(query: str, client: AzureOpenAI | None = None) -> dict:
    """
    Run the customer support agent on a single query.

    Returns a dict with:
      - response: the final text answer
      - tool_calls: list of tool calls made (name + args + result)
      - messages: full conversation history
    """
    if client is None:
        client = create_client()

    model = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    tool_call_log = []
    max_iterations = 10  # safety limit to prevent infinite tool-calling loops

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        choice = response.choices[0]

        # If the model wants to call tools
        if choice.message.tool_calls:
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                # Execute the tool
                tool_fn = TOOL_MAP.get(fn_name)
                if tool_fn:
                    result = tool_fn(**fn_args)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                tool_call_log.append({
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": json.loads(result),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # If the model produced a final response (no more tool calls)
        else:
            final_response = choice.message.content or ""
            messages.append({"role": "assistant", "content": final_response})

            return {
                "response": final_response,
                "tool_calls": tool_call_log,
                "messages": messages,
            }

    # If we hit the iteration limit
    return {
        "response": "[Agent reached maximum tool-calling iterations]",
        "tool_calls": tool_call_log,
        "messages": messages,
    }


if __name__ == "__main__":
    # Quick interactive test
    print("Customer Support Agent (type 'quit' to exit)")
    print("=" * 50)
    client = create_client()
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        result = run_agent(query, client)
        print(f"\nAgent: {result['response']}")
        if result["tool_calls"]:
            print(f"\n  [Tools used: {', '.join(tc['tool'] for tc in result['tool_calls'])}]")
