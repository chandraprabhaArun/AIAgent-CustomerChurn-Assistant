"""Retention Agent — OpenAI tool-calling orchestration."""
import json
from openai import OpenAI
from part2.tools import (
    lookup_customer, predict_churn, get_retention_offers,
    log_interaction, escalate_to_supervisor
)

# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Retrieves a customer profile by ID. Returns demographics, contract type, tenure, charges, satisfaction score, and service details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID (e.g., 'TC-001234')"}
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn",
            "description": "Runs the churn prediction model on customer features. Returns churn_probability (0-1), risk_tier (high/medium/low), and top_risk_factors (list of 3 features).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_data": {
                        "type": "object",
                        "description": "Dictionary of customer features including: age, tenure_months, monthly_charges, total_charges, avg_monthly_gb_used, num_support_tickets, avg_monthly_minutes, satisfaction_score, num_additional_services, gender, contract_type, internet_service, phone_service, payment_method"
                    }
                },
                "required": ["customer_data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_retention_offers",
            "description": "Returns available retention offers filtered by customer risk tier and contract type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_tier": {"type": "string", "enum": ["high", "medium", "low"], "description": "Customer risk tier from churn prediction"},
                    "contract_type": {"type": "string", "description": "Customer's current contract type (Month-to-month, One year, Two year)"}
                },
                "required": ["risk_tier", "contract_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_interaction",
            "description": "Records the outcome of a retention conversation. Call after the interaction concludes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "agent_id": {"type": "string", "description": "ID of the retention representative"},
                    "outcome": {"type": "string", "enum": ["retained", "churned", "escalated", "callback_scheduled"]},
                    "offers_presented": {"type": "array", "items": {"type": "string"}, "description": "List of offer IDs presented"},
                    "offer_accepted": {"type": "string", "description": "Offer ID accepted, or null"},
                    "notes": {"type": "string"}
                },
                "required": ["customer_id", "agent_id", "outcome", "offers_presented"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_supervisor",
            "description": "Transfers the case to a human supervisor. Use when: customer threatens legal action, makes abusive statements, has a complex billing dispute, or the situation is outside the agent's capabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Why escalation is needed (e.g., 'legal_threat', 'complex_dispute', 'abusive_language', 'out_of_scope')"},
                    "context_summary": {"type": "string", "description": "Brief summary of the situation for the supervisor"},
                    "urgency": {"type": "string", "enum": ["normal", "high", "critical"], "default": "normal"}
                },
                "required": ["customer_id", "reason", "context_summary"]
            }
        }
    },
]

SYSTEM_PROMPT = """You are a retention assistant for TeleConnect, a telecommunications company. You help retention representatives handle at-risk customers.

Your workflow when a rep asks about a customer:
1. Look up the customer profile using their ID
2. Run the churn prediction model on their data
3. Retrieve appropriate retention offers based on risk tier and contract type
4. Synthesize a clear recommendation for the rep

Guidelines:
- Always look up the customer before making predictions
- If no customer ID is provided, ask for it
- For high-risk customers, be proactive with strong offers
- If a customer threatens legal action, is abusive, or has a complex dispute — escalate immediately
- Present recommendations in a clear, actionable format for the rep
- Never fabricate customer data — only use what the tools return
- After a successful interaction, log it

Response format for recommendations:
- Customer summary (1-2 lines)
- Risk assessment (probability + tier + key factors)
- Recommended offers (ranked by fit)
- Suggested talking points for the rep
"""

# Tool dispatch map
TOOL_FUNCTIONS = {
    "lookup_customer": lookup_customer,
    "predict_churn": predict_churn,
    "get_retention_offers": get_retention_offers,
    "log_interaction": log_interaction,
    "escalate_to_supervisor": escalate_to_supervisor,
}


def run_agent(user_message: str, client: OpenAI = None, model: str = "gpt-4o-mini",
              conversation_history: list = None) -> dict:
    """
    Run the retention agent. Returns:
    {
        "response": str,           # Final agent response
        "tool_calls": list,        # Ordered list of tool calls made
        "conversation": list       # Full message history
    }
    """
    if client is None:
        client = OpenAI()

    if conversation_history is None:
        conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversation_history.append({"role": "user", "content": user_message})
    tool_call_trace = []

    # Agent loop — keep calling until no more tool calls
    while True:
        response = client.chat.completions.create(
            model=model,
            messages=conversation_history,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        conversation_history.append(message)

        # If no tool calls, we're done
        if not message.tool_calls:
            break

        # Execute each tool call
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            # Dispatch
            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            # Record trace
            tool_call_trace.append({
                "tool": fn_name,
                "arguments": fn_args,
                "result": result,
            })

            # Append tool result to conversation
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    return {
        "response": message.content,
        "tool_calls": tool_call_trace,
        "conversation": conversation_history,
    }
