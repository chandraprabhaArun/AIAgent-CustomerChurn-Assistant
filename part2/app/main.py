"""TeleConnect Retention Agent — Streamlit App with visible tool call traces."""
import streamlit as st
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import OpenAI
from part2.tools import (
    lookup_customer, predict_churn, get_retention_offers,
    log_interaction, escalate_to_supervisor
)
from part2.agent import TOOLS, SYSTEM_PROMPT, TOOL_FUNCTIONS

# --- Page Config ---
st.set_page_config(
    page_title="TeleConnect Retention Agent",
    page_icon="📞",
    layout="wide"
)

st.title("📞 TeleConnect Retention Agent")
st.caption("AI-powered assistant for customer retention representatives. Tool calls are shown transparently.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    demo_mode = st.toggle("🎮 Demo Mode (no API key needed)", value=False)
    st.divider()
    st.header("📋 Sample Queries")
    st.markdown("""
    - `Pull up customer TC-004711`
    - `TC-000692 wants to cancel. What should I do?`
    - `What offers for high-risk month-to-month?`
    - `TC-003427 is threatening to sue us`
    - `I have an angry customer on the line` (no ID)
    """)
    st.divider()
    st.header("🔧 Available Tools")
    for tool in TOOLS:
        st.markdown(f"**{tool['function']['name']}** — {tool['function']['description'][:80]}...")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.tool_traces = []
        st.rerun()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_traces" not in st.session_state:
    st.session_state.tool_traces = []

# --- Display Chat History ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show tool traces for this assistant message
        if msg["role"] == "assistant" and i < len(st.session_state.tool_traces):
            traces = st.session_state.tool_traces[i]
            if traces:
                with st.expander(f"🔧 Tool Calls ({len(traces)} calls)", expanded=False):
                    for j, trace in enumerate(traces, 1):
                        st.markdown(f"**Step {j}: `{trace['tool']}`**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("*Input:*")
                            st.json(trace["arguments"])
                        with col2:
                            st.markdown("*Output:*")
                            st.json(trace["result"])
                        if j < len(traces):
                            st.divider()

# --- Demo Mode Logic ---
import re

def run_demo_mode(user_input: str) -> tuple:
    """Simulate agent behavior without API key for demonstration."""
    tool_call_trace = []
    response_text = ""

    # Extract customer ID if present
    id_match = re.search(r'TC-\d{6}', user_input)
    input_lower = user_input.lower()

    # Escalation triggers
    if any(w in input_lower for w in ['sue', 'legal', 'lawyer', 'threaten']):
        cid = id_match.group() if id_match else "TC-000000"
        result = escalate_to_supervisor(cid, "legal_threat", f"Customer threatening legal action: {user_input[:100]}")
        tool_call_trace.append({"tool": "escalate_to_supervisor", "arguments": {"customer_id": cid, "reason": "legal_threat", "context_summary": user_input[:100]}, "result": result})
        response_text = f"⚠️ **Case Escalated**\n\nI've immediately escalated this to a supervisor (Ticket: {result['ticket_id']}).\n\n**Next steps for you:**\n- Inform the customer that a specialist will follow up within 2 hours\n- Do not discuss billing details further\n- Document any specific claims they made"

    elif any(w in input_lower for w in ['scream', 'abus', 'profan', 'yelling']):
        cid = id_match.group() if id_match else "TC-000000"
        result = escalate_to_supervisor(cid, "abusive_language", f"Abusive customer: {user_input[:100]}", "high")
        tool_call_trace.append({"tool": "escalate_to_supervisor", "arguments": {"customer_id": cid, "reason": "abusive_language", "context_summary": user_input[:100], "urgency": "high"}, "result": result})
        response_text = f"⚠️ **Escalated (High Urgency)**\n\nTicket: {result['ticket_id']}\n\nYour safety comes first. A supervisor will take over shortly. You may end the call if you feel unsafe."

    elif id_match:
        cid = id_match.group()
        # Full workflow: lookup → predict → offers
        profile = lookup_customer(cid)
        tool_call_trace.append({"tool": "lookup_customer", "arguments": {"customer_id": cid}, "result": profile})

        if 'error' in profile:
            response_text = f"❌ Customer `{cid}` not found in our system. Please verify the ID and try again."
        else:
            features = {k: v for k, v in profile.items() if k not in ['customer_id', 'last_interaction_date']}
            prediction = predict_churn(features)
            tool_call_trace.append({"tool": "predict_churn", "arguments": {"customer_data": "<customer features>"}, "result": prediction})

            offers = get_retention_offers(prediction['risk_tier'], profile['contract_type'])
            tool_call_trace.append({"tool": "get_retention_offers", "arguments": {"risk_tier": prediction['risk_tier'], "contract_type": profile['contract_type']}, "result": offers})

            # Build recommendation
            offer_lines = "\n".join([f"  - **{o['offer_id']}**: {o['description']} ({o['value']})" for o in offers['offers']])
            response_text = f"""## 📋 Customer Analysis: {cid}

**Profile:** {profile['gender']}, age {profile['age']:.0f}, {profile['contract_type']} contract, {profile['tenure_months']:.0f} months tenure, satisfaction {profile['satisfaction_score']}/10

**Risk Assessment:** {prediction['risk_tier'].upper()} risk — {prediction['churn_probability']:.0%} churn probability

**Top Risk Factors:** {', '.join(prediction['top_risk_factors'])}

---

### 🎯 Recommended Offers:
{offer_lines}

---

### 💬 Suggested Talking Points:
1. Acknowledge their tenure and value as a customer
2. Ask what's driving their consideration to leave
3. Present the strongest offer ({offers['offers'][0]['offer_id']}) as a "loyalty appreciation"
4. If they hesitate, mention the second option as a fallback"""

    elif 'offer' in input_lower and ('high' in input_lower or 'medium' in input_lower or 'low' in input_lower):
        tier = 'high' if 'high' in input_lower else ('medium' if 'medium' in input_lower else 'low')
        contract = 'Month-to-month' if 'month' in input_lower else 'One year'
        offers = get_retention_offers(tier, contract)
        tool_call_trace.append({"tool": "get_retention_offers", "arguments": {"risk_tier": tier, "contract_type": contract}, "result": offers})
        offer_lines = "\n".join([f"- **{o['offer_id']}**: {o['description']} ({o['value']})" for o in offers['offers']])
        response_text = f"### Available Offers ({tier} risk, {contract}):\n\n{offer_lines}"

    else:
        response_text = "I'd be happy to help! Could you provide a **customer ID** (e.g., TC-004711) so I can look up their profile and run a churn analysis?\n\nOr you can ask me about:\n- Retention offers for a specific risk tier\n- Escalation for difficult situations\n- Customer lookup by ID"

    return response_text, tool_call_trace


# --- Chat Input ---
if user_input := st.chat_input("Ask about a customer or request retention help..."):
    if not api_key and not demo_mode:
        st.error("Please enter your OpenAI API key in the sidebar, or enable Demo Mode.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.tool_traces.append(None)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if demo_mode:
                # Demo mode — no API key needed
                response_content, tool_call_trace = run_demo_mode(user_input)
            else:
                # Live mode — uses OpenAI API
                client = OpenAI(api_key=api_key)
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in st.session_state.messages:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})

                tool_call_trace = []

                while True:
                    response = client.chat.completions.create(
                        model=model,
                        messages=api_messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                    message = response.choices[0].message
                    api_messages.append(message)

                    if not message.tool_calls:
                        break

                    for tool_call in message.tool_calls:
                        fn_name = tool_call.function.name
                        fn_args = json.loads(tool_call.function.arguments)

                        fn = TOOL_FUNCTIONS.get(fn_name)
                        result = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}

                        tool_call_trace.append({"tool": fn_name, "arguments": fn_args, "result": result})
                        api_messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result, default=str)})

                response_content = message.content

        # Display response
        st.markdown(response_content)

        # Display tool traces
        if tool_call_trace:
            with st.expander(f"🔧 Tool Calls ({len(tool_call_trace)} calls)", expanded=True):
                for j, trace in enumerate(tool_call_trace, 1):
                    st.markdown(f"**Step {j}: `{trace['tool']}`**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("*Input:*")
                        st.json(trace["arguments"])
                    with col2:
                        st.markdown("*Output:*")
                        st.json(trace["result"])
                    if j < len(tool_call_trace):
                        st.divider()

    # Save to session
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.session_state.tool_traces.append(tool_call_trace)
