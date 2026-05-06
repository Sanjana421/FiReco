import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="FiReco AI Assistant", page_icon="💳", layout="wide")
st.title("FiReco AI Assistant")
st.write("Describe what you want. I’ll ask a follow-up only if needed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False

if "clarifying_questions" not in st.session_state:
    st.session_state.clarifying_questions = []

if "initial_goal" not in st.session_state:
    st.session_state.initial_goal = ""

if "clarification_answers" not in st.session_state:
    st.session_state.clarification_answers = {}

if "final_result" not in st.session_state:
    st.session_state.final_result = None

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Initial free-text input OR follow-up prompt
user_text = st.chat_input("Describe what you want...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.initial_goal = user_text

    payload = {
        "goal": user_text,
        "preferred_category": "all",
        "max_annual_fee": 2000,
        "top_k": 5,
        "use_ai_summary": True
    }

    response = requests.post(API_URL, json=payload, timeout=60)
    data = response.json()

    if data.get("needs_clarification"):
        st.session_state.awaiting_clarification = True
        st.session_state.clarifying_questions = data.get("clarifying_questions", [])
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I need one quick clarification before I recommend anything."
        })
        st.rerun()
    else:
        st.session_state.final_result = data
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("ai_summary", "Here are your recommendations.")
        })
        st.rerun()

# Clarification UI
if st.session_state.awaiting_clarification and st.session_state.clarifying_questions:
    q = st.session_state.clarifying_questions[0]
    st.subheader(q["question"])

    choice = st.radio("Choose one", q["options"], key="clarify_choice")

    if st.button("Continue"):
        st.session_state.clarification_answers[q["question"]] = choice

        final_payload = {
            "goal": st.session_state.initial_goal,
            "clarifications": st.session_state.clarification_answers,
            "preferred_category": choice if choice in ["cashback", "travel", "shopping", "fuel", "student", "premium"] else "all",
            "max_annual_fee": 2000,
            "top_k": 5,
            "use_ai_summary": True
        }

        response = requests.post(API_URL, json=final_payload, timeout=60)
        data = response.json()

        st.session_state.final_result = data
        st.session_state.awaiting_clarification = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("ai_summary", "Here are your recommendations.")
        })
        st.rerun()

# Show final results
if st.session_state.final_result:
    data = st.session_state.final_result

    if data.get("ai_summary"):
        st.chat_message("assistant").write(data["ai_summary"])

    st.subheader("Recommendations")
    for rec in data.get("recommendations", []):
        with st.container(border=True):
            st.markdown(f"### {rec['product_name']}")
            st.write(f"**Bank:** {rec['bank_name']}")
            st.write(f"**Category:** {rec['primary_usage_category']}")
            st.write(f"**Reason:** {rec['reason']}")
            st.metric("Score", f"{rec['score']:.3f}")