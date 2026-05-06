import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(
    page_title="FiReco AI Assistant",
    page_icon="💳",
    layout="wide"
)

st.title("💳 FiReco AI Assistant")

user_text = st.text_input(
    "Describe your ideal credit card"
)

if st.button("Get Recommendations"):

    payload = {
        "goal": user_text,
        "preferred_category": "all",
        "max_annual_fee": None,
        "top_k": 5,
        "use_ai_summary": True,
    }

    try:

        with st.spinner("Finding recommendations..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

        if data.get("ai_summary"):
            st.subheader("AI Summary")
            st.info(data["ai_summary"])

        st.subheader("Top Recommendations")

        recommendations = data.get(
            "recommendations",
            []
        )

        for rec in recommendations:

            with st.container(border=True):

                st.markdown(
                    f"## #{rec['rank']} - {rec['product_name']}"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"**Bank:** {rec['bank_name']}"
                    )

                    st.write(
                        f"**Category:** {rec['primary_usage_category']}"
                    )

                    st.write(
                        f"**Reward Type:** {rec['reward_type']}"
                    )

                    st.write(
                        f"**Speciality:** {rec['speciality']}"
                    )

                with col2:
                    st.metric(
                        "Match Score",
                        f"{rec['score']:.3f}"
                    )

                    st.write(
                        f"**Annual Fee:** ₹{rec['annual_fee']:.0f}"
                    )

                st.success(rec["reason"])

    except Exception as e:
        st.error(str(e))