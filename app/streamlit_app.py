import streamlit as st
from recommender.ranker import recommend
from recommender.explain import explain_recommendation

st.set_page_config(
    page_title="FiReco AI Assistant",
    page_icon="💳",
    layout="wide"
)

st.title("💳 FiReco AI Assistant")
st.markdown(
    "AI-powered financial product recommendation platform"
)

user_text = st.text_input(
    "Describe your ideal credit card"
)

preferred_category = st.selectbox(
    "Preferred Category",
    [
        "all",
        "travel",
        "shopping",
        "fuel",
        "business",
        "lifestyle",
        "rewards",
        "cashback",
    ]
)

max_fee = st.number_input(
    "Maximum Annual Fee (₹)",
    min_value=0,
    value=5000,
    step=500
)

top_k = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Get Recommendations"):

    try:

        with st.spinner("Finding recommendations..."):

            results = recommend(
                goal_text=user_text,
                preferred_category=preferred_category,
                max_annual_fee=max_fee,
                top_k=top_k
            )

        st.subheader("Top Recommendations")

        if results.empty:
            st.warning("No recommendations found.")
        else:

            for idx, row in results.iterrows():

                with st.container(border=True):

                    st.markdown(
                        f"## #{idx + 1} - {row.get('product_name', 'Unknown Product')}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            f"**Bank:** {row.get('bank_name', 'N/A')}"
                        )

                        st.write(
                            f"**Category:** {row.get('primary_usage_category', 'N/A')}"
                        )

                        st.write(
                            f"**Reward Type:** {row.get('reward_type', 'N/A')}"
                        )

                        st.write(
                            f"**Speciality:** {row.get('speciality', 'N/A')}"
                        )

                    with col2:

                        st.metric(
                            "Match Score",
                            f"{row.get('score', 0):.3f}"
                        )

                        st.write(
                            f"**Annual Fee:** ₹{float(row.get('annual_fee', 0)):.0f}"
                        )

                    explanation = explain_recommendation(
                        row.to_dict(),
                        user_text
                    )

                    st.success(explanation)

    except Exception as e:
        st.error(f"Error: {str(e)}")