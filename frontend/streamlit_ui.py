import streamlit as st
import pandas as pd
import requests

# API_URL = "http://localhost:8000/evaluate" # running on local
API_URL = "https://llm-backend-583275352336.us-central1.run.app/evaluate" # running on cloud

st.title("LLM Response Evaluator")
st.subheader("Evaluate LLM Responses for Toxicity, Bias, and Similarity")
st.write("### Enter Evaluation Inputs")

# Input fields
user_query = st.text_area("User Query", height=80)
expected_response = st.text_area("Expected Response", height=80)
actual_response = st.text_area("Actual Response", height=80)

if st.button("Evaluate"):
    if not user_query or not expected_response or not actual_response:
        st.warning("Please fill in all fields.")
    else:
        payload = [{
            "user_query": user_query,
            "expected_response": expected_response,
            "actual_response": actual_response
        }]

        try:
            with st.spinner("Evaluating..."):
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                results = response.json()
            
            st.success("Evaluation complete.")
            df = pd.DataFrame(results)
            st.write("### Results")
            st.dataframe(df)

        except requests.exceptions.RequestException as e:
            st.error(f"API call failed: {e}")