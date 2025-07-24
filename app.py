import streamlit as st
import joblib
from utils import preprocess_text

model = joblib.load('sentiment_model.pkl')


st.title("Customer Review Analysis System")
st.write("Enter a customer review to classify it as Positive, Negative, or Neutral.")

review = st.text_area("Customer Review")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = preprocess_text(review)
        prediction = model.predict([cleaned])[0]
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")
