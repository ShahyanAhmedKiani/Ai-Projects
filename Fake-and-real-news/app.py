import streamlit as st
import joblib


model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Enter a news article below to check if it's **real** or **fake**")

# User input text area
user_input = st.text_area("Paste the news article here...", height=200)

if st.button("Predict"):
    if user_input.strip():
        # Transform input and predict
        transformed_text = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(transformed_text)[0]
        print(prediction)
        if prediction == 0:
            result="🛑 Fake News" 

        else:
            result= "✅ Real News"

        # Display result
        st.markdown(f"### **Prediction:** {result}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.sidebar.markdown("### About")
st.sidebar.info("This application uses a trained **Naïve Bayes model** to classify news articles as real or fake.")


