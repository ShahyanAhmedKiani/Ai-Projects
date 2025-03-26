import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load pre-trained model and vectorizer
model = joblib.load("APP/sentiment_model.pkl")
vectorizer = joblib.load("APP/vectorizer.pkl")

nltk.download('stopwords')
nltk.download('punkt')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Streamlit UI
st.title("Movie Review Sentiment Analysis")

user_input = st.text_area("Enter a movie review:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.error("Please enter a movie review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            processed_text = preprocess_text(user_input)
            text_vector = vectorizer.transform([processed_text])
            prediction = model.predict(text_vector)[0]
            
            # Print the raw prediction output for debugging
            print("Raw prediction output:", prediction)
            
            if prediction == "positive":
                st.success("Sentimental: Positive 😊")
            else:
                st.warning("Sentimental: Negative 😐")
