
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

st.title("Multi-Label Emotion Recognition")

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28, problem_type="multi_label_classification")
    return tokenizer, model

tokenizer, model = load_model()

label_map = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0]
    threshold = 0.5
    predicted = [i for i, p in enumerate(probs) if p > threshold]
    return predicted

user_input = st.text_area("Enter your text")
if st.button("Analyze"):
    preds = predict_emotions(user_input)
    st.write("Predicted emotions:")
    st.success(", ".join([label_map[i] for i in preds]) if preds else "None")
