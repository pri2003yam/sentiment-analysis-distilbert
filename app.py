import streamlit as st
from inference import SentimentClassifier

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")

@st.cache_resource
def load_classifier():
    return SentimentClassifier()

classifier = load_classifier()

st.title("Sentiment Analysis with DistilBERT")
st.write("Enter text below and see the predicted sentiment in real time.")

# Text input
user_input = st.text_area("Your text", height=150)

# Predict button
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running model..."):
            result = classifier.predict(user_input)

        label = result["label"]
        score = result["score"]

        st.subheader("Prediction")
        st.write(f"**Sentiment:** {label.capitalize()}")
        st.write(f"**Confidence:** {score:.4f}")

# Footer
st.markdown("---")
st.caption("Built with HuggingFace Transformers + Streamlit")
