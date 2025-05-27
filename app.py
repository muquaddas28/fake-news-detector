import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page config
st.set_page_config(
    page_title="Fake News Detection",  # Changed for browser tab
    page_icon="📰",
    layout="wide"
)

model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove content inside brackets [like this]
    text = re.sub(r'\[[^]]*\]', '', text)
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back to string
    return ' '.join(tokens)



st.title("📰 Fake News Detection App")

user_input = st.text_area("Paste any news content here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.success("✅ This news is **REAL**.")
        else:
            st.error("🚨 This news is **FAKE**.")
