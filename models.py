import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        # Lowercase
        text = text.lower()
        # Remove content inside brackets [like this]
        text = re.sub(r'\[[^]]*\]', '', text)
        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        # Join tokens back to string
        return ' '.join(tokens) 