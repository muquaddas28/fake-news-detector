# Fake News Detection System

A machine learning-based system for detecting fake news articles using Python and Streamlit.

## Project Structure

```
fake_news/
├── data/               # Data files
│   ├── True.csv       # Real news dataset
│   ├── Fake.csv       # Fake news dataset
│   └── combined_news.csv  # Combined dataset
├── models/            # Trained models
│   ├── fake_news_model.pkl  # Main classification model
│   ├── vectorizer.pkl      # TF-IDF vectorizer
│   └── tfidf_matrix.pkl    # TF-IDF matrix
└── src/              # Source code
    └── app.py        # Streamlit application
```

## Setup and Installation

1. Make sure you have Python 3.7+ installed
2. Install required packages:
   ```bash
   pip install streamlit joblib scikit-learn numpy pandas
   ```

## Running the Application

1. Navigate to the project root directory
2. Run the Streamlit app:
   ```bash
   streamlit run src/app.py
   ```
3. The application will open in your default web browser

## Features

- Real-time fake news detection
- Confidence score for predictions
- User-friendly interface
- Trained on 2016-2017 news data
- Visual confidence indicators

## Note

The model was trained on news articles from 2016-2017. The accuracy may vary for more recent news or different writing styles. 