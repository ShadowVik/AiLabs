
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

def preprocess_review(review, max_length=500):
    word_to_id = imdb.get_word_index()
    tokens = [word_to_id.get(word, 0) for word in review.lower().split()]

    padded_tokens = pad_sequences([tokens], maxlen=max_length)

    return padded_tokens

def predict_sentiment(review, model):
    preprocessed_review = preprocess_review(review)
    prediction = model.predict(preprocessed_review)[0][0]

    sentiment = "positive" if prediction >= 0.5 else "negative"
    confidence = prediction if sentiment == "positive" else 1 - prediction

    return sentiment, confidence * 100

model = load_model('movie_review_model_final.h5')

user_review = input("Enter a movie review: ")

sentiment, confidence = predict_sentiment(user_review, model)
print(f"The review is {sentiment} with a {confidence:.2f}% confidence.")
