import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import io
import unicodedata
import numpy as np
import re
import string
from numpy import linalg
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def read_file(file_path):
    """Read the content of a text file."""
    with open(file_path, encoding='ISO-8859-2') as f:
        return f.read()

def tokenize_text(text):
    """Tokenize text into words and sentences."""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return words, sentences

def stem_words(words):
    """Stem words using Porter Stemmer."""
    porter_stemmer = PorterStemmer()
    return [(word, porter_stemmer.stem(word)) for word in words]

def lemmatize_words(words):
    """Lemmatize words using WordNet Lemmatizer."""
    wordnet_lemmatizer = WordNetLemmatizer()
    return [(word, wordnet_lemmatizer.lemmatize(word)) for word in words]

def pos_tagging(words):
    """Perform part-of-speech tagging."""
    return nltk.pos_tag(words)

def sentiment_analysis(sentences):
    """Perform sentiment analysis on sentences."""
    sid = SentimentIntensityAnalyzer()
    results = {}
    for sentence in sentences:
        scores = sid.polarity_scores(sentence)
        results[sentence] = scores
    return results

def main():
    # Read the text file
    text = read_file('kindle.txt')

    # Tokenize the text
    words, sentences = tokenize_text(text)

    # Stemming
    stemmed_words = stem_words(words)
    print("Stemming Results:")
    for original, stemmed in stemmed_words:
        print(f"Actual: {original} Stem: {stemmed}")

    # Lemmatization
    lemmatized_words = lemmatize_words(words)
    print("\nLemmatization Results:")
    for original, lemma in lemmatized_words:
        print(f"Actual: {original} Lemma: {lemma}")

    # Part-of-Speech Tagging
    pos_tags = pos_tagging(words)
    print("\nPart-of-Speech Tagging Results:")
    for word, tag in pos_tags:
        print(f"{word}: {tag}")

    # Sentiment Analysis
    sentiment_results = sentiment_analysis(sentences)
    print("\nSentiment Analysis Results:")
    for sentence, scores in sentiment_results.items():
        print(f"Sentence: {sentence}")
        for key in sorted(scores):
            print(f'{key}: {scores[key]}', end=', ')
        print()  # New line for better readability

if __name__ == "__main__":
    main()