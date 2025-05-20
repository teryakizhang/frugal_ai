import nltk
from nltk.stem import WordNetLemmatizer
import contractions
import re



def preprocess(X):
    lemmatizer  = WordNetLemmatizer()
    preprocessed_texts = []
    for doc in X:
        # Expand contractions
        expanded = contractions.fix(doc)
        # Remove special characters
        # expanded = re.sub(r"[^a-zA-Z0-9]", " ", expanded)
        # Lowercase
        lowered = expanded.lower()
        # Tokenize and lemmatize
        lemmatized = " ".join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(lowered)])
        preprocessed_texts.append(lemmatized)
    return preprocessed_texts