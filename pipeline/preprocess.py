import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("hospital.csv")

df = df[['Feedback']].dropna()


stop_words = set(stopwords.words('english'))lemmatizer = WordNetLemmatizer()def preprocess(text):    text = text.lower()    text = re.sub(r'[^a-zA-Z\s]', '', text)    words = text.split()    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]    return " ".join(words)df['clean_text'] = df['Feedback'].apply(preprocess)