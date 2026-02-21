1import re
2import nltk
3from nltk.corpus import stopwords
4from nltk.stem import WordNetLemmatizer
5
6nltk.download('stopwords', quiet=True)
7nltk.download('wordnet', quiet=True)
8
9STOP_WORDS = set(stopwords.words('english'))
10LEMMATIZER = WordNetLemmatizer()
11
12def preprocess(text: str) -> str:
13    text = str(text).lower()
14    text = re.sub(r'[^a-zA-Z\s]', '', text)
15    words = text.split()
16    words = [LEMMATIZER.lemmatize(w) for w in words if w not in STOP_WORDS and len(w) > 2]
17    return " ".join(words)
18
19def clean_dataframe(df, text_col='Feedback'):
20    df = df[[text_col]].dropna().reset_index(drop=True)
21    df['clean_text'] = df[text_col].apply(preprocess)
22    return df
