# -*- coding: cp1252 -*-
import pandas as pd
import html
from nltk import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Clean a tweet
def function_clean_tweet(text):
    # Escaping HTML characters
    text = html.unescape(text)
    # Removal of URLs
    text = re.sub(r"http\S+", "", text)
    # Removal of mentions
    text = re.sub("@[^\s]*", "", text)
    # Removal of hashtags
    text = re.sub("#[^\s]*", "", text)
    # Removal of numbers
    text = re.sub('[0-9]*[+-:]*[0-9]+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Apostrophe lookup
    text = re.sub("'ll", " will", text)
    text = re.sub("'ve", " have", text)
    text = re.sub("n't", " not", text)
    text = re.sub("'d", " would", text)
    text = re.sub("'re", " are", text)
    text = re.sub("i'm", "i am", text)
    text = re.sub("it's", "it is", text)
    text = re.sub("she's", "she is", text)
    text = re.sub("he's", "he is", text)
    text = re.sub("here's", "here is", text)
    text = re.sub("that's", "that is", text)
    text = re.sub("there's", "there is", text)
    text = re.sub("what's", "what is", text)
    text = re.sub("who's", "who is", text)
    text = re.sub("'s", "", text)
    # Handling slang words
    text = re.sub(r"\btmrw\b", "tomorrow", text)
    text = re.sub(r"\bur\b", "your", text)
    text = re.sub(r"\burs\b", "yours", text)
    text = re.sub(r"\bppl\b", "people", text)
    text = re.sub(r"\byrs\b", "years", text)
    # Handling acronyms
    text = re.sub(r"\b(rt)\b", "retweet", text)
    text = re.sub(r"\b(btw)\b", "by the way", text)
    text = re.sub(r"\b(asap)\b", "as soon as possible", text)
    text = re.sub(r"\b(fyi)\b", "for your information", text)
    text = re.sub(r"\b(tbt)\b", "throwback thursday", text)
    text = re.sub(r"\b(tba)\b", "to be announced", text)
    text = re.sub(r"\b(tbh)\b", "to be honest", text)
    text = re.sub(r"\b(faq)\b", "frequently asked questions", text)
    text = re.sub(r"\b(icymi)\b", "in case you missed it", text)
    text = re.sub(r"\b(aka)\b", "also known as", text)
    text = re.sub(r"\b(ama)\b", "ask me anything", text)
    # Word lemmatization
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# Read the CSV files into DataFrames and encode data
df_train = pd.read_csv("../input/train.csv", delimiter=',', encoding='latin-1')
df_test = pd.read_csv("../input/test.csv", delimiter=',', encoding='latin-1')

# Train set: print column types to understand the data
print("\nColumn types:")
print(df_train.dtypes)

# Train set: print sentiment values count to see if the classes are balanced
print("\nSentiment values count:")
print(df_train['Sentiment'].value_counts())

# Train and test sets: clean the tweets
df_train['SentimentText'] = df_train['SentimentText'].apply(lambda text: function_clean_tweet(text))
df_test['SentimentText'] = df_test['SentimentText'].apply(lambda text: function_clean_tweet(text))

# Train and test sets: print the tweets after cleaning
print("\nTrain set: tweets after cleaning:")
print(df_train['SentimentText'])
print("\nTest set: tweets after cleaning:")
print(df_test['SentimentText'])

# Create a vectorizer to convert tweets to a numeric matrix of TF-IDF features
vectorizer = TfidfVectorizer()

# Create and train a Classifier using the train set
clf = LogisticRegression()
model = clf.fit(vectorizer.fit_transform(df_train['SentimentText']), df_train['Sentiment'])

# Predictions on the test set
predictions = clf.predict(vectorizer.transform(df_test['SentimentText']))
print("\nPredictions on the test set:")
print(predictions)

# Prediction on an input string
input_string = "I'm very sad"
print("\nPrediction on an input string: " + input_string)
print(clf.predict(vectorizer.transform([function_clean_tweet(input_string)])))
