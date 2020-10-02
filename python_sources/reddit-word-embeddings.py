import sqlite3
import pandas as pd
import gensim
import unicodedata
from keras.preprocessing import text

sql_conn = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql("SELECT body FROM May2015 LIMIT 5;", sql_conn)

print("Tokenization Started")
tokenizer = text.Tokenizer(num_words=500)
print("Fitting..")
tokenizer.fit_on_texts(list(df['body']))
print("Transforming..")
tokenized_text = tokenizer.texts_to_sequences(df['body'])
print("Tokenization Done")
print(df['body'])

def unicodeToAscii(series):
    return series.apply(lambda s: unicodedata.normalize('NFKC', str(s)))

def normalizeString(series):
    series = unicodeToAscii(series)
    series = series.lower()
    series = series.replace(r"(\n){1,}", " ")
    series = series.replace(r"\'", "")
    series = series.replace(r"\-", "")
    series = series.replace(r"[^0-9a-z]+", " ")
    series = series.replace("([a-z0-9]{2,}\.){2,}[a-z]{2,}", " ") 
    series = series.replace(" \d ", "")
    return series

count = 0 
for series in df['body']:
    df['body'][count] = normalizeString(series)
    count = count + 1

print("Training Started..")
#model = gensim.models.Word2Vec(tokenized_text, min_count = 0, size = 300)
