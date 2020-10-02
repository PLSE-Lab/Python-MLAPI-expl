#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install contractions


# In[ ]:


# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")


# In[ ]:


# Loading the data
path = "/kaggle/input/amazon-fine-food-reviews/Reviews.csv"
reviews = pd.read_csv(path)
reviews.head()


# In[ ]:


# Visualizing the score
sns.countplot(x=reviews["Score"])


# In[ ]:


# Removing rows with score of 3
filtered_reviews = reviews.loc[reviews["Score"].isin([1, 2, 4, 5])]

def category(x):
    if x > 3:
        return 1
    else:
        return 0
    
ratings = filtered_reviews["Score"].map(category)
filtered_reviews["Score"] = ratings

filtered_reviews.head()


# In[ ]:


# Data cleaning
sorted_reviews = filtered_reviews.sort_values(by="ProductId", axis=0, ascending=True)
final_reviews = sorted_reviews.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep="first", inplace=False)
final_reviews.shape


# In[ ]:


# % of data left
(final_reviews.size*1.0 / filtered_reviews.size*1.0) * 100


# In[ ]:


# Removing incorrect entries
final_reviews = final_reviews[final_reviews.HelpfulnessNumerator <= final_reviews.HelpfulnessDenominator]
final_reviews.shape


# In[ ]:


# Value counts
final_reviews["Score"].value_counts()


# In[ ]:


# Making a list of stopwords
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")

print(stop_words)


# In[ ]:


# Text preprocessing
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import contractions
preprocessed_reviews = []

for sentence in tqdm(final_reviews['Text'].values):
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = contractions.fix(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = sentence.lower()
    sentence = stemmer.stem(sentence)
    sentence = " ".join([word for word in sentence.split() if word not in stop_words])
    preprocessed_reviews.append(sentence)


# In[ ]:


# Appending the cleaned text to the dataset
final_reviews["Cleaned Text"] = preprocessed_reviews
final_reviews.head()


# In[ ]:


# Creating vocubulary
corpus = []

for text in tqdm(final_reviews["Cleaned Text"]):
    for word in text.strip().split():
        corpus.append(word.strip())
    
print(len(corpus))


# In[ ]:


# Word count
from collections import Counter
word_count = Counter(corpus)
print("Words =", len(word_count))
word_count.most_common(5)


# In[ ]:


# Creating word count dataframe
word_count_df = []

for idx, (word, count) in enumerate(word_count.most_common(len(word_count))):
    word_count_df.append([idx+1, word, count])

word_count_df = pd.DataFrame(columns=["Index", "Word", "Count"], data=word_count_df)    
word_count_df.head()


# In[ ]:


# Creating word count dictionary
word_count_dict = {}

for _, row in word_count_df.iterrows():
    word_count_dict[row["Word"]] = [row["Index"], row["Count"]]

print("#Keys =", len(word_count_dict.keys()))


# In[ ]:


# Data preprocessing
indexed_X = []
indexed_y = []

for sentence in final_reviews["Cleaned Text"]:
    indexed_X.append([word_count_dict[word][0] for word in sentence.strip().split()])

indexed_y = final_reviews["Score"]


# In[ ]:


# Data loader
from sklearn.model_selection import train_test_split

def load_data(num_words, tst_size):
    X = indexed_X
    y = indexed_y
    
    for i in range(len(X)):
        for j in range(len(X[i])):
            if X[i][j] > num_words:
                X[i][j] = 0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_size)
    return (X_train, y_train), (X_test, y_test)


# In[ ]:


# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[ ]:


top_words = 5000
t_size = 0.33
(X_train, y_train), (X_test, y_test) = load_data(num_words=top_words, tst_size=t_size)


# In[ ]:


# Truncate and/or pad input sequences
max_review_length = 100

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

print(X_train.shape)
print(X_train[1])


# In[ ]:


# Build the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


# Final evaluation of the model
model.fit(X_train, y_train, nb_epoch=10, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




