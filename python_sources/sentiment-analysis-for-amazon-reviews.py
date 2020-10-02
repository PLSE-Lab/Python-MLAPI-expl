#!/usr/bin/env python
# coding: utf-8

# # **Sentiment Analysis for Amazon Reviews**
# 
# In this project, we study the correlation between Amazon's product reviews and the rating of the products given by the customers. 
# 
# We use Natural Language Processing(NLP) to predict whether the sentiment of review is positive or average or negative.
# 
# Both traditional machine learning algorithms including Bernoulli Naive Bayes,  Multinomial Naive Bayesian(MNB), Logistic regression method and deep neural networks using such as Long Short Term Memory (LSTM) is used. 
# 
# We also use gloVe input features with dropout to obtain results.
# 
# By comparing these results, we get a good understanding of the these algorithms.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


# Read the dataset
# 

# In[ ]:


data = pd.read_csv("/kaggle/input/consumer-reviews-of-amazon-products/1429_1.csv")


# In[ ]:


data.head()


# In[ ]:


data = data[['reviews.rating' , 'reviews.text']]
data=data.dropna()
data.head()


# In[ ]:


counts = data['reviews.rating'].value_counts()
plt.bar(counts.index, counts.values)
plt.show()


# Due to the imbalance of our dataset, we try to add in more datas to reduce overfitting.
# 
# We add files such that we include rows with reviews lesser than 4 inorder to balance the dataset.

# In[ ]:


data2 = pd.read_csv("/kaggle/input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
data2 = data2[['reviews.rating' , 'reviews.text']]
data2 = data2[data2["reviews.rating"]<=3]

data3 = pd.read_csv("/kaggle/input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
data3 = data3[['reviews.rating' , 'reviews.text']]
data3 = data3[data3["reviews.rating"]<=3]

frames = [data, data2, data3]
df = pd.concat(frames)
df = df.dropna()


# # We have 3 classifications.
# 1. BAD (0) for star rating 1 and 2.
# 2. AVERAGE (1) for star rating 3.
# 3. EXCELLENT (2) for star rating 4 and 5.

# In[ ]:


sentiment = {1: 0,
            2: 0,
            3: 1,
            4: 2,
            5: 2}

df["sentiment"] = df["reviews.rating"].map(sentiment)

#print(df[df["sentiment"].isnull()])
df["sentiment"] = pd.to_numeric(df["sentiment"], errors='coerce')                                    
df = df.dropna(subset=["sentiment"])
df["sentiment"]  = df["sentiment"] .astype(int)


# # Preprocessing the data
# 
# Applying various NLP techniques - tokenize and remove all the puncuations and uneseccary jargons.

# In[ ]:


df["reviews.text"]=df["reviews.text"].apply(lambda elem: re.sub("[^a-zA-Z]", " ", str(elem)))
df["reviews.text"]=df["reviews.text"].str.lower()
#tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = df["reviews.text"].str.split()

stopword_list = stopwords.words('english')
ps = PorterStemmer()
words_descriptions = words_descriptions.apply(lambda elem: [word for word in elem if not word in stopword_list])
words_descriptions = words_descriptions.apply(lambda elem: [ps.stem(word) for word in elem])

df['cleaned'] = words_descriptions.apply(lambda elem: ' '.join(elem))
df['cleaned'].head()


# Vectorizing the array.

# In[ ]:


vectorizer =TfidfVectorizer()
text = vectorizer.fit_transform(df['cleaned']).toarray()
texts=pd.DataFrame(text)


# Splitting the data into training and testing set.

# In[ ]:


y=df["sentiment"].values
X=pd.DataFrame(texts)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
lr.fit(X_train,y_train)


# In[ ]:


print('Train accuracy :', (lr.score(X_train, y_train))*100)
print('Test accuracy :', (lr.score(X_test, y_test))*100)
      
print('\n CONFUSION MATRIX')
print(confusion_matrix(y_test, lr.predict(X_test)))
print('\nCLASSIFICATION REPORT')
print(classification_report(y_test, lr.predict(X_test)))


# # Multinomial Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)


# In[ ]:


print('Train accuracy :', (nb.score(X_train, y_train))*100)
print('Test accuracy :', (nb.score(X_test, y_test))*100)
      
print('\n CONFUSION MATRIX')
print(confusion_matrix(y_test, nb.predict(X_test)))
print('\nCLASSIFICATION REPORT')
print(classification_report(y_test, nb.predict(X_test)))


# # Bernoulli Naive Bayes

# In[ ]:


from sklearn.naive_bayes import BernoulliNB
dt = BernoulliNB()
dt.fit(X_train,y_train)


# In[ ]:


print('Train accuracy :', (dt.score(X_train, y_train))*100)
print('Test accuracy :', (dt.score(X_test, y_test))*100)
      
print('\n CONFUSION MATRIX')
print(confusion_matrix(y_test, dt.predict(X_test)))
print('\nCLASSIFICATION REPORT')
print(classification_report(y_test, dt.predict(X_test)))


# # Using RNN - Long Short Term Memory(LSTM)

# In[ ]:


from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, GRU, Dropout, LSTM, Bidirectional
from keras.layers import GlobalMaxPooling1D


# Tokenizer is done to vectorize a text corpus, by turning each text into either a sequence of integers or into a vector.
# And then we split it to train and test sets.

# In[ ]:


token = Tokenizer()
token.fit_on_texts(df["reviews.text"])
word_index = token.word_index
max_len = 120
X_train, X_test, y_train, y_test = train_test_split(df["reviews.text"], df["sentiment"], test_size=0.25, random_state=42)


# 1.Converting Text to Sequence
# 
# 2.Padding to ensure that all sequences in a list have the same length.

# In[ ]:


X_train = token.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_len, padding = "post",truncating = "post")

X_test = token.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_len, padding = "post", truncating = "post")

y_train = np_utils.to_categorical(y_train, num_classes=3)
y_test = np_utils.to_categorical(y_test, num_classes=3)

len(y_test),len(X_test),len(X_train),len(y_train)


# **Model**

# In[ ]:


vocab_size = len(word_index)+1
embedding_dim = 16
optimizer = Adam(lr=0.0001, decay=0.0001)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim , input_length=max_len))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150, return_sequences=False))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=16, epochs=8)


# In[ ]:


result = model.evaluate(X_train, y_train)
print( 'Train accuracy :' , result[1]*100)
result = model.evaluate(X_test,y_test)
print( 'Test accuracy :' , result[1]*100)


# # Using GloVe
# 
# GloVe, coined from Global Vectors, is a model for distributed word representation. The model is an unsupervised learning algorithm for obtaining vector representations for words. This is achieved by mapping words into a meaningful space where the distance between words is related to semantic similarity

# In[ ]:


f = open('/kaggle/input/glove6b50dtxt/glove.6B.50d.txt',encoding="utf8")
embidx = {}
for line in f:
    val = line.split()
    word = val[0]
    coeff = np.asarray(val[1:],dtype = 'float')
    embidx[word] = coeff

f.close()

print('Found %s word vectors.' % len(embidx))


# In[ ]:


vocab_size=len(word_index)
embedding_dim = 50

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));

for word, i in word_index.items():
    embedding_vector = embidx.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;

print(len(embeddings_matrix))


# In[ ]:


embeddings_matrix.shape


# In[ ]:


embedd_layer = Embedding(vocab_size+1, embedding_dim, input_length=max_len, 
                         weights=[embeddings_matrix], trainable=False)

model = Sequential()
model.add(embedd_layer)
model.add(Bidirectional(LSTM(64 , return_sequences = True , dropout = 0.1 , recurrent_dropout = 0.1)))
model.add(GlobalMaxPooling1D())
model.add(Dense(150,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation="softmax"))
model.compile(loss = 'binary_crossentropy' , optimizer = Adam(lr = 0.01) , metrics = ['accuracy'])
model.summary()


# In[ ]:


hist = model.fit(X_train,y_train,epochs = 10 , batch_size = 512, validation_data = (X_test,y_test))


# In[ ]:


result = model.evaluate(X_test,y_test)
print('Test accuracy :', result[1]*100)


# # Conclusion
# 
# We extracted the features of our dataset and built several supervised model based on that. 
# These models not only include traditional algorithms such as naive bayes, logistic regression but also deep learning models such as Recurrent Neural Networks and Convolutional Neural networks. 
# 
# From the results, our highest accuracy on the test set is 91.3% when using LSTM with GloVe and dropout. 
# 
# One of the main reason the accuracy is not high enough is because of the data imbalance. 
