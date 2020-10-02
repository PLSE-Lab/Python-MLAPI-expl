#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis is performed in the first part of this notebook.
# ## Later RNN is used for classification of reviews.

# In[ ]:


# Load Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.tokenize import RegexpTokenizer
import statsmodels.api as sm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import re
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SimpleRNN

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv", index_col=0)
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # EDA - What can you Explore?
# - what age gives what type of rating?
# - What are recommended in each Division, Class, department of Clothes?
# - Which age group gives more comments/ratings on what type of clothes?
# - Rating vs Positive feedback count
# - Lengthy Reviews for what type of cloth?
# - Positive/Negative Reviews for what type of clothes?
# 

# ## What age group has given what types of Ratings?

# In[ ]:


# The age distribution in data
plt.hist(df['Age'], color="green", label = "Age")
plt.legend()
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution in Data")


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x = 'Rating', y = 'Age', data = df)


# ## What are Recommended Clothes item?

# In[ ]:


print(df['Division Name'].unique())
print(df['Department Name'].unique())
print(df['Class Name'].unique())


# In[ ]:


rd = df[df['Recommended IND'] == 1] # recommended
nrd = df[df['Recommended IND'] == 0] # not recommended
rd.head()


# In[ ]:


plt.style.use('ggplot')

fig = plt.figure(figsize=(18, 18))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = plt.xticks(rotation=45)
ax1 = plt.hist(rd['Division Name'], color = "red", alpha = 0.5, label = "Recommended")
ax1 = plt.hist(nrd['Division Name'], color = "blue", alpha = 0.5, label = "Not Recommended")
ax1 = plt.title("Recommended Items in each Division")
ax1 = plt.legend()

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = plt.xticks(rotation=45)
ax2 = plt.hist(rd['Department Name'], color="green", alpha = 0.5, label = "Recommended")
ax2 = plt.hist(nrd['Department Name'], color="yellow", alpha = 0.5, label = "Not Recommended")
ax2 = plt.title("Recommended Items in each Department")
ax2 = plt.legend()

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3 = plt.xticks(rotation=45)
ax3 = plt.hist(rd['Class Name'], color="blue", alpha = 0.5, label = "Recommended")
ax3 = plt.hist(nrd['Class Name'], color="cyan", alpha = 0.5, label = "Not Recommended")
ax3 = plt.title("Recommended Items in each Class")
ax3 = plt.legend()


# # Which age group gives what length of comments on what type of clothes?

# In[ ]:


df['Review Length'] = df['Review Text'].astype(str).apply(len)
df.head()


# In[ ]:


fig = plt.figure(figsize=(10, 5))
#ax1 = plt.hist(df['Review Length'], color = "red", bins = 20)
ax = sns.distplot(df['Review Length'], color="blue")
ax = plt.title("Length of Reviews")


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x = 'Age', y = 'Review Length', data = df)


# In[ ]:


plt.style.use('ggplot')

fig = plt.figure(figsize=(18, 18))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = plt.xticks(rotation=45)
ax1 = sns.boxplot(x = 'Division Name', y = 'Review Length', data = df)
ax1 = plt.title("Review Length in each Division")

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = plt.xticks(rotation=45)
ax2 = sns.boxplot(x = 'Department Name', y = 'Review Length', data = df)
ax2 = plt.title("Review Length in each Department")

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3 = plt.xticks(rotation=45)
ax3 = sns.boxplot(x = 'Class Name', y = 'Review Length', data = df)
ax3 = plt.title("Review Length in each Class")


# # Ratings vs. Positive Feedback Count

# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x = 'Rating', y = 'Positive Feedback Count', data = df)


# # Review Analysis

# In[ ]:


ps = PorterStemmer()
Reviews = df['Review Text'].astype(str)
print(Reviews.shape)
Reviews[Reviews.isnull()] = "NULL"


# In[ ]:


tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')
stop_words = set(stopwords.words('english'))
def preprocessing(data):
    txt = data.str.lower().str.cat(sep=' ') #1
    words = tokenizer.tokenize(txt) #2
    words = [w for w in words if not w in stop_words] #3
    #words = [ps.stem(w) for w in words] #4
    return words


# In[ ]:


df['tokenized'] = df["Review Text"].astype(str).str.lower() # Turn into lower case text
df['tokenized'] = df.apply(lambda row: tokenizer.tokenize(row['tokenized']), axis=1) # Apply tokenize to each row
df['tokenized'] = df['tokenized'].apply(lambda x: [w for w in x if not w in stop_words]) # Remove stopwords from each row


# In[ ]:


def string_unlist(strlist):
    return " ".join(strlist)

df["tokenized_unlist"] = df["tokenized"].apply(string_unlist)
df.head()


# ### Sentiment Analysis

# In[ ]:


# Pre-Processing
SIA = SentimentIntensityAnalyzer()

# Applying Model, Variable Creation
df['Polarity Score']=df["tokenized_unlist"].apply(lambda x:SIA.polarity_scores(x)['compound'])
df['Neutral Score']=df["tokenized_unlist"].apply(lambda x:SIA.polarity_scores(x)['neu'])
df['Negative Score']=df["tokenized_unlist"].apply(lambda x:SIA.polarity_scores(x)['neg'])
df['Positive Score']=df["tokenized_unlist"].apply(lambda x:SIA.polarity_scores(x)['pos'])

# Converting 0 to 1 Decimal Score to a Categorical Variable
df['Sentiment']=''
df.loc[df['Polarity Score']>0,'Sentiment']='Positive'
df.loc[df['Polarity Score']==0,'Sentiment']='Neutral'
df.loc[df['Polarity Score']<0,'Sentiment']='Negative'


# In[ ]:


conditions = [
    df['Sentiment'] == "Positive",
    df['Sentiment'] == "Negative",
    df['Sentiment'] == "Neutral"]
choices = [1,-1,0]
df['label'] = np.select(conditions, choices)
df.head()


# ### Simple Embedding Deep Neural Network

# In[ ]:


samples = df["tokenized_unlist"].tolist()
maxlen = 100 
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)


# In[ ]:


labels = np.asarray(df["label"].values)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[ ]:


indices = np.arange(df.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


# In[ ]:


training_samples = 11743
validation_samples = 17614
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: validation_samples] 
y_val = labels[training_samples: validation_samples]
x_test = data[validation_samples:]
y_test = labels[validation_samples:]
x_train = pad_sequences(x_train, maxlen=maxlen)
x_val = pad_sequences(x_val, maxlen=maxlen)


# ##### Baseline for Sentiment Analysis

# In[ ]:


# BASELINE
# That is, if all the labels are predicted as 1
(np.sum(df['label'] == 1)/df.shape[0]) * 100

# we have to make model that performs better than this baseline


# In[ ]:


def build_model():
    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
    return model


# In[ ]:


model = build_model()
model.summary()
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save("model1.h5")


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


model.evaluate(x_test, y_test)


# ### RNN

# In[ ]:


def build_RNN():
    model = Sequential() 
    model.add(Embedding(max_words, 100, input_length=maxlen)) 
    #model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32)) 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
    return model


# In[ ]:


model = build_RNN()
model.summary()
history_RNN = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save("model_RNN.h5")


# In[ ]:


acc = history_RNN.history['acc']
val_acc = history_RNN.history['val_acc']
loss = history_RNN.history['loss']
val_loss = history_RNN.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


model.evaluate(x_test, y_test)


# ## Prediction of Recommended IND[](http://)

# In[ ]:


# BASELINE
# That is, if all the labels are predicted as 1
(np.sum(df['Recommended IND'] == 1)/df.shape[0]) * 100

# we have to make model that performs better than this baseline


# ### Naive Bayes Classifier

# In[ ]:


def create_dict(tokenized_list):
    my_dict = dict([(word, True) for word in tokenized_list])
    return my_dict
df["NBCdata"] = df["tokenized"].apply(create_dict)
r_data = df["NBCdata"].values
reviews_labels = df["Recommended IND"].values


# In[ ]:


reviews_data = []
for i in range(len(r_data)):
    reviews_data.append([r_data[i], reviews_labels[i]])


# In[ ]:


train_data = reviews_data[:18788]
test_data = reviews_data[18788:]


# In[ ]:


classifier = NaiveBayesClassifier.train(train_data)


# In[ ]:


classifier.show_most_informative_features()


# In[ ]:


accuracy = nltk.classify.util.accuracy(classifier, test_data)
print("Classification Accuracy for Recommendation is...")
print(accuracy * 100)


# ### Deep Neural Network Classifier

# In[ ]:


# Deep learning models
labels = np.asarray(df["Recommended IND"].values)
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: validation_samples] 
y_val = labels[training_samples: validation_samples]
x_test = data[validation_samples:]
y_test = labels[validation_samples:]
x_train = pad_sequences(x_train, maxlen=maxlen)
x_val = pad_sequences(x_val, maxlen=maxlen)


# In[ ]:


model = build_model()
model.summary()
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save("model2.h5")


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


model.evaluate(x_test, y_test)


# In[ ]:


model = build_RNN()
model.summary()
history_RNN = model.fit(x_train, y_train,
                    epochs=2,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save("model_RNN2.h5")


# In[ ]:


acc = history_RNN.history['acc']
val_acc = history_RNN.history['val_acc']
loss = history_RNN.history['loss']
val_loss = history_RNN.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


model.evaluate(x_test, y_test)


# In[ ]:




