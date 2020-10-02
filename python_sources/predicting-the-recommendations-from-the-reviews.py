#!/usr/bin/env python
# coding: utf-8

# # Do the user recommend the product ?

# We will analyse the text in the reviews to assess if the user recommends the tablet.

# ## Looking at a sample of the data

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# In[ ]:


nRowsRead = 100

df = pd.read_csv("../input/final_data.csv", delimiter=',', nrows = nRowsRead)
df = df.sample(frac=1)
df.head()


# In[ ]:


df.info()


# This dataset contains many useless columns for our objective, we will remove them.

# ## Preprocessing the full dataset

# ### Making the training and testing datasets

# In[ ]:


df = pd.read_csv("../input/final_data.csv", delimiter=',', encoding = "ISO-8859-1", nrows = None)
df = df.sample(frac=1)
df = df[["reviews.doRecommend", "reviews.text", "reviews.title"]]


# In[ ]:


df.head()


# Our dataframe only contains useful information, we can now separate the data into a training and a testing dataset.
# 
# However, we need to make sure that the training set is representative of the complete dataset.

# In[ ]:


df.info()


# Many rows do not have a value for the recommendation, we will drop them as they are useless for the training.
# 
# We can also drop the reviews without titles, it is a very small portion of the dataset.

# In[ ]:


df = df.dropna()
df.info()


# In[ ]:


df["reviews.doRecommend"].value_counts()


# In[ ]:


df["reviews.doRecommend"].astype(float).hist(figsize=(8,5))


# Very few reviews do not recommend the product, we must be careful not to create a sample bias.
# 
# We will use stratified sampling to ensure a representative proportion of "False" in the training dataset.

# In[ ]:


df.rename(columns={'reviews.title':'title','reviews.doRecommend':'doRecommend'}, inplace=True)


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in split.split(df, df["doRecommend"]):
    train_set = df.iloc[train_index]
    test_set = df.iloc[test_index]


# In[ ]:


print(train_set["doRecommend"].value_counts()/len(train_set))
print(test_set["doRecommend"].value_counts()/len(test_set))


# The proportions are respected.
# 
# Let's only consider the titles for now.

# In[ ]:


train_set = train_set.drop(columns = ["reviews.text"])
test_set = test_set.drop(columns = ["reviews.text"])

train_set["title"] = train_set["title"].astype(str)
test_set["title"] = test_set["title"].astype(str)

train_set["doRecommend"] = train_set["doRecommend"].astype(float)
test_set["doRecommend"] = test_set["doRecommend"].astype(float)


# In[ ]:


train_set.head()


# ### Processing the text

# In[ ]:


maxlen = 10 # Maximal length of sequence considered
num_words = 600 # Number of words in your vocabulary
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789' # Some chars you want to remove in order to have a clean text
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=num_words, filters=filters) # Word tokenizer
tokenizer.fit_on_texts(train_set['title'].tolist()) # Fit on training samples


train_titles_tok = tokenizer.texts_to_sequences(train_set['title'].tolist())
test_titles_tok = tokenizer.texts_to_sequences(test_set['title'].tolist())


from keras.preprocessing.sequence import pad_sequences
train_titles_pad = pad_sequences(train_titles_tok, maxlen=maxlen)
test_titles_pad = pad_sequences(test_titles_tok, maxlen=maxlen)


# In[ ]:


print(train_set.iloc[0]["title"],train_titles_pad[0])
print(train_set.iloc[1]["title"],train_titles_pad[1])
print(train_set.iloc[2]["title"],train_titles_pad[2])
print(train_set.iloc[3]["title"],train_titles_pad[3])


# We can see the correlation between the text and the associated arrays : every word corresponds to an integer, we only consider the ten first words of the title.

# In[ ]:


from sklearn.preprocessing import LabelBinarizer
label_enc = LabelBinarizer().fit(list(set(train_set['doRecommend'].values))) 
train_labels = label_enc.transform(train_set['doRecommend'].values)
test_labels = label_enc.transform(test_set['doRecommend'].values)


# ## Creating a model

# In[ ]:


from keras.optimizers import Adam, RMSprop, SGD, Adagrad
from keras.callbacks import EarlyStopping

class LearningModel():
    def __init__(self, dim=20):
        self.dim= dim # Dimension of word embeddings
        self.n_label = 2 # Number of labels
        self.optimizer = Adam(lr=0.01) # Optimizer method for stochastic gradient descent
        self.epochs = 20
        self.batch_size = 128
        self.callbacks = EarlyStopping(monitor='val_loss', patience=2)
        self.model = None # Keras model, it will be instantiated later...
        
    def compile(self):
        print(self.model.summary())
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self, mode=''):
        self.compile()
        if mode == 'EarlyStopping':
            self.model.fit(train_titles_pad, train_labels,
                           epochs=self.epochs, 
                           batch_size=self.batch_size, 
                           validation_data=(test_titles_pad, test_labels), 
                           callbacks=[self.callbacks], verbose=2)
        else: 
            for _ in range(self.epochs):
                self.model.fit(train_titles_pad, train_labels,
                               epochs=1, 
                               batch_size=self.batch_size, 
                               validation_split=0.1, verbose=2)
                self.test()
    
    def test(self):
        print('Evaluation : ', self.model.evaluate(test_titles_pad, test_labels, batch_size=2048))


# ## Training the model

# This model will take into input a vector indicating the words present in the title and create an embedding vector out of it. This vector will help determine if the review is positive or negative.

# In[ ]:


from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, BatchNormalization

lm = LearningModel(dim=20)

lm.model = Sequential()

lm.model.add(Embedding(num_words, lm.dim, input_length=maxlen))
lm.model.add(GlobalAveragePooling1D())
lm.model.add(Dense(1, activation='sigmoid'))

lm.train()


# We achieve an accuracy of 96%, which seems good but given the unbalance in the data, a classifier giving always the output 'True' would have an accuracy close to that.

# In[ ]:


print(train_set["doRecommend"].value_counts()/len(train_set))
print(test_set["doRecommend"].value_counts()/len(test_set))


# ## Let's try the same method with a balanced dataset

# In[ ]:


df = df.reset_index()
df.head()


# In[ ]:


df1 = df[(df["doRecommend"] == False) | (df["index"]<3000)]
df1.info()


# In[ ]:


print(df1["doRecommend"].value_counts()/len(df1))


# From now on, the dataset contains 64% of "True", a model with a better precision on the validation set actually learns from the training set.

# In[ ]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in split.split(df1, df1["doRecommend"]):
    train_set1 = df1.iloc[train_index]
    test_set1 = df1.iloc[test_index]


# In[ ]:


train_set1["title"] = train_set1["title"].astype(str)
test_set1["title"] = test_set1["title"].astype(str)

train_set1["doRecommend"] = train_set1["doRecommend"].astype(float)
test_set1["doRecommend"] = test_set1["doRecommend"].astype(float)


# In[ ]:


tokenizer = Tokenizer(num_words=num_words, filters=filters) # Word tokenizer
tokenizer.fit_on_texts(train_set1['title'].tolist()) # Fit on training samples


train_titles_tok = tokenizer.texts_to_sequences(train_set1['title'].tolist())
test_titles_tok = tokenizer.texts_to_sequences(test_set1['title'].tolist())


from keras.preprocessing.sequence import pad_sequences
train_titles_pad = pad_sequences(train_titles_tok, maxlen=maxlen)
test_titles_pad = pad_sequences(test_titles_tok, maxlen=maxlen)


# In[ ]:


label_enc = LabelBinarizer().fit(list(set(train_set1['doRecommend'].values))) 
train_labels = label_enc.transform(train_set1['doRecommend'].values)
test_labels = label_enc.transform(test_set1['doRecommend'].values)


# In[ ]:


lm = LearningModel(dim=20)

lm.model = Sequential()

lm.model.add(Embedding(num_words, lm.dim, input_length=maxlen))
lm.model.add(GlobalAveragePooling1D())
lm.model.add(Dense(1, activation='sigmoid'))

lm.train()


# We could improve the accuracy by looking at "n-grams" of words, for example by considering each word with the following one.
# 
# It would allow the network to detect the difference between "very good" and "not good"

# ## Let's try with the complete text

# In[ ]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for train_index, test_index in split.split(df1, df1["doRecommend"]):
    train_set1 = df1.iloc[train_index]
    test_set1 = df1.iloc[test_index]


# In[ ]:


train_set1 = train_set1.drop(columns = ["title"])
test_set1 = test_set1.drop(columns = ["title"])

train_set1["reviews.text"] = train_set1["reviews.text"].astype(str)
test_set1["reviews.text"] = test_set1["reviews.text"].astype(str)

train_set1["doRecommend"] = train_set1["doRecommend"].astype(float)
test_set1["doRecommend"] = test_set1["doRecommend"].astype(float)


# In[ ]:


maxlen = 200
num_words = 600

tokenizer = Tokenizer(num_words=num_words, filters=filters) # Word tokenizer
tokenizer.fit_on_texts(train_set1["reviews.text"].tolist()) # Fit on training samples

# We will use the same name even in they are not title to directly use the learning model we created earlier
train_titles_tok = tokenizer.texts_to_sequences(train_set1["reviews.text"].tolist())
test_titles_tok = tokenizer.texts_to_sequences(test_set1["reviews.text"].tolist())


from keras.preprocessing.sequence import pad_sequences
train_titles_pad = pad_sequences(train_titles_tok, maxlen=maxlen)
test_titles_pad = pad_sequences(test_titles_tok, maxlen=maxlen)


# In[ ]:


label_enc = LabelBinarizer().fit(list(set(train_set['doRecommend'].values))) 
train_labels = label_enc.transform(train_set1['doRecommend'].values)
test_labels = label_enc.transform(test_set1['doRecommend'].values)


# In[ ]:


lm = LearningModel(dim=20)

lm.model = Sequential()

lm.model.add(Embedding(num_words, lm.dim, input_length=maxlen))
lm.model.add(GlobalAveragePooling1D())
lm.model.add(Dense(1, activation='sigmoid'))

lm.train()

