#!/usr/bin/env python
# coding: utf-8

# **FAKE AND REAL NEWS **
# 
# This dataset has 17903 uniques news titles and their corresponding text. 
# It has been categorized into two .csv files, one for True and the other one for Fake.
# 
# Our aim to to train a model to predict if a particular news is real of fake

# **1) IMPORT ALL LIBRARIES**

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# **2) DATA PREPROCESSING**

# Let us import the two csv files

# In[ ]:


true=pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake=pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
true.head()


# We will categorise the news into true=1 and fake=0

# In[ ]:


true['result']=1
fake['result']=0


# In[ ]:


true.head()


# Let us concatenate these two dataframes

# In[ ]:


df=pd.concat([true,fake])
df.tail()


# We have no null values

# In[ ]:


df.isna().sum()


# In[ ]:


df['text']=df['title']+""+df['text']+""+df['subject']
del df['title']
del df['date']
del df['subject']
df.head()


# In[ ]:


sentence = df['text'].values.tolist()
result= df['result'].values.tolist()


# **3) TRAIN TEST SPLIT**

# We have a split size of 0.2, hence 20% of the data will be used for testing.

# In[ ]:


X_train, X_test, Y_train,Y_test= train_test_split(sentence, result, test_size=0.2)


# In[ ]:


Y_train=np.array(Y_train)
Y_test=np.array(Y_test)


# **4) NLP TECHNIQUES**

# Tokenization: is the process of tokenizing or splitting a string, text into a list of tokens. One can think of token as parts like a word is a token in a sentence, and a sentence is a token in a paragraph.
# 
# Padding: When processing sequence data, it is very common for individual samples to have different lengths. Hence, we pad the sequences to make an array with each row being the vectorized values of each sentence.

# In[ ]:


tokenizer=Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(X_train)
padded_train=pad_sequences(sequences,5000,truncating='post')



# In[ ]:


sequences_test=tokenizer.texts_to_sequences(X_test)
padded_test=pad_sequences(sequences_test,5000,truncating='post')


# In[ ]:


padded_test.shape


# In[ ]:


Y_test.shape


# 5) CONVOLUTIONAL NEURAL NETWORK

# In[ ]:


model= tf.keras.Sequential([
    tf.keras.layers.Embedding(10000,16,input_length=5000),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


history=model.fit(padded_train, Y_train, epochs=10, validation_data=(padded_test, Y_test))


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# We have a great accuracy with just about 3 epochs!
# 
# This dataset can be found on Kaggle.

# In[ ]:




