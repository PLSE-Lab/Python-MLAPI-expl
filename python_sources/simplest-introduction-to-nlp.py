#!/usr/bin/env python
# coding: utf-8

# Importing necessary modules and dependencies

# In[ ]:


#Ignore the warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

#Data visualization and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#configure
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style = 'whitegrid', color_codes = True)

import nltk

#importing stop-words
from nltk.corpus import stopwords
stop_words = set(nltk.corpus.stopwords.words('english'))

#Tokenization
from nltk import word_tokenize, sent_tokenize

#Keras
import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Input
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf


# Creating sample texts

# In[ ]:


sample_text_1="bitty bought a bit of butter"
sample_text_2="but the bit of butter was a bit bitter"
sample_text_3="so she bought some better butter to make the bitter butter better"

corp = [sample_text_1, sample_text_2, sample_text_3]
no_docs = len(corp)
print(no_docs)


# Integer encoding all texts

# In[ ]:


VOCAB_SIZE = 50
encod_corp = []
for i, doc in enumerate(corp):
    encod_corp.append(one_hot(doc, VOCAB_SIZE))
    print('The encoding for document', i+1, 'is : ', one_hot(doc, VOCAB_SIZE))


# In[ ]:


encod_corp #list of lists


# Padding texts (to make all texts of the same length)

# In[ ]:


#Finding max_len
MAX_LEN = -1
for doc in corp:
    tokens = nltk.word_tokenize(doc)
    if(len(tokens) > MAX_LEN):
        MAX_LEN = len(tokens)
print('The maximum number of unique words in any document is : ', MAX_LEN)


# In[ ]:


#How nltk word tokenizes a text
nltk.word_tokenize(corp[0])


# In[ ]:


#Actual padding
pad_corp = pad_sequences(encod_corp, maxlen=MAX_LEN, padding='post', value=0)
pad_corp


# Creating embeddings

# In[ ]:


#Specifying the input shape
input = Input(shape=(no_docs, MAX_LEN), dtype='float64')
input


# In[ ]:


word_input = Input(shape = (MAX_LEN,), dtype = 'float64')


# In[ ]:


#Creating the embedding
word_embedding = Embedding(input_dim = VOCAB_SIZE, output_dim = 8, input_length= MAX_LEN)(word_input)


# In[ ]:


#Flattening the embedding
word_vec = Flatten()(word_embedding)
word_vec


# In[ ]:


#combining all into a Keras Model
embed_model = Model([word_input], word_vec)


# In[ ]:


#Training the model
embed_model.compile(optimizer = Adam(lr = 1e-3), loss='binary_crossentropy', metrics = ['acc'])


# In[ ]:


#Model summary
print(embed_model.summary())


# In[ ]:


#Getting the embeddings
embeddings = embed_model.predict(pad_corp)


# In[ ]:


print('Shape of embeddings : ', embeddings.shape)
print(embeddings)


# In[ ]:


#Reshaping embeddings
embeddings = embeddings.reshape(-1, MAX_LEN, 8)
print('Shape of embeddings : ', embeddings.shape)
print(embeddings)


# Each word is 8D!

# Getting encoding for a word in a document

# In[ ]:


for i, doc in enumerate(embeddings):
    for j, word in enumerate(doc):
        print('The encoding for Word', j+1, 'in Document', i+1, ' : ', word)


# That's it for Today! Please Upvote if you enjoyed through this kernel like me!
