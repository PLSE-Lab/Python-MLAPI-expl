#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#import the required libraries
import keras
from keras.layers import Embedding
from keras.layers import Dense, Flatten, LSTM
from keras.layers import Input, GlobalMaxPool1D, Dropout
from keras.layers import Activation
from keras.models import Model, Sequential
from keras import optimizers

import os
import pandas as pd


# In[ ]:


#read the data files
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')


# In[ ]:


#lets look at the data
train.head()


# **Let's break down our input features and the output labels that we have to predict**
# 
# **1. Our input features which will be our inputs to our DNN architecture will be the column consisting of the comments text.**
# 
# **2. As our input features are in the form of a text and not in the form of numbers and integers which is usually prefferd by the ML algorithms we will have to find a way to convert the raw text in the form of integers so that they can be fed to our model. (how to do this explained later)**
# 
# **3. Also the input texts are of different lengths. But our model expects the input sequences to have the same length. So we need to find a way to make all the lenght of input sequences equal.**
# 
# **4. Coming to our labels that we have to predict, we got in total six predictor classes. Depending on the input text we have to predict to which class our input text can be classified.**
# 
# **5. The values of our output labels are 0 and 1, corresponding to whether a particular statement can be classified in that class or not.**
# 
# 
# **LET'S SEE FURTHER HOW TO SOLVE OUR PROBLEM**

# In[ ]:


#let's look at the test data
test.head()


# In[ ]:


#check for any null values in our dataset
print(train.isnull().any())


# In[ ]:


#again checking for any null values in the test dataset
print(test.isnull().any())


# In[ ]:


#divide our training data into features X and label Y
X_train = train['comment_text'] #will be used to train our model on
X_test = test['comment_text'] #will be used to predict the output labels to see how well our model has trained
y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values


# In[ ]:


y_train.shape #as expected total 6 columns for 6 predictor classes


# ## PREPROCESSING USING KERAS

# **Preprocessing is the most important step when it comes to any of the dataset before it is passed to a Machine Learning model. This fact is no less true when it comes to dealing with text dataset.**
# 
# **Infact for text dataset preprocessing is a must and without which we cannot move forward in training our model.**
# 
# **The fact that our algorithm only understand numbers and integers and cannot understand strings or text data makes preprocessing even more valuable and essential in Natural Language Processing.**
# 
# **When it comes to preprocessing a text data the most common steps are:**
# 
# **1. The entire sentence needs to be converted in the form of small tokens or into individual words. By doing this it helps us to assign each word with a unique integer value called as index or ID of that word.**
# 
# **2. Tokenization also helps us to find the total number of unique words in our vocabulary which is later to be fed in our model.**
# 
# **3. Once the tokenization is done, the next common step is to remove stopwords. Stopwords are the the most commonly occuring words in our vocabulary. Words that won't generally matter even if they are taken out from our vocabulary. For example words like 'the', 'and', 'so', 'if' etc are the stopwords. But we don't always have to remove the stopwords. Stopwords are only removed when their absence won't affect our model prediction much, in cases like classification of documents according to the topics, stopwords can be removed as we only have to focus on words that carry more importance in our documents.**
# 
# **4. There are various other ways in which preprocessing of the text data can be done.**

# 
# **KERAS is a high level deep learning API which is also very easy when it comes to applying deep learning algorithms. Keras offers a 'Tokenizer' class which carries out tokenisation for us and also convert all the text data into lowercase.**

# In[ ]:


#import the tokenizer class from the keras api
from keras.preprocessing.text import Tokenizer


# In[ ]:


#let's calculate the vocabulary size as well which will be given as an input to the Embedding layer
tokens = Tokenizer() #tokenizes our data
tokens.fit_on_texts(X_train)
vocab_size = len(tokens.word_index) + 1 #size of the total number of uniques tokens in our dataset
tokenized_train = tokens.texts_to_sequences(X_train) #converting our tokens into sequence of integers
tokenized_test = tokens.texts_to_sequences(X_test)


# In[ ]:


print(X_train[0]) #the first text
print(100 * '-')
print(tokenized_train[0]) #the correspondin first comment in the vectorized form


# 
# **1. Now if you observe the vector representation of each comment, the size of each vector is different.**
# 
# **2. But our machine learning model expects the size of our input data to be same throughout.**
# 
# **3. Hence we will be doing padding.**

# In[ ]:


#import the pad_sequences class from the keras api
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


max_len = 300 #maximum length of the padded sequence that we want (one of the hyperparameter that can be tuned)
padded_train = pad_sequences(tokenized_train, maxlen = max_len, padding = 'post') #post padding our sequences with zeros
padded_test = pad_sequences(tokenized_test, maxlen = max_len)


# In[ ]:


padded_train[:10] #as you can observe once our sentence ends the padding starts and continues until we have a vector of max_len


# ## LSTM MODEL USING FUNCTIONAL API

# **While building our model using the functional API we have used some new layers in our network**
# 
# **1. Starting few layers are same as the previous, the layer called Dropout is been added here. Now Dropout is a regularisation technique which we implement to prevent out model from overfitting to our training data. This technique randomly drops (or switches off) nodes from a particular layer with the given probability. So that the input and output both are blocked for the node which is been dropped.**
# 
# **2. Here we have stacked two (LSTM-->Activation-->GlobalMaxPool1D-->Dropout) onto each other followed by few fully connected layers and finally another fully connected layer with 6 neurons as the output layer of our model.**

# In[ ]:


input_model = Input(shape = (max_len, ))
x = Embedding(input_dim = vocab_size, output_dim = 120)(input_model)
x = LSTM(60, return_sequences = True)(x)
x = Activation('relu')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)

# x = LSTM(100, return_sequences = True)(x)
# x = Activation('relu')(x)
# x = GlobalMaxPool1D()(x)
# x = Dropout(0.2)(x)

x = Dense(150, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(100, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(6, activation = 'softmax')(x)


# In[ ]:


model = Model(inputs = input_model, outputs = x)
optim = optimizers.Adam(lr = 0.01, decay = 0.01 / 64) #defining our optimizer
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy']) #compiling the model


# In[ ]:


model.summary()


# In[ ]:


#fit the model onto our dataset
history = model.fit(padded_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.3)


# ## USING PRE-TRAINED WORD EMBEDDINGS

# **Pre-trained word embeddings are the ones which are usually trained on a large amount of corpus. Using such pre-trained word embeddings on your model can help achieve the model very good amount of accuracy. Using pre-trained word embeddings helps you to avoid training your own word embedding from scratch. This method can be employed when the available amount of dataset is not too much. Word2Vec(by google) and  Glove(by stanford NLP group) are the two most commonly use pre-trained word embeddings.**

# In[ ]:


import numpy as np

embedding_dim = 50
#def create_embedding_matrix(filepath, embedding_dim):
vocab_size = len(tokens.word_index) + 1  # Adding again 1 because of reserved 0 index
embedding_matrix = np.zeros((vocab_size, embedding_dim))

with open('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt', encoding = 'utf-8') as f:
    for line in f:
        word, *vector = line.split()
        if word in tokens.word_index:
            idx = tokens.word_index[word] 
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]


# In[ ]:


model1 = Sequential()
model1.add(Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = max_len, trainable = False))
model1.add(LSTM(60, return_sequences = True))
model1.add(Activation('relu'))
model1.add(GlobalMaxPool1D())
model1.add(Dropout(0.2))
model1.add(Dense(16, activation = 'relu'))
model1.add(Dropout(0.5))
model1.add(Dense(6, activation = 'softmax'))
model1.summary()


# In[ ]:


#compile the model
model1.compile(optimizer = optimizers.Adam(lr = 0.01, decay = 0.01 / 32), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


#fit the model on the dataset
history1 = model1.fit(padded_train, y_train, epochs = 5, batch_size = 32, validation_split = 0.3)


# In[ ]:


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_pred = model1.predict(padded_test)

sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")

sample_submission[list_classes] = y_pred

sample_submission.to_csv("model_submission.csv", index = False)


# In[ ]:




