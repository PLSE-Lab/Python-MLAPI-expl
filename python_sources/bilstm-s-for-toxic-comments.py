#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.models import Model, Sequential
from keras import optimizers

import os
import pandas as pd


# In[ ]:


#read the data files
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')


# In[ ]:


#divide our training data into features X and label Y
X_train = train['comment_text'] #will be used to train our model on
X_test = test['comment_text'] #will be used to predict the output labels to see how well our model has trained
y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values


# ## PREPROCESSING

# 
# 
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
# 

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
# 
# 1. Now if you observe the vector representation of each comment, the size of each vector is different.
# 
# 2. But our machine learning model expects the size of our input data to be same throughout.
# 
# 3. Hence we will be doing padding.
# 

# In[ ]:


#import the pad_sequences class from the keras api
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


max_len = 300 #maximum length of the padded sequence that we want (one of the hyperparameter that can be tuned)
padded_train = pad_sequences(tokenized_train, maxlen = max_len, padding = 'post') #post padding our sequences with zeros
padded_test = pad_sequences(tokenized_test, maxlen = max_len)


# In[ ]:


padded_train[:10] #as you can observe once our sentence ends the padding starts and continues until we have a vector of max_len


# ## USING PRE-TRAINED WORD EMBEDDINGS

# **Pre-trained word embeddings are the ones which are usually trained on a large amount of corpus. Using such pre-trained word embeddings on your model can help achieve the model very good amount of accuracy. Using pre-trained word embeddings helps you to avoid training your own word embedding from scratch. This method can be employed when the available amount of dataset is not too much. Word2Vec(by google) and Glove(by stanford NLP group) are the two most commonly use pre-trained word embeddings.**

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
            embedding_matrix[idx] = np.array(vector, dtype = np.float32)[:embedding_dim]


# ## Implementing Bidirectional LSTM's

# **1. Here we will be impelemnting Bidirectional LSTMs which are an extension of traditional LSTMs. The main reason to use bidirectional LSTM's instead of traditional LSTM's is that we can achieve a significant improvement in the performance of the model on sequence classification problems.**
# 
# **2. Bidirectional LSTMs train two instead of one LSTMs on the input sequence. The first on the input sequence as-is and the second on a reversed copy of the input sequence. This can provide additional context to the network and result in faster and even fuller learning on the problem.**
# 
# **3. Using bidirectional LSTM's helps us to interpret the past as well as the future information of our problem.**

# In[ ]:


model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = max_len, trainable = False))
model.add(Bidirectional(LSTM(50, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(50, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(6, activation = 'sigmoid'))
model.summary()


# In[ ]:


#compile the model
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 0.01, decay = 0.01/32), metrics = ['accuracy'])


# In[ ]:


#fit the model on the dataset
history = model.fit(padded_train, y_train, epochs = 5, batch_size = 128, validation_split = 0.3)


# ## Making predictions on the test data

# In[ ]:


#list all the output class labels
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

#make the predictions
y_pred = model.predict(padded_test, verbose = 1, batch_size = 128)


# In[ ]:


#making submission
#read in the submission file
sample_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv")

sample_submission[list_classes] = y_pred

sample_submission.to_csv("BiLSTM_submission.csv", index = False)


# In[ ]:




