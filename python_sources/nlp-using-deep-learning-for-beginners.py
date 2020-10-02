#!/usr/bin/env python
# coding: utf-8

# Hi, if you are a beginner in Tensorflow and would like to catch up the essence of Natural Language Processing (Like me), please upvote <3

# In[ ]:


# importing the libraries
import pandas as pd
import tensorflow as tf
import numpy as np


# In[ ]:


# importing the Deep Learning Libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout


# In[ ]:


# loading the training data
training_data = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')


# In[ ]:


training_data.head()


# 1. We dont need the qid to train the model.
# 2. The question_text is the text input that has to be fitted in the model along with the target
# 3. The target has 2 classes.

# In[ ]:


# dropping the qid
training_data = training_data.drop(['qid'], axis = 1)


# In[ ]:


# creating a feature length that contains the total length of the question
training_data['length'] = training_data['question_text'].apply(lambda s: len(s))
# I used a basic way of utilizing a lambda function.


# In[ ]:


# now checking the mean length of the text for tokenizing the data.
min(training_data['length']), max(training_data['length']), round(sum(training_data['length'])/len(training_data['length']))


# minimum length = 1 ?? looks like outliers, How can a question contain just a single word ? Let us do some preprocessing. 

# In[ ]:


training_data[training_data['length'] <= 9]


# oops...**Are these even complete questions ???**

# In[ ]:


training_data = training_data.drop(training_data[training_data['length'] <= 9].index, axis = 0)
min(training_data['length']), max(training_data['length']), round(sum(training_data['length'])/len(training_data['length']))


# Looks sensible. Now lets check for missing values (if any)

# In[ ]:


training_data.isnull().sum()


# No Missing Values !
# 
# Let us start the Deep Learning part now !

# In[ ]:


# Tokenizing the text - Converting each word, even letters into numbers. 
max_length = round(sum(training_data['length'])/len(training_data['length']))
tokenizer = Tokenizer(num_words = max_length, 
                      filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                     lower = True,
                     split = ' ')


# In[ ]:


tokenizer.fit_on_texts(training_data['question_text'])


# In[ ]:


# Actual Conversion takes place here.
X = tokenizer.texts_to_sequences(training_data['question_text'])


# In[ ]:


print(len(X), len(X[0]), len(X[1]), len(X[2]))


# As you can see the lengths are not same. So Pad sequences are used. Pad sequences adds a specific value, usually 0, before or after the text sequence to make them equal in length

# In[ ]:


X = pad_sequences(sequences = X, padding = 'pre', maxlen = max_length)
print(len(X), len(X[0]), len(X[1]), len(X[2]))


# In[ ]:


y = training_data['target'].values
y.shape


# Now the data is ready to be fed into the neural network. Now constructing the neural network NLP. 
# 
# I will create a neural network with minimum layers so that beginners like me can understand without complexity.

# In[ ]:


# LSTM Neural Network
lstm = Sequential()
lstm.add(Embedding(input_dim = max_length, output_dim = 120))
lstm.add(LSTM(units = 120, recurrent_dropout = 0.2))
lstm.add(Dropout(rate = 0.2))
lstm.add(Dense(units = 120, activation = 'relu'))
lstm.add(Dropout(rate = 0.1))
lstm.add(Dense(units = 2, activation = 'softmax'))

lstm.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


lstm_fitted = lstm.fit(X, y, epochs = 1)


# Note: you can play with the hyperparameters to get the expected accuracy.

# In[ ]:


# importing the testing data
testing_data = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')


# In[ ]:


testing_data.head()


# In[ ]:


# converting the data into tokens
X_test = tokenizer.texts_to_sequences(testing_data['question_text'])


# In[ ]:


print(len(X_test), len(X_test[0]), len(X_test[1]), len(X_test[2]))


# In[ ]:


# paddding the sequences
X_test = pad_sequences(X_test, maxlen = max_length, padding = 'pre')
print(len(X_test), len(X_test[0]), len(X_test[1]), len(X_test[2]))


# We are good to go !!

# In[ ]:


# predicting the test set
lstm_prediction = lstm.predict_classes(X_test)


# In[ ]:


# creating a dataframe for submitting
submission = pd.DataFrame(({'qid':testing_data['qid'], 'prediction':lstm_prediction}))


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index = False)


# Thank you for viewing my kernel. Please comment if you have any creative ideas of doing traditional methods.
# 
# *Let's Learn ! Let's Learn !* 
