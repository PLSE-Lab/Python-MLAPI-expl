#!/usr/bin/env python
# coding: utf-8

# # Introduction

# I made this kernel for Kaggle's 'NLP with Disaster Tweets' competition. In this kernel, given thousands of tweets, I tried to identify whether the tweet talks about a disaster or not.

# # Contents

# * Preliminary steps
#     * Importing the necessary libraries
#     * Converting the CSV file into a pandas dataframe
# * Creating new columns
# * Visualizing the data
# * Encoding the features of the train data
# * Defining the features and prediction target
# * Creating the model
# * Fitting the model
# * Dealing with the test data
#     * Encoding the features of the test data
# * Prediction
# * Ending Note

# ### Preliminary Steps

# Importing the necessary libraries - 

# In[ ]:


import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

tqdm.pandas()
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


# Converting the CSV file into a pandas dataframe - 

# In[ ]:


os.listdir('../input/nlp-getting-started')


# In[ ]:


train_data = pd.read_csv('../input/nlp-getting-started/train.csv')


# A look at the train data - 

# In[ ]:


train_data.head(10)


# ### Creating new columns

# Creating 3 new columns - tweet_length, tweet_words and average_word_length

# In[ ]:


train_data["tweet_length"] = train_data["text"].progress_apply(len)
train_data["tweet_words"] = train_data["text"].progress_apply(lambda x: len(x.split()))
train_data["average_word_length"] = train_data["tweet_length"]/train_data["tweet_words"]


# A look at the train data with the new columns - 

# In[ ]:


train_data.head(10)


# ### Visualizing the data

# Tweet Words Distributions - 

# In[ ]:


sns.distplot(train_data["tweet_words"], color="deeppink")
plt.show()


# Tweet Length Distribution - 

# In[ ]:


sns.distplot(train_data["tweet_length"], color="teal")
plt.show()


# Average Word Length Distribution - 

# In[ ]:


sns.distplot(train_data["average_word_length"], color="darkorchid")
plt.show()


# Target vs. Tweet Words - 

# In[ ]:


sns.boxplot(data=train_data, x="target", y="tweet_words", palette=["turquoise", "hotpink"])
plt.show()


# Target vs. Tweet Length -

# In[ ]:


sns.boxplot(data=train_data, x="target", y="tweet_length", palette=["turquoise", "hotpink"])
plt.show()


# Target vs. Average Tweet Length -

# In[ ]:


sns.boxplot(data=train_data, x="target", y="average_word_length", palette=["turquoise", "hotpink"])
plt.show()


# ### Encoding the features of the train data

# Converting the tweets to vectors - 

# In[ ]:


count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_data["text"]).todense()


# ### Defining the features and prediction target

# In[ ]:


X = train_vectors/train_vectors.max(axis=1)
y = train_data["target"].values.reshape((len(train_data), 1))


# Splitting the training data into training data and validation data -

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y)


# ### Creating the model

# In[ ]:


model = Sequential()
model.add(Dropout(0.85))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


# Providing the input size to the model -

# In[ ]:


model.build(input_shape=(None, 21637))
model.summary()


# ### Fitting the model

# In[ ]:


model.fit(x=train_X, y=train_y, validation_data=(val_X, val_y), epochs=100)


# ### Dealing with the test data

# Converting the CSV file into a pandas dataframe - 

# In[ ]:


test_data = pd.read_csv('../input/nlp-getting-started/test.csv')


# ### Encoding the features of the test data

# Encoding the features of the test data and defining a new variable to hold the features - 

# In[ ]:


test_vectors = count_vectorizer.transform(test_data["text"]).todense()
X_test = test_vectors/test_vectors.max(axis=1)
X_test[np.isnan(X_test)] = 0


# ### Prediction

# In[ ]:


predictions = np.round(model.predict(X_test)).reshape((len(X_test)))


# Since sample_submission.csv is of the format in which our submission is supposed to be made, I'm first importing it and converting it into a pandas dataframe -

# In[ ]:


sample_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')


# Replacing the 'target' column in the dataframe with the values we got -

# In[ ]:


sample_submission["target"] = np.int32(predictions)


# A final look at the dataframe with our predictions -

# In[ ]:


sample_submission.head(10)


# Converting the dataframe into a csv file without the index column -

# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# ### Ending Note

# Through this project, I learnt about the conversion of text to vectors. I really enjoyed it, and look forward to learning more in the future. This being only my third ml model, I really appreciate feedback to help me improve both the accuracy and efficiency of my model :)
