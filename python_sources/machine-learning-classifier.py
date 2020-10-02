#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/train.csv",index_col=  0)


# In[ ]:


df.columns


# In[ ]:


df.size


# In[ ]:


# To change the names of DataFrame's Columns
df.columns = ["label","message"]


# In[ ]:


df.head()


# In[ ]:


# Change all the messages to string format
df['message'] = df['message'].apply(lambda x : str(x))


# In[ ]:


df["message"]


# In[ ]:


import nltk


# In[ ]:


df["length of message"] = df["message"].apply(len)


# In[ ]:


df.head()


# In[ ]:


# To plot a countplot to view how many SPAM messages are there and how many HAM
sns.countplot("label", data = df)


# In[ ]:


# To visualize the length of the messages based on the label
asdf = sns.FacetGrid(data = df, col = 'label')
asdf.map(sns.distplot, 'length of message', kde = False, hist_kws = dict(edgecolor = "k"))


# # Preprocessing

# In[ ]:


from nltk.corpus import stopwords
import string


# In[ ]:


# These are the most common word which we have to remove from text messages
stopwords.words("french")


# In[ ]:


# We need to remove punctuations too
string.punctuation


# *So we need to do following things for preprocessing <br>*
# *1. Remove the Punctuations <br>*
# *2. Remove the most common words*

# In[ ]:


# function for preprocessing
def all_words(msg):
    no_punctuation = [char for char in msg if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    word = [word for word in no_punctuation.split() if word.lower() not in stopwords.words("english")]
    return word


# In[ ]:


word=all_words(df["message"])


# # Traning

# In[ ]:


from collections import Counter


# In[ ]:


a=Counter(word)


# In[ ]:


len(a)


# In[ ]:


a.transform(df["message"])


# In[ ]:


print(bag_of_words_transformer)


# In[ ]:


len(bag_of_words_transformer)


# In[ ]:


# This will create the sparse matrix of all the messages based on the frequecy of words in that message
message_bow = bag_of_words_transformer.transform(df['message'])


# In[ ]:


# This is the shape of sparse matrix
# 310 is no. of message
# 1406 is the no. of words after preprocessing
message_bow.shape


# In[ ]:


tfid_transformer = TfidfTransformer().fit(message_bow)


# In[ ]:


message_tfid = tfid_transformer.transform(message_bow)


# In[ ]:


# Here Naive bayes has been used for traning
from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(message_tfid,df['label'])


# # Testing

# *For testing we need to get the Tfidf of all the messages*

# In[ ]:


test = pd.read_csv("../input/test.csv", index_col = 0)


# In[ ]:


test.head()


# In[ ]:


test["'text'"] = test["'text'"].apply(lambda x: str(x))


# In[ ]:


test_message_bow = bag_of_words_transformer.transform(test["'text'"])


# In[ ]:


test_message_tfid = tfid_transformer.transform(test_message_bow)


# In[ ]:


# Prediction
test["'label'"] = spam_detection_model.predict(test_message_tfid)


# In[ ]:


sns.countplot(test["'label'"])

