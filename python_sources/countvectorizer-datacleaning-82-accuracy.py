#!/usr/bin/env python
# coding: utf-8

# In[70]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords # for stopwords
import string 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[13]:


train_data = pd.read_csv("../input/train.csv")


# In[14]:


test_data = pd.read_csv("../input/test.csv") 


# In[16]:


train_data.head()


# In[18]:


train_data["author"].unique()


# In[20]:


train_data["author"].value_counts()


# In[21]:


# we don't need id for now, id has no effect on output
train_data.drop( "id" , axis=1 , inplace = True)


# In[25]:


# Let's seperate the documents and output(author name) 

train_document = train_data["text"]
train_authors = train_data["author"]

#for testing data

test_document = test_data["text"]


# In[33]:


#Let's create stopword list 

stopword_list = stopwords.words('english')
stopword_list[0:5]


# In[32]:


#let's create new list of punctuation which should be removed from documents

punct = list(string.punctuation)
punct[0:5]


# In[34]:


stopword_list = stopword_list + punct


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer #for converting documents into matrix form
from sklearn.model_selection import train_test_split # for splitting the training data


# In[42]:


# Let's split the training data into train and test data
x_train , x_test , y_train , y_test = train_test_split ( train_document , train_authors)


# In[40]:


cv = CountVectorizer(stop_words = stopword_list)


# In[48]:


xtrain = cv.fit_transform( x_train ) # train the model and find features using training data


# In[49]:


xtest = cv.transform( x_test )


# In[54]:


#Let's import classifiers 
from sklearn.naive_bayes import MultinomialNB 


# In[55]:


clf = MultinomialNB()


# In[56]:


clf.fit(xtrain,y_train)


# In[59]:


clf.score(xtest , y_test)


# In[60]:


# a gud accuracy we found using multinomial naive bayes


# Now let's predict the actual test data

# In[71]:


training_data = cv.fit_transform(train_document)


# In[72]:


testing_data = cv.transform(test_document)


# In[73]:


#It's time to predict the output


# In[75]:


clf.fit(training_data ,train_authors)


# In[76]:


prediction = clf.predict(testing_data)


# **prediction array contain the prediction of testing data**
