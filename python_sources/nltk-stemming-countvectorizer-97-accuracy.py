#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


data = pd.read_csv("../input/spam.csv" ,encoding='latin1')


# In[3]:


#let's seperate the output and documents
y = data["v1"].values
x = data["v2"].values


# In[4]:


from nltk.corpus import stopwords # for excluding the stopwords
import re # for excluding the integers and punctuation


# In[5]:


from nltk.stem import PorterStemmer # for finding the root words


# In[7]:


ps = PorterStemmer()


# In[8]:


ps.stem("joking")  #how port stemmer works


# In[9]:


stopword = set(stopwords.words('english')) # list of stopwords


# In[10]:


x = [re.sub('[^a-zA-Z]',' ',doc) for doc in x ] #  include only characters and replace other characters with space
 


# In[11]:


document = [doc.split() for doc in x ] # split into words


# In[12]:


def convert(words) :
  
    current_words = list()
    for i in words :
        if i.lower() not in stopword :
            
            updated_word = ps.stem(i)
            current_words.append(updated_word.lower())
    return current_words
            


# In[13]:


document = [ convert(doc)   for doc in document ] # update the documetns


# In[14]:


document = [ " ".join(doc) for doc in document] # again join the words into sentences


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# In[16]:


xtrain , xtest , ytrain , ytest = train_test_split(document,y)


# In[17]:


cv = CountVectorizer(max_features = 1000) # 1000 features we will use


# In[18]:


a = cv.fit_transform(xtrain) # fit using training data and transform into matrix form


# In[20]:


b= cv.transform(xtest) #transform testing data into matrix form


# In[21]:


a.todense()


# In[22]:


from sklearn.naive_bayes import MultinomialNB


# In[23]:


clf = MultinomialNB()


# In[24]:


clf.fit(a,ytrain)


# In[25]:


clf.score(b,ytest)


# **97.8% accuracy we are getting using multinomial naive bayes which is good enough**

# ***if you find this notebook useful then please upvote and for any queries ask in comment i will clear your queries***
