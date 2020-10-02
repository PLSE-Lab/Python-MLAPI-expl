#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# 1. [Loading Data](#1)
# 1. [Preprocessing Data](#2)
#     * [Stopwords](#3)
#     * [Lemmatization](#4)
#     * [Data Cleaning](#5)
# 1. [Bag of words](#6)
# 1. [Text Classification](#7)
#     * [Train Test Split](#8)
#     * [Naive Bayes](#9)
#     * [Data Prediction](#10)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id="1"></a> <br>
# # Loading Data

# In[ ]:


data = pd.read_csv(r"../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding='latin-1')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data = pd.concat([data.gender,data.description],axis=1)


# In[ ]:


data.head()


# In[ ]:


data.dropna(axis = 0, inplace = True)


# In[ ]:


data.gender = [1 if each == "female" else 0 for each in data.gender]


# In[ ]:


data.gender.unique()


# <a id="2"></a> <br>
# # Preprocessing Data
# * Regular Expression : RE

# In[ ]:


import re


# In[ ]:


first_description = data.description[4]
first_description


# In[ ]:


description = re.sub("[^a-zA-Z]"," ",first_description) # Don't choose a to z and A to Z, another ones replace with space


# In[ ]:


description


# In[ ]:


description = description.lower()
description


# <a id="3"></a> <br>
# ## Stopwords
# * Irrelavent words for exapmle : the, and..

# In[ ]:


import nltk # natural language tool kit
nltk.download("stopwords") # downloading into corpus file
from nltk.corpus import stopwords # importing from corpus file


# In[ ]:


# description = description.split()
# tokenizer from nltk can be used instead of split
# but if we use split, words like "shouldn't" don't seperate like "should" and "not"
description = nltk.word_tokenize(description)


# In[ ]:


print(description)


# In[ ]:


description = [ word for word in description if not word in set(stopwords.words("english"))]


# In[ ]:


print(description)


# <a id="4"></a> <br>
# ## Lemmatization
# * Finding root of words
# * EX:
#     * love -> loved

# In[ ]:


import nltk as nlp


# In[ ]:


lemma = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description ]


# In[ ]:


print(description)


# In[ ]:


description = " ".join(description)
print(description)


# <a id="5"></a> <br>
# ## Data Cleaning

# In[ ]:


description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description ]
    description = " ".join(description)
    description_list.append(description)


# <a id="6"></a> <br>
# # Bag of words

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
max_features = 500


# In[ ]:


count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english",)


# In[ ]:


sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()


# In[ ]:


print("Most common {} words : {}".format(max_features,count_vectorizer.get_feature_names()))


# <a id="7"></a> <br>
# # Text Classification

# In[ ]:


y = data.iloc[:,0].values # male ofr female classes
x = sparce_matrix


# <a id="8"></a> <br>
# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


# <a id="9"></a> <br>
# ## Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)


# <a id="10"></a> <br>
# ## Prediction

# In[ ]:


y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))

