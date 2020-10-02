#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.svm import SVC
from wordcloud import STOPWORDS, WordCloud, ImageColorGenerator
import unidecode
import xgboost

import nltk

lemmatizer = nltk.WordNetLemmatizer()

import string

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_train = pd.read_json("../input/train.json")
data_test = pd.read_json("../input/test.json")


# **Training Data**
# 
# Training data contains Cuisine and Ingredients column. 
# 
# * Objective is to use this training data to train the model. Model learns from the ingredients to predict the cuisine. 
# * This is a classification model building excercise. 
# * There are 20 different cuisines to predict. 
# * There are about 40,000 rows in the training data and 10,000 in test data. 

# In[ ]:


data_train.head()


# In[ ]:


print ('Number of rows in training data are',data_train.shape[0]) 


# In[ ]:


print('Number of unique cuisines in the training data are',len(data_train.cuisine.unique()))


# **Test Data**
# 
# Test data has about 10,000 rows with only ingredients columns. 

# In[ ]:


data_test.head()


# In[ ]:


print ('Number of rows in training data are',data_test.shape[0])


# **Number of Recipes**
# 
# Looking at the plot below we can notice that most number of recipes are Italian, followed by Mexian and Southern. Least number of recipies are Brazilian. 

# In[ ]:


data_train.cuisine.value_counts().plot(kind='bar')


# In[ ]:


y_train = data_train['cuisine'].apply(lambda x: x.lower())


# In[ ]:


def clean_text(text):
    text = " ".join([word.lower() for word in text])
    text = "".join([ps.stem(word) for word in text])
    return text


# In[ ]:


X_train = data_train['ingredients'].apply(lambda x: clean_text(x))
X_test = data_test['ingredients'].apply(lambda x: clean_text(x))


# In[ ]:


#X_train = data_train['ingredients'].apply(lambda x: ' '.join(lemmatizer.lemmatize(unidecode.unidecode(i)) for i in x).strip().lower())
#X_test = data_test['ingredients'].apply(lambda x: ' '.join(lemmatizer.lemmatize(unidecode.unidecode(i)) for i in x).strip().lower())


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


# Encode Lables of Cusine
labler = LabelEncoder()
y_target = labler.fit_transform(y_train)


# In[ ]:


# Vectorize Train and Test Data Columns
vectorizer = TfidfVectorizer(binary=True)
X_train_vec = vectorizer.fit_transform(X_train.values)
X_test_vec = vectorizer.transform(X_test.values)


# In[ ]:


#XGBoost implementation
model = xgboost.XGBClassifier(max_depth = 12, eta = 0.05, subsample = 0.7)
clf = OneVsRestClassifier(model, n_jobs = -1)
clf.fit(X_train_vec, y_target)


# In[ ]:


clf_pred = clf.predict(X_test_vec)


# In[ ]:


y_test = labler.inverse_transform(clf_pred)
test_id = data_test["id"]
submit_xg = pd.DataFrame({'id': test_id, 'cuisine': y_test}, columns=['id', 'cuisine'])
submit_xg.to_csv('xgboost.csv', index=False)


# In[ ]:




