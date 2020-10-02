#!/usr/bin/env python
# coding: utf-8

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


# **Importing data**

# In[ ]:


train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# **Data Analysing**
# 

# In[ ]:


train.head()


# **Checking missing values**

# In[ ]:


train.isnull().sum()

train.info
# In[ ]:


test.isnull().sum()


# In[ ]:


test.info


#  **Dropping LOCATION column **

# In[ ]:


train.drop(columns = 'location', inplace = True)
test.drop(columns = 'location', inplace = True)


# **Checking fro relations in KEYWORD and TARGET**

# In[ ]:


train[train['keyword'].notnull()][train['target']== 1]


# In[ ]:


train[train['keyword'].notnull()][train['target']== 0]


# In[ ]:


train['keyword'].value_counts().index


# In[ ]:


train.head(10)


# Data Cleaning
# 

# Lowercase text

# In[ ]:


def lowercase_text(text):
    text = text.lower()
    return text
train['text'] = train['text'].apply(lambda x : lowercase_text(x))
test['text'] = test['text'].apply(lambda x : lowercase_text(x))


# In[ ]:


train['text'].head(10)


# In[ ]:


import string
string.punctuation


# In[ ]:


train.head(10)


# **Removing punctuations**

# In[ ]:


def remove_punctuation(text):
    text_no_punctuation = "".join([c for c in text if c not in string.punctuation])
    return text_no_punctuation
train["text"] = train["text"].apply(lambda x: remove_punctuation(x))
test["text"] = test["text"].apply(lambda x: remove_punctuation(x))


# In[ ]:


train['text'].head(10)


# **Removing noise**

# **Tokenize**

# In[ ]:


import nltk
from nltk.tokenize import RegexpTokenizer
# Tokenizing the training and the test set
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))
test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
train['text'].head()


# **Deleting stopwords**

# In[ ]:


from nltk.corpus import stopwords
print(stopwords.words('english'))


# In[ ]:


train.head(10)


# **Remove stop words
# **

# In[ ]:


def remove_stopwords(text):
    """
    Removing stopwords belonging to english language
    
    """
    
    words = [w for w in text if w not in stopwords.words('english')]
    return words


train['text'] = train['text'].apply(lambda x : remove_stopwords(x))
test['text'] = test['text'].apply(lambda x : remove_stopwords(x))


# In[ ]:


train.head(10)


# **Stemming and lemmatization(optional)**

# In[ ]:


def combine_text(list_of_text):
    combine_text = ' '.join(list_of_text)
    return combine_text
train["text"] = train["text"].apply(lambda x: combine_text(x))
test["text"] = test["text"].apply(lambda x: combine_text(x))


# In[ ]:


train.head()


# **Count vectorize**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# that the tokens in the train vectors are the only ones mapped to the test vectors - 

train_vector = count_vectorizer.fit_transform(train["text"]).todense()
test_vector = count_vectorizer.transform(test["text"]).todense()


# In[ ]:


print(count_vectorizer.vocabulary_)


# In[ ]:


print(train_vector.shape)
print(test_vector.shape)


# **Model fitting and implementation**

# In[ ]:


from sklearn.model_selection import train_test_split

Y = train["target"]
x_train, x_test, y_train,y_test = train_test_split(train_vector,Y, test_size = 0.3, random_state = 0)
y_train


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C = 3.0)
scores = cross_val_score(model, train_vector, train['target'], cv=5)


# In[ ]:


print(scores.mean())


# In[ ]:


model.fit(x_train, y_train)
y_pred_model_1 = model.predict(x_test)


# In[ ]:


print(accuracy_score(y_test,y_pred_model_1))


# In[ ]:


y_pred_test = model.predict(test_vector)


# **Accuracy score**

# **Putting result in sample_submission file******

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svm = SVC(kernel = "linear", C = 0.15, random_state = 100)
svm.fit(x_train, y_train)
y_pred = svm.predict(test_vector)


# In[ ]:


sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv("submission.csv", index=False)
sub.head(10)


# In[ ]:




