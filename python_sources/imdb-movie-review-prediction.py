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


df_train = pd.read_csv('../input/Train.csv')
df_test = pd.read_csv('../input/Test.csv')
df_sample = pd.read_csv('../input/Sample_submission.csv')


# In[ ]:


df_train.head(2)


# In[ ]:


df_sample.head(3)


# In[ ]:


df_test.head(3)


# In[ ]:


train_reviews = df_train.review
test_reviews = df_test.review
labels = df_train.label


# In[ ]:


train_reviews[0]


# ### text  preprocessing

# In[ ]:


from nltk.tokenize  import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# In[ ]:


def clean_view(text):
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer('[a-zA-Z]+')
    stopword = set(stopwords.words('english'))
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    new_token = [ps.stem(token) for token in tokens if token not in stopword] # stemming and stopword removing
    return ' '.join(new_token)
    


# In[ ]:


clean_train = [clean_view(each) for each in train_reviews]


# In[ ]:


clean_test = [clean_view(each) for each in test_reviews]


# ### vectorization

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tf = TfidfVectorizer(ngram_range=(2, 2))
tf.fit(clean_train)


# In[ ]:


x_train = tf.transform(clean_train)
x_test = tf.transform(clean_test)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()


# In[ ]:


y = le.fit_transform(labels)


# In[ ]:


y


# ### Applying Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


model = MultinomialNB()


# In[ ]:


model.fit(x_train, y)


# In[ ]:


model.score(x_train, y)


# In[ ]:


pred = model.predict(x_test)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.DataFrame')


# In[ ]:


x_test.shape


# In[ ]:


df = pd.DataFrame({"Id":[i for i in range(pred.shape[0])],"label":pred}, index=None)


# In[ ]:


df['label'] = ['pos' if each == 1 else 'neg' for each in df.label ]


# In[ ]:


df.to_csv('submission.csv', index=None)


# In[ ]:




