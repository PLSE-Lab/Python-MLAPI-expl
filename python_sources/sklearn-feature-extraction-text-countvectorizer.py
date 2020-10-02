#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


corpus_1 = [
    'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]

corpus_2 = [
    'Abhishek Anand.',
     'Jai ram.',
     'Poem?',
     'What are you.',
    'are you ravi?'
 ]


# In[ ]:


corpus_1


# In[ ]:


#corpus_2


# In[ ]:


vectorizer = CountVectorizer()


# In[ ]:


x_1 = vectorizer.fit_transform(corpus_1)
x_2 = vectorizer.fit_transform(corpus_2)
# print(x_1)
# print("----------")
# print(x_2)


# In[ ]:


x_1


# In[ ]:


x_2


# In[ ]:


print(vectorizer.get_feature_names())
print(len(vectorizer.get_feature_names()))


# In[ ]:


print(x_1.toarray())


# In[ ]:


print(x_2.toarray())


# In[ ]:


vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))


# In[ ]:


X2 = vectorizer2.fit_transform(corpus_1)


# In[ ]:


print(vectorizer2.get_feature_names())


# In[ ]:


print(X2.toarray())


# In[ ]:




