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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/nlp-topic-modelling/Reviews.csv')


# In[ ]:


df.head()


# In[ ]:


df = df[['Text']]


# In[ ]:


pd.set_option('display.max_colwidth', 200)
df.head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')


# In[ ]:


dtm = tfidf.fit_transform(df['Text'])


# In[ ]:


dtm


# In[ ]:


from sklearn.decomposition import NMF


# In[ ]:


nmf_model = NMF(n_components=9, random_state=101)


# In[ ]:


nmf_model.fit(dtm)


# In[ ]:


for index, topic in enumerate(nmf_model.components_):
    print(f"The new 15 models# {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# In[ ]:


topic_model = nmf_model.transform(dtm)


# In[ ]:


topic_model.argmax(axis =1)


# In[ ]:


df['topic'] = topic_model.argmax(axis =1)


# In[ ]:


df.head(10)


# In[ ]:


model = {0:'sweet related', 1:'coffee related', 2:'product review', 3:'tea related', 4:'animal food', 5:'shopping Related', 6:'food love', 7:'food related', 8:'cookie & chocolate'}


# In[ ]:


df['Title'] = df['topic'].map(model)


# In[ ]:


df.head()


# In[ ]:


df['Title'].value_counts()


# In[ ]:


plt.figure(figsize=(15,9))
sns.countplot(x = 'Title', data = df)

