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


df = pd.read_csv('/kaggle/input/topics-classification/topics.csv')


# In[ ]:


df.head(2)


# In[ ]:


del df['Unnamed: 0']


# In[ ]:


df.head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')


# In[ ]:


dtm = tfidf.fit_transform(df['question_text'])


# In[ ]:


dtm


# In[ ]:


from sklearn.decomposition import NMF
nmf_model = NMF(n_components=9, random_state=101)


# In[ ]:


nmf_model.fit(dtm)


# In[ ]:


for index, topic in enumerate(nmf_model.components_):
    print(f"The top 15 model#{index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# In[ ]:


topic_model = nmf_model.transform(dtm)


# In[ ]:


topic_model.argmax(axis =1)


# In[ ]:


df['mark'] = topic_model.argmax(axis =1)


# In[ ]:


df.head()


# In[ ]:


model = {0: 'ecommerce site', 1: 'shipped related', 2: 'product info', 3: 'dress info', 4: 'member promo code', 5: 'problem', 6: 'banking card', 7: 'refund policy', 8: 'cloth info'}


# In[ ]:


pd.set_option('display.max_colwidth', 200)


# In[ ]:


df['qus_title'] = df['mark'].map(model)


# In[ ]:


df.head(20)


# In[ ]:


plt.figure(figsize=(20,12))
sns.countplot(x = 'qus_title', data =df)

