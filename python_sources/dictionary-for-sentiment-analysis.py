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


df = pd.read_csv('/kaggle/input/dictionary-for-sentiment-analysis/dict.csv', encoding = 'latin1', header = None)


# In[ ]:


df.head()

Create columns name
# In[ ]:


df.columns = ['review', 'level']


# In[ ]:


df.head()


# **Create Sentiment Analyzer**

# In[ ]:


import nltk


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[ ]:


sid = SentimentIntensityAnalyzer()


# In[ ]:


test = 'he is the best boy in the class'


# In[ ]:


sid.polarity_scores(test)


# In[ ]:


sid.polarity_scores(df['review'].iloc[4])


# **New column score**

# In[ ]:


df['scores'] = df['review'].apply(lambda re: sid.polarity_scores(re))


# In[ ]:


df.head()


# **select compound number******

# In[ ]:


df['compound'] = df['scores'].apply(lambda d: d['compound'])


# In[ ]:


df.head()


# **Create another new column******

# In[ ]:


df['compound_label'] = df['compound'].apply(lambda number: 'positive' if number>=0 else 'negative')


# In[ ]:


df.head()


# **Use confusion_matrix, classification_report, accuracy_score**

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[ ]:


print(accuracy_score(df['level'], df['compound_label']))


# In[ ]:


print(classification_report(df['level'], df['compound_label']))


# In[ ]:


print(confusion_matrix(df['level'], df['compound_label']))


# In[ ]:


df

