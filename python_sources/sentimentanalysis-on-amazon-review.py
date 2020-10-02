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


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv('../input/moviereviews.tsv',sep='\t')


# In[ ]:


df.head()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


blanks = []

for i , la , rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)


# In[ ]:


blanks


# In[ ]:


df.drop(blanks,inplace=True)


# In[ ]:


df['label'].value_counts()


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[ ]:


sid = SentimentIntensityAnalyzer()


# In[ ]:


df['scores'] = df['review'].apply(lambda review : sid.polarity_scores(review))


# In[ ]:


df['compound'] = df['scores'].apply(lambda d:d['compound'])


# In[ ]:


df['com_score'] = df['compound'].apply(lambda score :'pos' if score >=0 else 'neg')


# In[ ]:


df.head()


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


accuracy_score(df['label'],df['com_score'])


# In[ ]:


print(classification_report(df['label'],df['com_score']))


# In[ ]:


print(confusion_matrix(df['label'],df['com_score']))


# In[ ]:




