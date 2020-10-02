#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/mbti_1.csv')


# In[ ]:


data.head()


# In[ ]:


mapping = {
    'I': 'Introversion',
    'E': 'Extroversion',
    'N': 'Intuition',
    'S': 'Sensing',
    'T': 'Thinking',
    'F': 'Feeling',
    'J': 'Judging',
    'P': 'Perceiving',
}


# In[ ]:


X = pd.DataFrame()
for c in 'INTJESFP':
    X[c] = data['type'].apply(lambda x: 1 if c in x else 0)


# In[ ]:


_ = X.sum().sort_values().rename(lambda x: mapping[x]).plot.barh()


# In[ ]:


cv = CountVectorizer(max_features=2000, strip_accents='ascii')
result = cv.fit_transform(data['posts'])


# In[ ]:


X = pd.concat([X, pd.DataFrame(result.toarray(), columns=['w_' + k for k in cv.vocabulary_.keys()])],
              axis=1)


# In[ ]:


wcols = [col for col in X.columns if col.startswith('w_') and len(col) > 5]
XX = X[wcols].T[X[wcols].mean() >= 0.5].T
def unique_words(a, b):
    (XX[X[a] == 1].mean() / XX[X[b] == 1].mean()).sort_values().rename(lambda x: x[2:]).tail(10).plot.barh()
    plt.title(mapping[a] + ' vs ' + mapping[b])


# In[ ]:


unique_words('E', 'I')


# In[ ]:


unique_words('I', 'E')


# In[ ]:


unique_words('N', 'S')


# In[ ]:


unique_words('S', 'N')


# In[ ]:


unique_words('T', 'F')


# In[ ]:


unique_words('F', 'T')


# In[ ]:


unique_words('J', 'P')


# In[ ]:


unique_words('P', 'J')

