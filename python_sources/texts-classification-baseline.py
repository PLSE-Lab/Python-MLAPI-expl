#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


DATA_DIR = '/kaggle/input/texts-classification-iad-hse-intro-2020/'

df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


sub.head()


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


df_train['Category'].value_counts()


# In[ ]:


len(df_train['Category'].unique())


# In[ ]:


plt.figure(figsize=(11, 8))
df_train['Category'].hist(bins=50)
plt.show()


# In[ ]:


skf = StratifiedKFold(n_splits=10)

for train_index, test_index in skf.split(df_train, df_train['Category']):
    df_train_small = df_train.iloc[test_index]
    break

df_train_small.shape, df_train.shape


# In[ ]:


plt.figure(figsize=(11, 8))
df_train_small['Category'].hist(bins=50)
plt.show()


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(df_train_small, df_train_small['Category'], random_state=13)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[ ]:


del df_train


# In[ ]:


X_train.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntfidf = TfidfVectorizer(max_features=5000)\nX_train_tfidf = tfidf.fit_transform(X_train['title'])")


# In[ ]:


len(tfidf.vocabulary_)


# In[ ]:


X_train_tfidf.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nlr = LogisticRegression(random_state=13)\nlr.fit(X_train_tfidf, y_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nX_val_tfidf = tfidf.transform(X_val['title'])")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ny_val_pred = lr.predict(X_val_tfidf)\naccuracy_score(y_val, y_val_pred)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nX_test_tfidf = tfidf.transform(df_test['title'])\ny_test_pred = lr.predict(X_test_tfidf)")


# In[ ]:


sub.shape, y_test_pred.shape


# In[ ]:


sub['Category'] = y_test_pred
sub.head()


# In[ ]:


plt.figure(figsize=(11, 8))
sub['Category'].hist(bins=50)
plt.show()


# In[ ]:


sub.to_csv('submission.csv', index=False)

