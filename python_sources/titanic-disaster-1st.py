#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train [train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.plot(kind = 'bar', stacked = True, figsize = (15,15))


# In[ ]:


bar_chart('Pclass')


# In[ ]:


bar_chart('Sex')


# In[ ]:


bar_chart('SibSp')


# In[ ]:


bar_chart('Parch')


# In[ ]:


bar_chart('Embarked')


# In[ ]:


bar_chart('Cabin')


# In[ ]:


bar_chart('Survived')


# In[ ]:




