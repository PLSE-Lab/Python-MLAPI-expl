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


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/mlcourse/telecom_churn.csv')
df.head()


# In[ ]:


df.shape


# ## Whole dataset visualization

# In[ ]:


df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})


# In[ ]:


df['Churn'] = df['Churn'].astype('int')


# In[ ]:


plt.rcParams['figure.figsize'] = (16, 12)
df.drop(['State'], axis = 1).hist()


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


df.drop([feat_name for feat_name in df.columns if 'charge' in feat_name], axis = 1, inplace=True) 


# In[ ]:


df.shape


# In[ ]:


df.shape


# In[ ]:


sns.countplot(x = 'State', data = df)


# In[ ]:


State = df['State']


# In[ ]:


df.drop('State', axis=1, inplace=True)


# In[ ]:


df['Total day minutes'].describe()


# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(random_state=17)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_repr_churn = tsne.fit_transform(df)')


# In[ ]:


X_repr_churn.shape


# In[ ]:


plt.rcParams['figure.figsize'] = (8,6)
plt.scatter(X_repr_churn[df['Churn'] == 1 , 0], 
            X_repr_churn[df['Churn'] == 1 ,1], alpha= .5, c='blue');
plt.scatter(X_repr_churn[df['Churn'] == 0 , 0], 
            X_repr_churn[df['Churn'] == 0 ,1], alpha= .5, c='orange');
plt.xlabel('tSNE axis #1')
plt.ylabel('tSNE axis #2')
plt.legend()
plt.title('tSNE Repr');
plt.savefig('churn_tsne.png', dpi= 300);


# <img src= 'churn_tsne.png'>

# In[ ]:





# In[ ]:


sns.boxplot(x = 'Total day minutes', data = df)


# In[ ]:


plt.rcParams['figure.figsize'] = (8,6)
df['Total day minutes'].hist()


# In[ ]:





# In[ ]:





# In[ ]:


df['Churn'].value_counts(normalize = True)


# In[ ]:


sns.countplot(x = 'Churn', data = df)


# In[ ]:





# In[ ]:


plt.scatter(df['Total day minutes'], df['Customer service calls'])


# In[ ]:



df.corrwith(df['Total day minutes'])


# In[ ]:





# In[ ]:


pd.crosstab(df['Churn'], df['Customer service calls'])


# In[ ]:


sns.countplot(x= 'Customer service calls', hue = 'Churn', data = df)
plt.title('Customer Service Calls For loyal and churned');


# In[ ]:





# In[ ]:




