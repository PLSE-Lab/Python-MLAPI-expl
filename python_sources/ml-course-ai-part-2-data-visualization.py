#!/usr/bin/env python
# coding: utf-8

# **Tutorial Practice**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for plotting purposes
from matplotlib import pyplot as plt
#pip install seaborn / conda install seaborn
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/telecom_churn.csv')


# In[ ]:


df.head()


# In[ ]:


print(df.shape)


# **Whole Dataset Visualizations**

# In[ ]:


df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})


# In[ ]:


df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})


# In[ ]:


df['Churn'] = df['Churn'].astype('int64');


# In[ ]:


df.head()


# In[ ]:


#histograms
#correcting figure size

plt.rcParams['figure.figsize'] = (16,12)
df.drop(['State'], axis = 1).hist();


# In[ ]:


df.corr()  #correlation matrix


# In[ ]:


sns.heatmap(df.corr());


# In[ ]:


# checking for feature names having 'charge'

[feat_name for feat_name in df.columns
if 'charge' in feat_name]


# In[ ]:


# drop 'charge' from dataset
df.drop([feat_name for feat_name in df.columns
if 'charge' in feat_name], axis=1)


# In[ ]:


#checking shape of the modified dataset
df.drop([feat_name for feat_name in df.columns
if 'charge' in feat_name], axis=1).shape


# In[ ]:


#checking initial dataset
#the initial or actual dataset did not change
df.shape


# In[ ]:


#if we want the dataset to be modified (inplace=True helps in acheiving this)
df.drop([feat_name for feat_name in df.columns
if 'charge' in feat_name], axis=1, inplace=True)


# In[ ]:


#And the dataset got modified
df.shape


# In[ ]:


# Features one at a time
# Numeric features

df['Total day minutes'].describe()


# In[ ]:


plt.rcParams['figure.figsize'] = (8,6)
sns.boxplot(x='Total day minutes', data=df);


# In[ ]:


plt.rcParams['figure.figsize'] = (8,6)
df['Total day minutes'].hist();


# **Categorical**

# In[ ]:


df['State'].nunique()


# In[ ]:


df['State'].value_counts().head()


# In[ ]:


df['Churn'].value_counts()


# In[ ]:


# use normalize to get the percentage values
df['Churn'].value_counts(normalize=True)


# In[ ]:


sns.countplot(x='Churn', data=df);


# **Interaction between features**
# 
# **Numeric-Numeric**
# (helpful in regression task where numeric is the target variable)

# In[ ]:


states = df['State']
df.drop('State', axis = 1, inplace = True)


# In[ ]:


# Correlation of 'Total day minutes' with other numerical values in the dataset
df.corrwith(df['Total day minutes'])


# In[ ]:


plt.scatter(df['Total day minutes'], df['Customer service calls']);


# **Categorical - Categorical**

# In[ ]:


pd.crosstab(df['Churn'], df['Customer service calls'])


# In[ ]:


sns.countplot(x = 'Customer service calls', hue = 'Churn', data = df);
plt.title('Customer Service Calls for Loyal & Churned');


# **SKlearn (Scikit Learn)**

# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


df.shape


# In[ ]:


#state = df['State']
#df.drop('State', axis = 1, inplace = True)
df.head()


# In[ ]:


tsne = TSNE(random_state=17)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_repr = tsne.fit_transform(df)')


# In[ ]:


X_repr.shape


# In[ ]:


#whole dataset
plt.rcParams['figure.figsize'] = (8,6)
plt.scatter(X_repr[:, 0], X_repr[:, 1]);


# In[ ]:


#Churned Customers
plt.rcParams['figure.figsize'] = (8,6)
plt.scatter(X_repr[df['Churn']==1, 0], 
            X_repr[df['Churn']==1, 1]);
                   


# In[ ]:


#Loyal Customers
plt.rcParams['figure.figsize'] = (8,6)
plt.scatter(X_repr[df['Churn']==0, 0], 
            X_repr[df['Churn']==0, 1]);


# In[ ]:


#Overlapping plot

plt.rcParams['figure.figsize'] = (10,8)
plt.scatter(X_repr[df['Churn']==1, 0], 
            X_repr[df['Churn']==1, 1], alpha = 0.5, c='red', 
            label='Churn')
plt.scatter(X_repr[df['Churn']==0, 0], 
            X_repr[df['Churn']==0, 1], alpha = 0.5, c='green', 
            label='Loyal');
plt.xlabel('TSNE axis #1');
plt.ylabel('TSNE axis #2');
plt.legend();
plt.title('TSNE representation');
plt.savefig('churn_tsne.png', dpi=300)


# <img src = 'churn_tsne.png'>

# In[ ]:


#<img src = 'churn_tsne.png'>


# **Categorical - Numeric**

# In[ ]:


df.groupby('Churn')['Total day minutes',
                   'Customer service calls'].mean()


# In[ ]:


df.groupby('Churn')['Total day minutes',
                   'Customer service calls'].agg(np.median)


# In[ ]:


df.groupby('Churn')['Total day minutes',
                   'Customer service calls'].agg([np.median, np.std])


# In[ ]:


sns.boxplot(x='Churn', y = 'Total day minutes', data = df, hue = 'Churn');


# In[ ]:




