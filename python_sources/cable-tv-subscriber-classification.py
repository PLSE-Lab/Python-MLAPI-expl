#!/usr/bin/env python
# coding: utf-8

# # Cable TV Subscriber Classification

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/CableTVSubscribersData.csv")


# In[ ]:


data.groupby('subscribe').size()


# ## Dataset Exploration

# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.dtypes


# Numerical columns

# In[ ]:


data.describe()


# Categorical columns

# In[ ]:


data[['gender','ownHome','subscribe','Segment']].describe()


# ### Missing values

# In[ ]:


import missingno as msno


# In[ ]:


msno.matrix(data)


# Looks like we don't have any null values, Nice

# ### Feature normalization

# We want our categorical values to be integers for better analysis.

# In[ ]:


catdf = pd.get_dummies(data[['gender','ownHome','Segment']])
data = pd.concat([data, catdf],axis=1, sort=False)
data.head()


# In[ ]:


data.drop(columns = ['gender', 'ownHome', 'Segment'], inplace = True)
data['subscribe'] = pd.Categorical(data['subscribe'])
data['subscribe'] = data['subscribe'].cat.codes


# In[ ]:


data.columns


# ## Variable analysis

# For this kernel, we will try to classify wheter someone will subscribe to tv services

# In[ ]:


sns.countplot(data['subscribe'])
plt.xticks([0,1], ['subNo', 'subYes'])
plt.show()


# In[ ]:


data.groupby(['subscribe']).size()


# We have a very imbalanced number of observation for our target classes. To deal with imbalance data for classification we may approach it in two ways: undersampling or oversampling. Since the minor class only have 40 observations, undersampling is not recommended, thus we go for oversampling. 
# <hr>
# But before that we will first split our test and training set before performing oversampling on the training dataset, accourding to [this page](https://beckernick.github.io/oversampling-modeling/) this is the proper way of oversampling

# In[ ]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data, data['subscribe'])


# In[ ]:


df_class_0 = x_train[x_train['subscribe'] == 0]
df_class_1 = x_train[x_train['subscribe'] == 1]


# In[ ]:


df_class_1_over = df_class_1.sample(len(df_class_0.index), replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

df_test_over['subscribe'].value_counts()
sns.countplot(df_test_over['subscribe'])
plt.xticks([0,1], ['subNo', 'subYes'])
plt.show()


# We now have balance classes, for the next step let us find the most important predictors

# In[ ]:


data_train = df_test_over


# In[ ]:


plt.figure(figsize = (12,9))
sns.heatmap(data_train.corr(), cmap='YlGnBu', annot=True)
plt.show()


# **Insights:**
# 1. Segment_Suburb mix is correlated to our target variable (Negative correlation) as well as Segment_Moving up (Positive correlation).
# 1. Income and age are highly correlated with each other, we may remove one of them later.
# 1. Age is also highly correlated with Segment_Travelers, kids and Segment_Urban hip
# 1. income is highly correlated with Segment_Urban hip
# 1. Gender_Male and Gender Male is 100% correlated with each other, duhh. (same with ownHome_Yes and ownHome_No)
# 
# Not much interesting inferences from the correlation map, let us proceed to feature importance using Random Forest
# <hr>

# ----- To be continued ----

# In[ ]:




