#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


df = pd.read_csv('../input/breastCancer.csv')
df.head()


# **This program is an example of supervised machine learning using different classification models**

# **Let us first check if our data is clean**

# In[4]:


df.info()


# **Our data is fortunately clean. We don't need the id column for our training model.**

# In[5]:


df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
df.info()


# **Let us first visualize some attributes**

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


sns.barplot(x = 'diagnosis', y = 'radius_mean',data = df, palette='RdBu_r')


# In[8]:


sns.barplot(x = 'diagnosis', y = 'area_mean',data = df, palette='RdBu_r')


# In[9]:


sns.barplot(x = 'diagnosis', y = 'symmetry_worst',data = df, palette='RdBu_r')


# **In the above examples we can see a clear difference between the values of attributes of Benign and Malignant patients. Similarily other features also have the differences**

# **Checking the correlation between the attributes**

# In[10]:


df.corr()


# **Negative values infer inverse relation.**
# Some of the basic points that can be infered from the above data are:
# (1) as the radius increases it makes the tumor worst.
# (2) as the size increases, in general, worstness increases.
# 

# **Ploting the heatmap for  the correlation matrix**

# In[11]:


f, ax = plt.subplots(figsize = (9,6))
sns.heatmap(df.corr(), ax = ax)


# In[12]:


labels = np.array(df.columns[1:])
labels


# **Checking the mean values by grouping into tumor categories**

# In[21]:


grp_df = df.groupby('diagnosis', as_index = False)[labels].mean()
grp_df


# In[30]:


grp_melt = grp_df.melt('diagnosis',value_vars=labels)
sns.barplot(x = 'variable', y = 'value', data = grp_melt.head(6), hue ='diagnosis')


# **import important libraries for modelling data**

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('diagnosis', axis = 1), df.diagnosis) 


# **Logistic Regression**

# In[32]:


from sklearn.linear_model import LogisticRegression
model_l = LogisticRegression()
model_l.fit(X_train,y_train)
model_l.score(X_test, y_test)


# **Decision Tree**

# In[33]:


from sklearn.tree import DecisionTreeClassifier
model_t = DecisionTreeClassifier()
model_t.fit(X_train, y_train)
model_t.score(X_test, y_test)


# **Random Forest**

# In[34]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators = 30)
model_rf.fit(X_train, y_train)
print('Score: {0:0.4f}'.format(model_rf.score(X_test, y_test)))


# In[35]:


Model = ['Logistic Regression','Decision Tree','Random Forest']
Accuracy = [model_l.score(X_test, y_test), model_t.score(X_test, y_test), model_rf.score(X_test, y_test)]
for i in range(3):
    print('Score for %s is: %0.2f' %(Model[i], Accuracy[i]))

