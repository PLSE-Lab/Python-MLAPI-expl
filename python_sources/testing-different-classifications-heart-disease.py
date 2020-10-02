#!/usr/bin/env python
# coding: utf-8

# **Three different classification algorithms on the heart disease dataset**

# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB


# In[41]:


# Load in data
df = pd.read_csv('../input/heart.csv')


# Exploratory data analysis

# In[42]:


df.head(10)


# In[43]:


# Convert target to bool
df['target'] = df['target'].astype('bool')


# In[44]:


df.dtypes


# In[45]:


# Summary Statistics
df.describe().round(decimals = 2)


# In[46]:


# Correlation
df.corr()


# In[47]:


# Correlation heatmap
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), cmap='YlGnBu', annot=True, linewidths=0.5, fmt='.1f', ax=ax)
plt.show()


# In[48]:


# Look for missing data
df.info()


# No missing data

# In[49]:


# cp has the strongest correlation with target, so let's look at that
df['cp'].value_counts()


# In[50]:


# And thalach
df['thalach'].plot(kind="hist", bins=20)


# **Modeling**
# 
# First with Naive Bayes classification

# In[ ]:


# Split training and testing
df_x = df.drop(['target'], axis=1)
x = df_x.values
y = df['target'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)


# In[51]:


# Train the model
gnb = MultinomialNB()
gnb.fit(x_train, y_train)


# In[52]:


# Test accuracy
print('Naive Bayes Score: %.3f' % gnb.score(x_test,y_test))


# Decision Tree

# In[53]:


# Train the model
dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)


# In[54]:


# Test accuracy
print('Decision Tree Score: %.3f' % dt.score(x_test,y_test))


# Random Forest Classifier

# In[56]:


# Train the model
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)


# In[57]:


# Test accuracy
print('Random Forest Score: %.3f' % rfc.score(x_test,y_test))

