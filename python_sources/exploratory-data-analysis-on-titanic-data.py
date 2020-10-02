#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA) & Machine Learning

# reference: https://www.datacamp.com/community/tutorials/kaggle-machine-learning-eda

# ## How To Start with Supervised Learning

# a good way to approach supervised learning is the following:
# 
# 1) Perform an Exploratory Data Analysis (EDA) on the data set.
# 
# 2) Build a quick and dirty model, or a baseline model, which can serve as a comparison against later models.
# 
# 3) Iterate this process. Do more EDA and build another model.
# 
# 4) Engineer features: take the features and combine them or extract more information from them to eventually come to the last point, which is
# 
# 5) Get a model that performs better.

# ## Import Data and Check it Out

# In[ ]:


# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


import os
print(os.listdir('../input/titanic'))


# In[ ]:


# Import test and train datasets
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')

# View first lines of training data
df_train.head(n=4)


# In[ ]:


# View first lines of test data
df_test.head()


# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# ## Visual Exploratory Data Analysis (EDA) And the First Model

# In[ ]:


sns.countplot(x='Survived', data=df_train);


# In[ ]:


df_test['Survived'] = 0
df_test[['PassengerId', 'Survived']].to_csv('survivors.csv', index=False)


# ## EDA on Feature Variables

# In[ ]:


sns.countplot(x='Sex', data=df_train);


# In[ ]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);


# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


df_test['Survived'] = df_test.Sex == 'female'
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test.head()


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('women_survive.csv', index=False)


# ## Explore the Data More!

# In[ ]:


sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);


# In[ ]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);


# ## EDA with Numeric Variables

# In[ ]:


sns.distplot(df_train.Fare, kde=False);


# In[ ]:


df_train.groupby('Survived').Fare.hist(alpha=0.6);


# In[ ]:


df_train_drop = df_train.dropna()
sns.distplot(df_train_drop.Age, kde=False);


# In[ ]:


sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.3, jitter=True);


# In[ ]:


sns.swarmplot(x='Survived', y='Fare', data=df_train);


# In[ ]:


df_train.groupby('Survived').Fare.describe()


# In[ ]:


sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.5});


# In[ ]:


sns.pairplot(df_train_drop, hue='Survived');


# ## Summary

# Successfully:
# 
# 1) loaded the data and had a look at it.
# 
# 2) explored the target variable visually and made the first predictions.
# 
# 3) explored some of the feature variables visually and made more predictions that did better based on the EDA.
# 
# 4) done some serious EDA of feature variables, categorical and numeric.
# 
# 

# In[ ]:




