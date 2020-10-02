#!/usr/bin/env python
# coding: utf-8

# # A simple bar chart notebook

# ### I have made single python function to draw bar chart for each feature compare to the Survived column

# In[1]:


#package

import pandas as pd
import numpy as np
import os


# In[2]:


df_train = pd.read_csv("../input/train.csv")
df_train_1 = df_train.copy()
df_train.head()


# In[3]:


df_test = pd.read_csv("../input/test.csv")
df_test.head()


# In[4]:


df_train.describe()


# In[5]:


df_train.info()


# In[6]:


df_train = df_train.drop(['Survived'],axis=1)
df_col = list(df_train.columns)
df_train_null = list(df_train.isnull().sum())
df_test_null = list(df_test.isnull().sum())


# In[7]:


data={
        "Column name":df_col,
        "Train": df_train_null,
        "Test": df_test_null
     }

df_null = pd.DataFrame(data)
df_null


# In[8]:


pclass = df_train["Pclass"].value_counts()
pclass


# In[9]:


df_train["Age"].fillna(df_train.groupby("Pclass")["Age"].transform("median"), inplace=True)


# In[10]:


df_train.isnull().sum()


# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
df_train = df_train_1


# In[12]:


def bar_chart(feature):
    survived = df_train[df_train['Survived']==1][feature].value_counts()
    not_survived = df_train[df_train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,not_survived])
    df.index=['Survived','Not_survived']
    df.plot(kind='bar',stacked=True,fig=(18,6),title=feature)


# In[13]:


bar_chart('Sex')


# In[14]:


bar_chart('Pclass')


# In[15]:


bar_chart('Embarked')


# In[16]:


bar_chart('SibSp')


# In[17]:


bar_chart('Parch')


# In[29]:


train=df_train.drop(['Survived'],1)
combine =  train.append(df_test) 
combine.reset_index(inplace=True)
combine.drop(['PassengerId','index'],1,inplace=True)
combine.head()


# In[25]:


combine.shape


# In[31]:


title = set()
for name in combine['Name']:
    title.add(name.split(",")[1].split(".")[0].strip())
print(title)


# ### A lot to do!!!
# 
# ## Thanks for reading it.
