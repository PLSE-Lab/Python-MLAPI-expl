#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train=pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.heatmap(df_train.isnull(),cmap='YlGnBu')


# In[ ]:


df_train.Cabin.isnull().value_counts()


# In[ ]:


df_train=df_train.drop('Cabin',axis=1)


# In[ ]:


df_train.head()


# In[ ]:


df_train.Age.isnull().value_counts()


# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_train)


# In[ ]:


df.groupby('Pclass').mean().Age


# In[ ]:


def age(cols):
    age=cols[0]
    pclass=cols[1]
    
    if pd.isnull(age):
        if pclass==1:
            return 38
        elif pclass==2:
            return 29
        else:
            return 25
    else:
        return age


# In[ ]:


df_train['Age'] = df_train[['Age','Pclass']].apply(age,axis=1)


# In[ ]:


sns.heatmap(df_train.isnull(),cmap='YlGnBu')


# In[ ]:


sns.countplot('Survived',data=df_train)


# In[ ]:


sns.countplot('Survived',hue='Sex',data=df_train)
sns.countplot('Pclass',hue='Survived',data=df_train)
sns.set_style('whitegrid')
sns.distplot(df_train['Age'],kde=False,bins=30)
sns.countplot(x='SibSp',data=df_train)


# In[ ]:


df_train['Fare'].hist(bins=40,figsize=(8,4))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




