#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('seaborn')


# In[ ]:


df = pd.read_csv('/kaggle/input/food-preferences/Food_Preference.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.drop(['Timestamp','Participant_ID'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.isnull().any()


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.dropna()


# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


sns.countplot(x='Dessert', data = df, color = '#C70039')


# In[ ]:


df['Dessert'] = df['Dessert'].replace('Yes',1)
df['Dessert'] = df['Dessert'].replace('No',0)
df['Dessert'] = df['Dessert'].replace('Maybe',1)


# In[ ]:


pd.to_numeric(df['Dessert'],errors = 'coerce')


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='Dessert',data=df,color='#3ff08a',ax=ax[0])
sns.countplot(x='Food', data = df, color = '#e14735',ax=ax[1])


# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='Gender',data=df,color='#4d5ae8',ax=ax[0])
sns.countplot(x='Dessert', data = df, color = '#C70039',ax=ax[1])


# In[ ]:


fig,ax = plt.subplots(figsize=(10,10))
sns.scatterplot(x='Age',y='Dessert',hue = 'Nationality',ax = ax,color = '#e14735',data=df)
plt.show()


# In[ ]:


fig,ax = plt.subplots(figsize=(15,7))
sns.countplot(y='Nationality',data = df)
plt.show()


# In[ ]:


fig,ax = plt.subplots(figsize=(15,17))
sns.countplot(y='Age',data = df)


# > ## That's all for today ;)
# #### For some awesome projects, visit my **GitHub** profile : [kanishksh4rma](http://github.com/kanishksh4rma)
# 

# In[ ]:




