#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/database.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.body_camera.value_counts()


# In[ ]:


df.manner_of_death.value_counts()


# In[ ]:


df.armed.value_counts()


# In[ ]:


df.gender.value_counts()


# In[ ]:


df.race.value_counts()


# In[ ]:


sns.countplot(df['race'],hue=df['threat_level'],palette='Blues_d')


# In[ ]:


df.flee.value_counts()


# In[ ]:


df.isnull().sum()


# In[ ]:


DF = df.dropna()
DF.shape


# In[ ]:


sns.distplot(DF['age'],kde=True,rug=True)


# In[ ]:


DF['age'].describe()


# **Exploring the dataset by age groups**

# In[ ]:


minor_by_gen = df.loc[df['age'] < 18].gender.value_counts()
minor_by_eth = df.loc[df['age'] < 18].race.value_counts()
n_minors = len(df.loc[df['age'] < 18])
print(minor_by_gen, '\n\n{}'.format(minor_by_eth), '\n\nThere were {} minors who were shot.' .format(str(n_minors)))


# In[ ]:


df.loc[df['age'] < 18].armed.value_counts()


# In[ ]:


sns.countplot(df.loc[df['age'] < 18]['armed'],palette = 'Paired')


# Of those 34 minors, there are 2 instances where no race information was given, and 9 at least occurrences involved an unarmed individual. 

# In[ ]:


unarmed_youth_df = df.loc[(df['age'] < 18) & ((df['armed'] == 'unarmed') | (df['armed'] == 'toy weapon'))]
unarmed_youth_df.shape


# In[ ]:


sns.countplot(unarmed_youth_df['race'],hue=unarmed_youth_df['threat_level'],palette='bright')


# In[ ]:


sns.countplot(unarmed_youth_df.loc[unarmed_youth_df['armed']=='unarmed']['race'],palette='bright')


# <br>
# 
# </br>
# 
# **18-30 age group**

# In[ ]:


df.loc[(df['age'] < 31) & (df['age'] > 17)].race.value_counts()


# In[ ]:


df.loc[(df['age'] < 31) & (df['age'] > 17)].armed.value_counts()


# In[ ]:


unarmed_ya_df = df.loc[((df['age'] < 31) & (df['age'] > 17)) & ((df['armed']=='unarmed') | (df['armed']=='toy weapon'))]
unarmed_ya_df.shape


# In[ ]:


unarmed_ya_df.flee.value_counts()


# In[ ]:


sns.countplot(unarmed_ya_df['race'],palette='bright')


# In[ ]:


sns.countplot(unarmed_ya_df[unarmed_ya_df['armed']=='unarmed']['race'],palette='bright')


# In[ ]:


sns.countplot(unarmed_ya_df.loc[(unarmed_ya_df['flee']=='Not fleeing') & (unarmed_ya_df['armed']=='unarmed')]['race'],palette='bright')


# In[ ]:


df.loc[df.body_camera==True].race.value_counts()


# In[ ]:


df.loc[df.body_camera==True].armed.value_counts()


# In[ ]:


sns.countplot(y='state',data=df,palette='Blues_d')


# In[ ]:




