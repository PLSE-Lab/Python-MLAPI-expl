#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns


# In[ ]:


df = sns.load_dataset("tips")


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.corr()


# # **CONTINUOUS PLOTS**

# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


sns.jointplot(x='tip', y='total_bill', data = df, kind='hex')


# In[ ]:


sns.jointplot(x='tip', y='total_bill', data=df, kind='reg') 


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.pairplot(df, hue = 'sex')


# In[ ]:


df['smoker'].value_counts()


# In[ ]:


sns.pairplot(df, hue = 'smoker')


# In[ ]:


sns.distplot(df['tip'])


# In[ ]:





# In[ ]:




