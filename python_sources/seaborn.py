#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns


# In[ ]:


df1 = sns.load_dataset("iris")
sns.pairplot(df1, hue="species")


# In[ ]:


df2 = sns.load_dataset('titanic')  
sns.pairplot(df1, hue="species")

