#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


tit = sns.load_dataset('titanic')
tit.head()


# In[ ]:


sns.countplot(x = 'sex',data = tit)


# In[ ]:


sns.jointplot(x = 'survived',y = 'age',data= tit,kind= 'kde')


# In[ ]:


tit.dropna().head()


# In[ ]:


sns.distplot(tit['fare'])


# In[ ]:


sns.pairplot(tit,hue = 'sex')


# In[ ]:


sns.pairplot(tit,hue = 'class')


# In[ ]:


sns.pairplot(tit,hue = 'embark_town')


# In[ ]:


sns.boxplot(x = 'class',y = 'age',data = tit,hue = 'who')


# In[ ]:


sns.barplot(x = 'sex',y = 'fare',data = tit)


# In[ ]:


sns.jointplot(x = 'fare', y ='age',data = tit,kind = 'hex')


# In[ ]:


tit.head()


# In[ ]:


import pandas as pd


# In[ ]:


a = tit


# In[ ]:


a.head()


# In[ ]:


df = a[['survived','fare','age']]


# In[ ]:


df.head()


# In[ ]:


hm = df.corr()


# In[ ]:


sns.heatmap(hm)


# In[ ]:




