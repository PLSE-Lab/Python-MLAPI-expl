#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/GDP_capita_spanish_regions.csv')


# In[ ]:


df.head()


# In[ ]:


df.plot()


# In[ ]:


df = df.transpose()


# In[ ]:


df = df.rename(columns={0:'anda',1:'arag',2:'astu',3:'bale',4:'cana',5:'cant',6:'casl',7:'casm',8:'cata',9:'vale',10:'extr',11:'gali',12:'madr',13:'murc',14:'nava',15:'eusk',16:'rioj',17:'ceut',18:'meli',19:'SPAIN'})


# In[ ]:


df.shape


# In[ ]:


df = df[1:]


# In[ ]:


df.head()


# In[ ]:


year = 2010
df['SPAIN'][year-2000] 


# In[ ]:


df['eusk'][year-2000]


# In[ ]:


df['anda'][year-2000]


# In[ ]:


df.mean().sort_values()


# In[ ]:


df.max().sort_values()


# In[ ]:


df.min().sort_values()


# In[ ]:


dforder = df.median().sort_values()
dforder


# In[ ]:


sorted_df = df[['extr','anda','meli','casm','murc','ceut','cana','gali','vale','astu','cant','casl','SPAIN','bale','rioj','arag','cata','nava','eusk','madr']]


# In[ ]:


sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.5})


# In[ ]:


s = sns.pointplot(data=sorted_df)
for item in s.get_xticklabels():
    item.set_rotation(60)


# In[ ]:


g = sns.stripplot(data=sorted_df)
for item in g.get_xticklabels():
    item.set_rotation(60)


# In[ ]:


ax = sns.violinplot(sorted_df)
for item in ax.get_xticklabels():
    item.set_rotation(60)


# In[ ]:




