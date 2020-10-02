#!/usr/bin/env python
# coding: utf-8

# ## EDA for NASA Astronauts, 1959-Present ##

# ## Import library ##

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/astronauts.csv')


# In[ ]:


df.head(10)


# ### Which American astronaut has spent the most time in space?

# In[ ]:


df['Space Flight (hr)'].max()


# In[ ]:


df[df['Space Flight (hr)']>=12818].sort('Name').head(1)


# ###  What university has produced the most astronauts? 

# In[ ]:


df['Alma Mater'].value_counts().head(10)


# In[ ]:


df['Alma Mater'].value_counts().head(10).plot(kind='bar')


# ### What subject did the most astronauts major in at college? 

# In[ ]:


df['Graduate Major'].value_counts().head(10)


# In[ ]:


df['Graduate Major'].value_counts().head(10).plot(kind='bar')


# In[ ]:


df['Undergraduate Major'].value_counts().head(10)


# In[ ]:


df['Undergraduate Major'].value_counts().head(10).plot(kind='bar')


# ### Have most astronauts served in the military? 

# In[ ]:


df['Military Rank'].value_counts()


# In[ ]:


df['Military Rank'].value_counts().plot(kind='bar')


# In[ ]:


df['Military Rank'].value_counts().sum()


# ### Which Military branch? 

# In[ ]:


df['Military Branch'].value_counts().head(10)


# In[ ]:


df['Military Branch'].value_counts().head(10).plot(kind='bar')


# ### What rank did they achieve?

# In[ ]:


df['Military Rank'].value_counts()


# In[ ]:


df['Gender'].value_counts()


# In[ ]:


df['Gender'].value_counts().plot(kind='bar')

