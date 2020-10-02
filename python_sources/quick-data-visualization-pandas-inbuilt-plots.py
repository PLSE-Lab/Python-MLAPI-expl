#!/usr/bin/env python
# coding: utf-8

# 

# # **DATA VISUALIZATION - PANDAS**

# ## **IMPORT LIBRARIES**

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# ## GET DATA

# In[ ]:


os.getcwd()
os.chdir("../input/")
df1 = pd.read_csv("df1", index_col = 0)
df1.head()

df2 = pd.read_csv("df2")
df2.head()


# ## **HISTOGRAM**

# In[ ]:


df1['A'].hist()
df1['A'].hist(bins=5)
df1['A'].plot(kind='hist')
df1['A'].plot.hist()


# ## **AREA AND BAR PLOT**

# In[ ]:


df2.plot.area()
df2.plot.bar(stacked=True)


# ## **LINE PLOT FOR TIME SERIES DATA**

# In[ ]:


df1.plot.line(x=df1.index, y='B')


# ## **SCATTER PLOT**

# In[ ]:


df1.plot.scatter(x='A', y='B', c='C',cmap='coolwarm')


# ## **BOX PLOT**

# In[ ]:


df2.plot.box()


# ## **HEXA PLOT**

# In[ ]:


df = pd.DataFrame(np.random.randn(1000,2), columns=['a','b'])
df.plot.hexbin('a', 'b', gridsize=25)


# ## **KDE PLOT**

# In[ ]:


df['a'].plot.kde()
df['a'].plot.density()

df2.plot.kde()


# 

# 

# 
