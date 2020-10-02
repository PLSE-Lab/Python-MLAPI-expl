#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


# In[ ]:


df = pd.read_csv('../input/cusersmarildownloadskawasakicsv/kawasaki.csv', sep=';')
df


# In[ ]:


# we have an unwanted string value in first numeric column
df['204252_at'][2]


# In[ ]:


# replace
df.loc[2,'204252_at'] = -0.1106154
# and explicitly convert to numeric
df['204252_at'] = pd.to_numeric(df['204252_at'])


# In[ ]:


df


# In[ ]:


df.describe()


# In[ ]:


df['204252_at'].hist()
plt.show()


# In[ ]:


df['211803_at'].hist()
plt.show()


# In[ ]:


df['211804_s_at'].hist()
plt.show()


# In[ ]:


# Scatter Matrix Plot
fig = px.scatter_matrix(df)
fig.show()


# In[ ]:


# 3D Scatter Plot
fig = px.scatter_3d(df, x='204252_at', y='211803_at', z='211804_s_at')
fig.show()

