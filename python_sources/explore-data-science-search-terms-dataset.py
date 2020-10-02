#!/usr/bin/env python
# coding: utf-8

# # Explore Data Science Search Terms Dataset
# 
# The numbers in this [dataset](https://www.kaggle.com/leonardopena/search-growth-for-data-science-terms) are scaled such that the maximum value for each search term is 100.

# In[ ]:


import pandas as pd
import plotly.express as px
dataframe = pd.read_csv('/kaggle/input/search-growth-for-data-science-terms/data.csv').drop('Unnamed: 0',axis=1)
dataframe.plot(title='A Quick Preview of the Data',figsize=(12,9), grid=True)


# In[ ]:


title = 'Popularity of 3 Data Science Terms Over Time'
yaxis_title = 'Relative Number of Google Searches Per Week (0-100)'
df_melt = dataframe.melt(id_vars='week', value_vars=['artificial intelligence','data science','machine learning'])
fig = px.line(df_melt, x="week", y="value", color="variable",title=title).update(layout=dict(xaxis_title='Date',yaxis_title=yaxis_title,legend_orientation="h",showlegend=True))
fig.show()


# In[ ]:


title = 'Popularity of 3 Data Science Languages Over Time'
yaxis_title = 'Relative Number of Google Searches Per Week (0-100)'
df_melt = dataframe.melt(id_vars='week', value_vars=['python', 'R','sql'])
fig = px.line(df_melt, x="week", y="value", color="variable",title=title).update(layout=dict(xaxis_title='Date',yaxis_title=yaxis_title,legend_orientation="h",showlegend=True))
fig.show()


# In[ ]:


title = 'Python and Data Science go Hand in Hand'
yaxis_title = 'Relative Number of Google Searches Per Week (0-100)'
df_melt = dataframe.melt(id_vars='week', value_vars=['data science', 'python'])
fig = px.line(df_melt, x="week", y="value", color="variable",title=title).update(layout=dict(xaxis_title='Date',yaxis_title=yaxis_title,legend_orientation="h",showlegend=True))
fig.show()


# In[ ]:




