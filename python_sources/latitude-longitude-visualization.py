#!/usr/bin/env python
# coding: utf-8

# ![](http://i.imgur.com/W2A7GS2.png)

# **Hello Kaggle world, this is Leela Kishan Kolla.
# The following is a Visualization of the data set of Airbnb. the final plot shows the detailed points on the map of Singapore.**

# Importing required packages

# In[ ]:


import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#reading the File
data = pd.read_csv("/kaggle/input/singapore-airbnb/listings.csv")


# In[ ]:


#This shows top 5 rows form the table
data.head(5)


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


#This shows the number of null values in every column
data.isnull().sum()


# Droping the columns with many null values

# In[ ]:


df1 = data.drop(columns="last_review")


# In[ ]:


df2 = data.drop(columns="reviews_per_month")


# In[ ]:


df2.isnull().sum()


# In[ ]:


df2


# In[ ]:



fig = px.scatter_mapbox(df2, lat="latitude", lon="longitude", hover_name="name", hover_data=["host_name", "room_type", "minimum_nights"], zoom=10, height=600,color="room_type", size="minimum_nights",
                  color_continuous_scale=px.colors.cyclical, size_max=15)
#color_discrete_sequence=["green"]
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:




