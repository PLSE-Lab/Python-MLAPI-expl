#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os


# In[ ]:


#Read Data
url = 'https://drive.google.com/file/d/1M5fEGqDjHJL3kF0QHkpXYM6Q9or674GG/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)


# I have added three columns in the csv file namely-
# 1.Month of Posting
# 2.Day of Posting
# 3.Hour of Posting
# These were added using text and date function in excel on Crawl Timestamp column

# In[ ]:


df.head()


# **Analysing the Schedule for Job Posting**

# In[ ]:


fig = px.histogram(df, x="Day of Posting")
fig.show()


# In[ ]:


fig = px.histogram(df, x="Hour of Posting")
fig.show()


# Location wise Job Postings

# In[ ]:


location=df['Location'].value_counts().nlargest(n=10)
fig=px.bar(y=location,x=location.index, orientation='v',color=location)
fig.update_layout(width=800, 
                  showlegend=False, 
                  xaxis_title="City",
                  yaxis_title="Count",
                  title="Top 10 cities by job count")
fig.show()


# In[ ]:




