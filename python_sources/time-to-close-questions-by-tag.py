#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import pylab
pylab.rcParams['figure.figsize'] = (20.0, 16.0)
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


questions_pd = pd.read_csv("../input/questions.csv", parse_dates=["ClosedDate", "CreationDate", "DeletionDate"])
questions_pd.head()


# In[ ]:


questions_pd.dtypes


# In[ ]:


questions_pd = questions_pd[pd.notnull(questions_pd.ClosedDate)]
questions_pd.head()


# In[ ]:


tags_pd = pd.read_csv("../input/question_tags.csv")
tags_pd.head()


# In[ ]:


questions_pd['closing_time'] = (questions_pd.ClosedDate - questions_pd.CreationDate)
questions_pd['days'] = questions_pd['closing_time'].apply(lambda x: float(x.days))
questions_pd['seconds'] = questions_pd['closing_time'].apply(lambda x: float(x.seconds))

questions_pd.head()


# In[ ]:


max(questions_pd.closing_time)


# In[ ]:


joined_df = questions_pd.merge(tags_pd, on="Id")
joined_df.head()


# In[ ]:


max_values = joined_df.groupby('Tag').days.aggregate(np.median)
max_values.sort_values(inplace=True)
max_values.tail(20).plot.barh()
plt.title("Top 20 tags by median days to close");


# In[ ]:


min_values = joined_df.groupby('Tag').seconds.aggregate(np.median)
min_values.sort_values(inplace=True)
min_values.head(100)


# In[ ]:




