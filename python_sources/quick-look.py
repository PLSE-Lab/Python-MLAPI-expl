#!/usr/bin/env python
# coding: utf-8

# #ISIS in Social Media - Twitter
# Here is a quick look of the dataset.  I'll update more findings soon.
# 
# There seems to be issues with the file location sometimes...

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as plt

from matplotlib import style
plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/tweets.csv', parse_dates=['time'], infer_datetime_format=True)


# ###Top 10 Contenders in the Dataset

# In[ ]:


top_handles = data.username.value_counts().sort_values(ascending=False)
top_handles.head(10).plot.bar(title='Top 10 Twitter Handles',
                            figsize=(16,8))


# ###Tweets over time

# In[ ]:


data.time.value_counts().plot(title='ISIS related tweets over time',
                             figsize=(16,8))


# In[ ]:


data.time.value_counts().sort_index().cumsum().plot(title='Total number of ISIS related tweets over time',
                                                   figsize=(16,8))

