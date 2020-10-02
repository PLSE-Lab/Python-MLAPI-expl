#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf

init_notebook_mode(connected=True)
cf.go_offline()


# For simiplicity we will focus only the global region.

# In[2]:


data = pd.read_csv('../input/data.csv', parse_dates=['Date'])
data = data.loc[data['Region'] == 'global']


# In[3]:


data.columns = (data.columns.str.lower()
                .str.replace(' ', '_'))

data.describe(include='all')


# Forecasting number of track streams is complicated by the fact the time series data for a given track is irregularly spaced. This happens because a track inclusion in the dataset is conditional on being in the daily top 200.

# In[4]:


track_counts = data['track_name'].value_counts()
track = track_counts.index[-700]
(data.loc[data['track_name'] == track, ('date', 'streams')]
 .set_index('date')
 .iplot(kind='bar',
        yTitle='# Streams',
        title=track))


# Because of these gaps, it makes sense to start with calculating the probability of a song to be included in the daily top 200 given that it was included yesterday. Lets denote by $S_t$ the set of songs included in the daily top 200 for day $t$.
# 
# $$P \left(s \in S_{t+1} \mid s \in S_{t} \right) = 
# \frac{P(s \in S_{t+1} \cap s \in S_{t})} {P(s \in S_{t})} =
# \frac{\left\vert{ S_{t+1} \cap S_{t}  }\right\vert} {\left\vert{ S_{t}  }\right\vert } = 
# \frac{\left\vert{ S_{t+1} \cap S_{t}  }\right\vert} { 200 } $$

# In[5]:


TOP_N = 200

data = data.sort_values(['artist', 'track_name', 'date'])
data['next_date'] = data.groupby(['artist', 'track_name'])['date'].shift(-1)

data['in_next_day'] = (data['next_date'] - data['date']).dt.days == 1
probabilities = data.groupby('date')['in_next_day'].sum().divide(TOP_N)

probabilities.iplot(title='Conditional Probability')


# It seems like we have missing data for some reason. Also probabilities dip with strange regularity on Thursdays.

# In[ ]:




