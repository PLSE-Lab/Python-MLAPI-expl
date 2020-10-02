#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import os

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm


# # Load dataset

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


path_data = Path('/kaggle/input/covid19-global-forecasting-week-1/')

df_train = pd.read_csv(path_data / 'train.csv', parse_dates=['Date'])
df_test  = pd.read_csv(path_data / 'test.csv',  parse_dates=['Date'])


# In[ ]:


df_train


# In[ ]:


df_test


# # Plot on Map, datetime animation

# ## Pre-process

# In[ ]:


df_train['Lat,Long'] = df_train['Lat'].astype(str) + ',' + df_train['Long'].astype(str)
df_test[ 'Lat,Long'] = df_test[ 'Lat'].astype(str) + ',' + df_test[ 'Long'].astype(str)


# In[ ]:


df_train.head()


# ## Plot simply

# In[ ]:


list_tmp = []

for date in tqdm(sorted(df_train['Date'].unique().tolist())):
    df_train_tmp                 = df_train[df_train['Date'] <= pd.to_datetime(date)]
    df_grouped                   = df_train_tmp.groupby('Lat,Long')
    df_tmp                       = df_grouped.mean()[['Lat', 'Long']]
    df_tmp['log_ConfirmedCases'] = df_grouped.max()['ConfirmedCases']
    df_tmp['Date']               = str(pd.to_datetime(date)).split(' ')[0]
    list_tmp.append(df_tmp)

fig = px.scatter_geo(pd.concat(list_tmp), lat='Lat', lon='Long', size='log_ConfirmedCases', projection='natural earth', animation_frame='Date')
fig.show()


# ## Plot log-scaled
# 
# Use log(1+X) scaling

# In[ ]:


list_tmp = []

for date in tqdm(sorted(df_train['Date'].unique().tolist())):
    df_train_tmp                 = df_train[df_train['Date'] <= pd.to_datetime(date)]
    df_grouped                   = df_train_tmp.groupby('Lat,Long')
    df_tmp                       = df_grouped.mean()[['Lat', 'Long']]
    df_tmp['log_ConfirmedCases'] = np.log(1+df_grouped.max()['ConfirmedCases'])
    df_tmp['Date']               = str(pd.to_datetime(date)).split(' ')[0]
    list_tmp.append(df_tmp)

fig = px.scatter_geo(pd.concat(list_tmp), lat='Lat', lon='Long', size='log_ConfirmedCases', projection='natural earth', animation_frame='Date')
fig.show()

