#!/usr/bin/env python
# coding: utf-8

# ### As several people have mentioned, it does seem as 'time' represent minutes. Let's see whether there are some trend or/and seasonality in 'accuracy'.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

def dateTimeMaker(times):
    return datetime.datetime(year=2000, month=1, day=1) + datetime.timedelta(minutes=int(times))
# 01-01-2000 is an arbitrary date


# In[ ]:


get_ipython().run_cell_magic('time', '', "df= pd.read_csv('../input/train.csv')\ndf['datetime']=df['time'].apply(dateTimeMaker)\ndf['date']=df['datetime'].dt.date\ndf['minutes']=df['datetime'].dt.minute\ndf['hours']=df['datetime'].dt.hour\ndf['weekday']=df['datetime'].dt.weekday")


# In[ ]:


f, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
for i, param in enumerate('date,minutes,hours,weekday'.split(',')):
    row,col=divmod(i,2)[0],divmod(i,2)[1]
    grouped = pd.groupby(df[['accuracy',param]],by=param)[['accuracy']].mean()
    axes[row,col].plot(grouped.index, grouped['accuracy'])
    axes[row,col].set_ylim([40, 100]) 
    axes[row,col].set_xlabel(param,size=15)
    axes[row,0].set_ylabel("accuracy",size=15)


# ### It looks like 'accuracy' have some trend and maybe seasonality. Not sure how to interpret it yet, but it's something to consider.
