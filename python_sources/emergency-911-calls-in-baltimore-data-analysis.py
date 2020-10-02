#!/usr/bin/env python
# coding: utf-8

# ## Breakdown of the project:
# 
# 1. **Loading the dataset:** Using pandas load the data file(.csv)
# 2. **Data Visualization:** Creating plots to find relations between the features.
# 3. **Insights and Inferences from results:** Using heat maps and plots to infer the relationships between priority of the call and time and date of the call.
# 
# This project can also be used to find more relations than the ones in this notebook.There could be a relationship between an area and a description.Feel free to use the code.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/911_calls_for_service.csv')
df.head()


# What is the most common priority type for a 911 call based off of this new column?

# In[ ]:


df['priority'].value_counts()


# In[ ]:


import seaborn as sns
sns.countplot(x='priority',data=df,palette='viridis')


# Now let us begin to focus on time information

# In[ ]:


df['callDateTime'].iloc[0]


# In[ ]:


df['callDateTime']=pd.to_datetime(df['callDateTime'])


# In[ ]:


time=df['callDateTime'].iloc[0]
df['Hour']=df['callDateTime'].apply(lambda time:time.hour)
df['Month']=df['callDateTime'].apply(lambda time:time.month)
df['Day of Week']=df['callDateTime'].apply(lambda time:time.dayofweek)


# In[ ]:


dmap={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week']=df['Day of Week'].map(dmap)


# In[ ]:


sns.countplot(x='Day of Week',data=df,hue='priority',palette='viridis')


# In[ ]:


sns.countplot(x='Month',data=df,hue='priority',palette='viridis')


# In[ ]:


byMonth=df.groupby('Month').count()
byMonth.head()


# In[ ]:


byMonth['incidentLocation'].plot()


# In[ ]:


sns.lmplot(x='Month',y='incidentLocation',data=byMonth.reset_index())


# In[ ]:


df['Date']=df['callDateTime'].apply(lambda p:p.date())


# In[ ]:


df.groupby('Date').count()['incidentLocation'].plot()
plt.tight_layout()


# In[ ]:


df[df['priority']=='Medium'].groupby('Date').count()['incidentLocation'].plot()
plt.title('Medium')
plt.tight_layout()


# In[ ]:


df[df['priority']=='Low'].groupby('Date').count()['incidentLocation'].plot()
plt.title('Low')
plt.tight_layout()


# In[ ]:


df[df['priority']=='High'].groupby('Date').count()['incidentLocation'].plot()
plt.title('High')
plt.tight_layout()


# In[ ]:


df[df['priority']=='Non-Emergency'].groupby('Date').count()['incidentLocation'].plot()
plt.title('Non-Emergency')
plt.tight_layout()


# In[ ]:


df[df['priority']=='Emergency'].groupby('Date').count()['incidentLocation'].plot()
plt.title('Emergency')
plt.tight_layout()


# In[ ]:


df[df['priority']=='Out of Service'].groupby('Date').count()['incidentLocation'].plot()
plt.title('Out of Service')
plt.tight_layout()


# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['priority'].unstack()
dayHour.head()


# In[ ]:


sns.heatmap(dayHour,cmap='viridis')


# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['priority'].unstack()
dayMonth.head()


# In[ ]:


sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# Feel free to contact me and ask questions. If this helped you in any way upvote :)
