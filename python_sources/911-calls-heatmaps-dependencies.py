#!/usr/bin/env python
# coding: utf-8

# ## Basic information about data

# ____
# ** Importing libraries, reading .csv file **

# In[ ]:


import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('../input/911.csv')
df.info()


# In[ ]:


df.head()


# ## Visualizations

# ** Countplot of 911 calls by Reason. **

# In[ ]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
sns.countplot(x='Reason',data=df,palette='viridis')


# ___
# ** Countplots of the day of week and month **

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df['Date']=df['timeStamp'].apply(lambda t: t.date())

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)


# In[ ]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ** Number of calls per month **

# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# ** Plots representing a Reason for the 911 call**

# In[ ]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# ____
# ** Heatmaps representing time and day of week influences on number of calls**

# In[ ]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# In[ ]:


sns.clustermap(dayHour,cmap='viridis')


# ** Heatmaps representing month and day of week influences on number of calls **

# In[ ]:


dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')


# In[ ]:


sns.clustermap(dayMonth,cmap='viridis')

