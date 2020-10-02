#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


emg = pd.read_csv('../input/911.csv')
emg.head(3)


# In[ ]:


emg.info()


# In[ ]:


emg['zip'].value_counts().head(5)


# In[ ]:


emg['twp'].value_counts().head(5)


# In[ ]:


emg['title'].nunique()


# _we have reason and department in the title column lets seperate this value and make an new column called Resone. This should have three different reason Fire, Trafffic and EMS_ .

# In[ ]:


emg['Reason'] = emg['title'].apply(lambda title: title.split(':')[0])


# In[ ]:


emg['Reason'].value_counts()


# The most common Reason for a 911 call based off EMS almost 210000 foloowd by Traffic 150000 and Fire 65000.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Reason',data=emg,palette='viridis')


# In[ ]:


emg.info()


# In[ ]:


emg['timeStamp'].iloc[0]


# In[ ]:


type(emg['timeStamp'].iloc[0])


# * timestamps are in strings formate so it needs to be changed to datetime formate Using pandas to_datetime function.

# In[ ]:


emg['timeStamp'] = pd.to_datetime(emg['timeStamp'])


# In[ ]:


type(emg['timeStamp'].iloc[0])


# In[ ]:


emg['Hour'] = emg['timeStamp'].apply(lambda timeStamp: timeStamp.hour)
emg['Day of week'] =emg['timeStamp'].apply(lambda timeStamp: timeStamp.dayofweek)
emg['Month'] = emg['timeStamp'].apply(lambda timeStamp: timeStamp.month)


# In[ ]:


emg.head()


# In[ ]:


dmap = { 0:'Mon', 1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[ ]:


emg['Day of week'] = emg['Day of week'].map(dmap)


# In[ ]:


emg['Day of week'].value_counts()


# In[ ]:


sns.countplot(x = 'Day of week', hue = 'Reason',data = emg, palette = 'viridis')
# moving the legend out of box
plt.legend(loc=2,borderaxespad=0.1,bbox_to_anchor =(1.05,1))


# In[ ]:


sns.countplot(x = 'Month', hue = 'Reason',data = emg, palette = 'viridis')
# moving the legend out of box
plt.legend(loc=2,borderaxespad=0.1,bbox_to_anchor =(1.05,1))


# 

# In[ ]:


sns.set_style("darkgrid")
byMonth = emg.groupby('Month').count()
byMonth.head()


# In[ ]:


byMonth['lat'].plot()


# In[ ]:


sns.countplot(x = 'Month',data = emg, palette = 'viridis')
# moving the legend out of box
plt.legend(loc=2,borderaxespad=0.1,bbox_to_anchor =(1.05,1))


# In[ ]:


byMonth.reset_index()


# In[ ]:



sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# In[ ]:


emg['Date'] = emg['timeStamp'].apply(lambda timeStamp: timeStamp.date())
emg['Date'].head()


# In[ ]:


byDate = emg.groupby('Date').count()


# In[ ]:


byDate.head()


# In[ ]:


byDate['lat'].head(3)


# In[ ]:


byDate['lat'].plot()
plt.tight_layout()


# In[ ]:


emg[emg['Reason'] == 'Traffic'].groupby('Date').count()['lat'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[ ]:


emg[emg['Reason'] == 'Fire'].groupby('Date').count()['lat'].plot()
plt.title('Fire')
plt.tight_layout()


# In[ ]:


emg[emg['Reason'] == 'EMS'].groupby('Date').count()['lat'].plot()
plt.title('EMS')
plt.tight_layout()


# In[ ]:


dayHour = emg.groupby(by=['Day of week','Hour']).count()['Reason'].unstack()
dayHour.head()
sns.set(style="ticks")
byDate = sns.load_dataset("byDate")
palette = dict (zip(byDate.zip.unique(), sns.color_palette("rocket_r",6)))
sns.relplot(x="lat", y="lng",
            hue="zip", size="timeStamp", col="desc",
            palette=palette,
            height=5, aspect=.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=bydate)


# ---- sns.lineplot(x="lat", y="lng", hue="zip", data= byDate)


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour, cmap='viridis')


# In[ ]:


sns.clustermap(dayHour, cmap='viridis')


# In[ ]:


dayMonth = emg.groupby(by=['Day of week','Month']).count()['Reason'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth, annot='',fmt='.2g', cmap='viridis')


# In[ ]:


#plt.figure(figsize=(15,8))
sns.clustermap(dayMonth, cmap="YlGnBu")


# In[ ]:




