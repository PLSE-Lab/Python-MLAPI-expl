#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm

sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Data

# In[ ]:


path = '../input/nfl-playing-surface-analytics/'
pList = pd.read_csv(path + 'PlayList.csv')
iRecord = pd.read_csv(path + 'InjuryRecord.csv')
# pTrack = pd.read_csv(path + 'PlayerTrackData.csv')


# ## Data Exploration

# In[ ]:


pList.sample(4)


# In[ ]:


print('The information of the PlayList.csv'.center(50, '-'))
print('The rows and columns of the data are {}'.format(pList.shape))
Types = pList.dtypes
Total = pList.isnull().sum().sort_values(ascending = False)
Percentage = Total / pList.shape[0]
pd.concat([Total, Percentage, Types], axis = 1, keys = ['Total', 'Percentage', 'Types']).sort_values(by = ['Total'], ascending = False)


# In[ ]:


iRecord.sample(4)


# In[ ]:


print('The information of the InjuryRecord.csv'.center(50, '-'))
print('The rows and columns of the data are {}'.format(iRecord.shape))
Types = iRecord.dtypes
Total = iRecord.isnull().sum().sort_values(ascending = False)
Percentage = Total / iRecord.shape[0]
pd.concat([Total, Percentage, Types], axis = 1, keys = ['Total', 'Percentage', 'Types']).sort_values(by = ['Total'], ascending = False)


# In[ ]:


# pTrack.sample(4)


# In[ ]:


# print('The information of the PlayerTrackData.csv'.center(50, '-'))
# print('The rows and columns of the data are {}'.format(pTrack.shape))
# Types = pTrack.dtypes
# Total = pTrack.isnull().sum().sort_values(ascending = False)
# Percentage = Total / pTrack.shape[0]
# pd.concat([Total, Percentage, Types], axis = 1, keys = ['Total', 'Percentage', 'Types']).sort_values(by = ['Total'], ascending = False)


# ### Injury Rcord Data Exploration

# In[ ]:


iRecord.sample(4)


# In[ ]:


iRecord.groupby(['BodyPart', 'Surface'])['DM_M1','DM_M7', 'DM_M28', 'DM_M42'].sum()


# In[ ]:


plt.figure(figsize = (12, 10))
ax1 = iRecord.groupby(['BodyPart', 'Surface'])['DM_M1','DM_M7', 'DM_M28', 'DM_M42'].sum().plot.bar()
plt.title('The Corr of Body Part and Surface')
plt.xlabel('Body Part , Surface')
plt.xticks(rotation = 45)
plt.ylabel('Count')


# In[ ]:


plt.figure(figsize = (12, 10))

sns.set_style('whitegrid')
ax2 = sns.catplot(x = 'Surface', hue = 'BodyPart', kind = 'count', data = iRecord, palette = 'mako')
ax2.set(title = 'Correlation between Surface and Body Part ', xlabel = 'Surface Types', ylabel = 'Count')


# ### PlayList Data Exploration

# In[ ]:


print(pList['PlayerKey'].nunique())
pList['PlayerKey'].unique().tolist()[: 10]


# In[ ]:


plt.figure(figsize = (12, 10))

sns.set_style('whitegrid')
ax3 = sns.catplot(x = 'FieldType', hue = 'StadiumType', kind = 'count', data = pList, palette = 'mako')
ax3.set(title = 'Correlation between FieldType and StadiumType ', xlabel = 'FieldType', ylabel = 'Count')


# In[ ]:


plt.figure(figsize=(55, 80))
sns.set_style('whitegrid')
ax4 = sns.catplot(y = 'Weather', kind = 'count', data = pList, palette = 'mako')
plt.show()


# In[ ]:


print(pList['StadiumType'].nunique())
pList['StadiumType'].unique().tolist()


# In[ ]:


def groupingStadiumType(data):
    """
    grouped the stadiumType to the indoor or outdoor.
    Params: data
    Return: indoor and outdoor
    """
    if data in ['Outdoor', 'Oudoor', 'Outdoors', 'Open', 'Outdoor Retr Roof-Open', 'Ourdoor', 'Outddors', 'Retr. Roof-Open', 'Open Roof', 'Domed, Open', 'Domed, open', 'Heinz Field',
 'Cloudy', 'Retr. Roof - Open', 'Outdor', 'Outside']:
        value = 'outdoor'
    else:
        value = 'indoor'
    return value


# In[ ]:


pList['groupedStadiumType'] = pList['StadiumType'].apply(groupingStadiumType)
pList['groupedStadiumType'].value_counts().to_frame()


# In[ ]:


plt.figure(figsize = (15, 10))
sns.set_style('whitegrid')
ax5 = sns.catplot(x = 'groupedStadiumType', kind = 'count', data = pList, palette = 'mako')
ax5.set(title = 'The grouped StadiumType data', xlabel = 'grouped StadiumType', ylabel = 'Count')
plt.show()


# In[ ]:


plt.figure(figsize = (12, 10))

sns.set_style('whitegrid')

ax6 = sns.violinplot(x = 'FieldType', y = 'Temperature', hue = 'groupedStadiumType', split = True, data = pList.loc[pList['Temperature'] > -500], palette = 'mako')
ax6.set(title = 'The tempeature of the Field Type and StadiumType')


# In[ ]:


print('The information of the playerlist dataset'.center(50, '-'))
print('The number of player list dataset {}'.format(pList.shape))
pList.sample(4)


# In[ ]:


print(pList['PlayType'].nunique())
pList['RosterPosition'].value_counts().sort_values(ascending = False)


# In[ ]:


dataToPlot = pList.groupby(['RosterPosition', 'PlayType', 'PlayerKey'])['PlayerDay'].count().reset_index()


# In[ ]:


dataToPlot


# In[ ]:


plt.figure(figsize = (42, 10))

sns.set_style('whitegrid')
ax7 = sns.catplot(x = 'RosterPosition', hue = 'PlayType', data = dataToPlot, kind = 'count', palette = 'mako')
ax7.set(title = 'The corr between the Play Type and Roster Position', xlabel = 'Roster Position', ylabel = 'Count')
plt.xticks(rotation = 45)
plt.show()

