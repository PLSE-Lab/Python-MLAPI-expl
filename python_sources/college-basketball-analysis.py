#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/college-basketball-dataset/cbb.csv')


# In[ ]:


data


# Add new column 'cw' odds of winning.

# In[ ]:


data['cw'] = data['W']/data['G']


# General information

# In[ ]:


data.info()


# Columns name. Lead to a convenient view

# In[ ]:


data.columns =data.columns.str.lower()
data.columns


# In[ ]:


data['postseason'] = data['postseason'].fillna('Not playoff')


# **Task Details :** 
# Figure out the top 10 teams according to the number of games won.

# In[ ]:


data.pivot_table(index='team',values = 'w',
                 aggfunc='max').reset_index().sort_values(by='w'
                    ,ascending=False).head(10).plot.bar('team',figsize=[10,5],color='red',alpha=0.35)


# Top 10 Highest AOE

# In[ ]:


data['adjoe'].hist(bins=50,figsize=[10,7],color='red',alpha=0.35)


# In[ ]:


data[data['postseason']=='Not_playoff']


# In[ ]:


data.pivot_table(index='team',values = 'adjoe',aggfunc='max').reset_index().sort_values(by='adjoe',ascending=False).head(10)


# **Top 10 Highest ADE**

# In[ ]:


data.pivot_table(index='team',values = 'adjde',aggfunc='max').reset_index().sort_values(by='adjde',ascending=False).head(10)


# Strongly different composition of the top 10 teams on attack and defense, further it is clearly shown that the winning strategy is highly dependent on the chosen tactics, for example, an attacking game on the result gives more advantages than defense.

# In[ ]:



data.pivot_table(index='year',columns='conf',values = 'w'
                 ,aggfunc='sum').boxplot(figsize=[15,7]).set_title('Multiple average annual conference winnings.')
plt.ylim(50,300)


# In[ ]:


data['barthag'].hist(bins=80,figsize=[10,5],color='red',alpha=0.25)


# #### Let's see how the overall team has evolved over the years.

# In[ ]:


data.pivot_table(index='year',values = 'cw').plot.line(figsize=[10,2],color='red',alpha=0.35)


# The abrupt jump occurred after 2016. We need to look closely at this period.

# In[ ]:


data.query('year in [2016,2017]').pivot_table(index='year',values=['adjoe','adjde'],aggfunc=['mean','median']).reset_index()


# It doesn't make any difference. Let's see how many games we've played and how many winning games we've won.

# In[ ]:


data.query('year in [2016,2017]').pivot_table(index='year',values=['g','w','cw']).reset_index()


# And so we found the reason, the average number of games won in 2017 season increased, we have a clear leader? Or a few leaders?

# ### Champions

# In[ ]:


data.query('postseason=="Champions"').sort_values(by='year')


# In[ ]:


data.pivot_table(index = 'postseason'
        ,values = 'cw').sort_values('cw',
        ascending='postseason').plot.barh(fill=True,figsize=[10,2],alpha=0.35,color='red',grid=True)
data.pivot_table(index = 'postseason'
        ,values = 'adjoe').sort_values('adjoe',
        ascending='postseason').plot.barh(fill=True,figsize=[10,2],alpha=0.35,color='red',grid=True)
data.pivot_table(index = 'postseason'
        ,values = 'adjde').sort_values('adjde',
        ascending='postseason').plot.barh(fill=True,figsize=[10,2],alpha=0.35,color='red',grid=True)


#     From these three charts we can see that the formula for success is a greater emphasis on the attack. On the other hand, the third graph shows that those teams that bet on a defensive game cannot achieve good results.

# Let's look at the weakest teams.

# In[ ]:


data.boxplot(['barthag','cw'])

