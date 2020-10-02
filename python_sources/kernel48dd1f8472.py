#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv('/kaggle/input/la-liga-dataset/LaLiga_dataset.csv')
df.head()


# In[ ]:


df.set_index('season',inplace=True)


# In[ ]:


twenty_sixteen=df[888:]


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
twenty_sixteen


# In[ ]:


plt.figure(figsize=(16,4))
sns.barplot(y='points',x='club',data=twenty_sixteen)


# In[ ]:


plt.figure(figsize=(16,4))
sns.barplot(y='goals_scored',x='club',data=twenty_sixteen)


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x=twenty_sixteen['points']/twenty_sixteen['goals_scored'],y=twenty_sixteen['club'])


# **Even though Alaves scored substantially fewer goals compared to the teams around them and even lesser than 2 of the relegated teams 
# They have a much larger points value to each goal they score which shows how much they make each goal count.**

# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(y=twenty_sixteen['goals_conceded'],x=twenty_sixteen['club'])


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x=twenty_sixteen['points']/twenty_sixteen['goals_scored'],y=twenty_sixteen['club'])


# **As we can see Alaves had to score only 0.75 goal to earn a point while the relegated teams Osasuna had to score more than 1.75,
# As we can predict from here that Alaves had a mean defence and so did Atletico Madrid.**

# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x=twenty_sixteen['goals_conceded']/twenty_sixteen['points'],y=twenty_sixteen['club'])


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x=twenty_sixteen['goals_scored']/twenty_sixteen['goals_conceded'],y=twenty_sixteen['club'])


# In[ ]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='home_win',y='away_win',data=twenty_sixteen[-6:],label='TOP 6')
sns.scatterplot(x='home_win',y='away_win',data=twenty_sixteen[4:13],label='Rest of the teams')
sns.scatterplot(x='home_win',y='away_win',data=twenty_sixteen[:3],label='Relegated Teams')
plt.legend()


# In[ ]:


plt.figure(figsize=(6,5))
sns.scatterplot(y='goals_scored',x='goals_conceded',data=twenty_sixteen[-6:],label='TOP 6')
sns.scatterplot(y='goals_scored',x='goals_conceded',data=twenty_sixteen[4:13],label='Rest of the teams')
sns.scatterplot(y='goals_scored',x='goals_conceded',data=twenty_sixteen[:3],label='Relegated Teams')
plt.legend()


# In[ ]:


h=df.groupby('club')['points'].agg(['sum','max','min','mean'])
from collections import Counter


# In[ ]:


s=Counter(df['club'])


# In[ ]:


u=[]
for x in h.index:
    u.append(s[x])
h['appearances']=u


# In[ ]:



h.columns=['Total Points','Max Points Earned in a Season','Min Points Earned in a Season','Average points earnt in a season','Number of times played in a la liga season']


# In[ ]:


h


# In[ ]:




