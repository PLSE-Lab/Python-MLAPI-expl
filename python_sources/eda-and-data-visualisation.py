#!/usr/bin/env python
# coding: utf-8

# ###Hi I tried to apply EDA and Data Visualisation.
# ###Kindly upvote this kernel if you find it worthwhile!
# ###Thanks

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


season_data=pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Season.csv')
ball_data=pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Ball_by_Ball.csv')
player_data=pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Player.csv')
team_data=pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Team.csv')
match_data=pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Match.csv')
pl_match_data=pd.read_csv('/kaggle/input/indian-premier-league-csv-dataset/Player_Match.csv')


# In[ ]:


season_data.head(5)


# In[ ]:


season_data.dtypes


# In[ ]:


ball_data.head(5)


# In[ ]:


ball_data.dtypes


# In[ ]:


player_data.head(5)


# In[ ]:


player_data.dtypes


# In[ ]:


team_data.head(5)


# In[ ]:


team_data.dtypes


# In[ ]:


match_data.head(5)


# In[ ]:


match_data.dtypes


# In[ ]:


pl_match_data.head(5)


# In[ ]:


pl_match_data.describe(include='all')


# In[ ]:


pl_match_data.dtypes


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Season_Id',data=match_data)


# In[ ]:


match_data.describe(include='all')


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(y='Venue_Name',data=match_data.head(10))


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(data=season_data,annot=True)


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(a=season_data['Man_of_the_Series_Id'],kde=False)


# In[ ]:


plt.figure(figsize=(10,6))
sns.kdeplot(data=match_data['Match_Id'],shade=True)


# In[ ]:


plt.figure(figsize=(10,6))
sns.lineplot(data=season_data)

