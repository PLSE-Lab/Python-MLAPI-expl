#!/usr/bin/env python
# coding: utf-8

# > The csv files in this dataset have a world of information but are not in an easily manipulatable format.
# 
# I've written a function that will hopefully make it a lot easier to use this data.

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


df = pd.read_csv('../input/2019_data.csv')


# The column Statistic has lots of possible stats with Variable having matching column headings that align with variable.
# 
# Below lists all the stats we can look into.

# In[ ]:


df.Statistic.value_counts()


# By using the function below by entering the dataframe and statistic you want to look at as variables you can create a new dataframe.

# In[ ]:


def convert_df(df, statistic):
    df = df[df.Statistic == statistic]

    df = pd.pivot_table(df, index=['Player Name','Date','Statistic'], 
                   columns = 'Variable', 
                   values='Value', 
                   aggfunc=[lambda x: ''.join(str(v) for v in x)]).reset_index()

    df.columns = df.columns.get_level_values(0)[:3].append(df.columns.get_level_values(1)[3:].map(lambda x:x.split('(')[-1]).str.replace(')','').str.title()).str.replace(' ','_')
    df.drop('Statistic', axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df
    


# Below is an example for looking into the Official World Golf Ranking. It has converted the date to datetime format. Some dates in the original datframe cover a period ie. Apr 14 - Apr 18 (2019). For this two columns are created for the start and end dates.

# In[ ]:


df_WR = convert_df(df, 'Official World Golf Ranking')
df_WR.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_WR.Avg_Points = pd.to_numeric(df_WR.Avg_Points)


# In[ ]:


df_WR.replace('nan',np.nan).head()


# In[ ]:


Top10 = df_WR[df_WR['Date']==df_WR.Date.max()].sort_values(by='Avg_Points', ascending=False)['Player_Name'].head(10).values


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))

columns = ['Player_Name','Date','Avg_Points']

ax = sns.lineplot(x="Date", y="Avg_Points", hue="Player_Name", 
                  data=df_WR[columns][df_WR.Player_Name.isin(Top10)])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Average Points for Current Top 10 Players")
plt.show()

