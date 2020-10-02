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
import seaborn as sns
import matplotlib.pyplot as plt

#plotly
import plotly.plotly as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio
init_notebook_mode(connected=True)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Load the data

# In[ ]:


df = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')


# Here I am using 'ISO-8859-1' encoding to load the dataset beacuse 'utf-8' codec can't decode byte 0xe9 in position 18'

# ## Some insights and basic statistics about the dataset

# In[ ]:


df.head()  #gives us top 5 rows of the dataframe


# In[ ]:


df.describe()


# In[ ]:


df.isnull().count()


# ## Finding missing values

# In[ ]:


df.isnull().sum() #gives us total no. of missing values for all the columns


# **Let us create dataframe to store the information about missing values for future reference**

# In[ ]:


totalms = df.isnull().sum().sort_values(ascending=False)
percentagems = (df.isnull().sum()/ df.isnull().count()).sort_values(ascending=False)
missingdata = pd.concat([totalms, percentagems], axis=1, keys=['Total', 'Percentage'])
missingdata.head()


# In[ ]:


print(df.shape)


# In[ ]:


print(missingdata.shape)


# In[ ]:


print(missingdata)


# **From this we can drop the columns with high percentage of missing values. But as off now I am not dropping those columns from df(main dataframe)**

# ## Univariate analysis/visualization

# **Let start from the left most column.** 

# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
ax = sns.countplot(x='iyear', palette='GnBu_d', data=df, orient="v")
_ = plt.xlabel('Years')
_ = plt.setp(ax.get_xticklabels(), rotation = 90)


# Here we can see year wise terrorist attacks starting from 1970 till 2017. Number of attacks were steadily increasing till 1992 and there was sudden drop from 1994 to 2004. Year 2014 witnessed the highest no. of attacks. 

# In[ ]:


df['country_txt'].value_counts()


# **Let us see how each country suffered**

# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax = sns.countplot(x='country_txt', data=df, order=df['country_txt'].value_counts()[:15].index, palette='inferno')
_ = plt.xlabel('Countries')
_ = plt.setp(ax.get_xticklabels(), rotation = 90)


# **Displaying to 15 countries most affected by terror attacks. Iraq suffered the most among all the countries followed by Pakistan, Afganistan & India. From this we can conclude that south, east Asia is high affected region.** 

# **The dataset is divided into the following regions**

# In[ ]:


df['region_txt'].value_counts()


# In[ ]:


df['region_txt'].isna().sum()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax = sns.countplot(x='region_txt', data=df, palette='plasma', order=df['region_txt'].value_counts().index)
_ = plt.xlabel('Region')
_ = plt.setp(ax.get_xticklabels(), rotation = 60)


# **Middle east and north africa together suffered as nearly as south asia**

# In[ ]:


terror_region = pd.crosstab(df['iyear'], df['region_txt'])
terror_region.plot(color=sns.color_palette('viridis', 12))
fig = plt.gcf()
fig.set_size_inches(10,6)
#use plotly


# Here we can see the trend in attacks over the years over different regions. We can see the spike in dark green line which represents middle east and north africa region, suffered most during the last decade followed by south asia. 

# In[ ]:


print(terror_region.head())


# **Successful attacks**

# In[ ]:


sns.countplot(x='success', data=df, palette='hls')


# Here 1 represents succesful attack and 0 represents failes attempt. From the above figure we can see that most of the attacks are successful.

# In[ ]:


sns.countplot(x='suicide', data=df, palette='twilight')


# In[ ]:


df['attacktype1_txt'].isnull().sum()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax = sns.countplot(x='attacktype1_txt', data=df, palette='plasma_r', order=df['attacktype1_txt'].value_counts().index)
_ = plt.xlabel('AttackType')
_ = plt.setp(ax.get_xticklabels(), rotation = 75)


# So, most of the attacks are bombing/explosion as it has highest range among all options, with equally high damaging. 

# In[ ]:


terror_attack = pd.crosstab(df['iyear'], df['attacktype1_txt'])
terror_attack.plot(color = sns.color_palette('Set3', 9))
fig = plt.gcf()
fig.set_size_inches(10, 6)


#    We can see from the above figure that bombing/explosion is the go to method for terror attacks over the years followed by armed assualt. 

# In[ ]:


df['attacktype2_txt'].value_counts()


# In[ ]:


df['attacktype3_txt'].value_counts()


# In[ ]:


df['attacktype1_txt'].isna().sum()


# In[ ]:


percentage_missing = (df['attacktype2_txt'].isna().sum() / df['attacktype2_txt'].isna().count()) * 100.00
print(percentage_missing)


# In[ ]:


percentage_missing2 = (df['attacktype3_txt'].isna().sum() / df['attacktype3_txt'].isna().count()) * 100.00
print(percentage_missing2)


#     attacktype2 & attacktype3 is not defined properly, at the same time both of these attributes hold high percentage of missing values.

# In[ ]:


df['targtype1_txt'].value_counts()


# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax = sns.countplot(x = 'targtype1_txt', data=df, palette='icefire', order=df['targtype1_txt'].value_counts().index)
_ = plt.xlabel('Targets of attack')
_ = plt.setp(ax.get_xticklabels(), rotation = 90)


# From the above figure we can see that private citizens and property is the highest target and the top five targets are private citizens and property, military, police, government(general) where people can be found in high numbers and also law & order is affected.  

# In[ ]:


df['weaptype1_txt'].value_counts()


# In[ ]:


df['weaptype1_txt'].isna().sum()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 5))
ax = sns.countplot(x='weaptype1_txt', data=df, palette='inferno', order=df['weaptype1_txt'].value_counts().index)
_ = plt.xlabel('Weapon Used')
_ = plt.setp(ax.get_xticklabels(), rotation = 90)


# As we have already seen that bombing is the goto method for attacks explosives are 1st choice of weapon.

# In[ ]:


df['gname'].value_counts()[0:1]


# **Here gname is terrorist group name**.  Here we can see that unknown group/groups has/have caused the highest damage. There can be indivisuals, small unorganized group, separatist etc under this "unknow group" umbrella or known groups but didn't claim the attack.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax = sns.countplot(x='gname', data=df, palette='inferno', order=df['gname'].value_counts()[1:16].index)
_ = plt.xlabel('Terrorist group name')
_ = plt.setp(ax.get_xticklabels(), rotation=90) 


# **Here we can see the top 15 terrerist groups excluding the highest unknown **

# ## Bivariate analysis/visualization

# In[ ]:


df['casualities'] = df['nkill'] + df['nwound']


# In[ ]:


df_country_cas = pd.concat([df['country_txt'], df['casualities']], axis=1)
df3 = pd.DataFrame(df_country_cas.groupby('country_txt').sum().reset_index())
df3.head()


# In[ ]:


print(df3.shape)


# In[ ]:


x = df3['country_txt']
y = df3['casualities']
sz = 10
colors = np.random.randn(205)
fig = go.Figure()
fig.add_scatter(
    x = x,
    y = y, 
    mode = 'markers',
    marker={
        'size':sz,
        'color':colors,
        'opacity':0.6,
        'colorscale':'Viridis'
    });
iplot(fig)


# In[ ]:


missingdata.loc['nkill', :]


# In[ ]:


missingdata.loc['nwound', :]


# In[ ]:


df_year_kill = pd.concat([df['iyear'], df['nkill']], axis=1)


# In[ ]:


df2 = pd.DataFrame(df_year_kill.groupby('iyear').sum().reset_index())
df2.head()


# In[ ]:


print(df2.shape)


# In[ ]:


df_year_wound = pd.concat([df['iyear'], df['nwound']], axis=1)
df3 = pd.DataFrame(df_year_wound.groupby('iyear').sum().reset_index())
df3.head()


# In[ ]:



x = df2['iyear']
y = df2['nkill']
colors = np.random.randn(47)
sz = 15
fig = go.Figure()
fig.add_scatter(
    x = x, 
    y = y, 
    mode = 'markers', 
    marker = {
        'size':sz,
        'color':colors,
        'opacity':0.6,
        'colorscale':'Viridis'
    });
iplot(fig)


# The figure above demonstrates total no. of people killed each year due to terror attacks from 1970 to 2017   

# In[ ]:


x1 = df3['iyear']
y1 = df3['nwound']
colors = np.random.randn(47)
sz = 15
fig = go.Figure()
fig.add_scatter(
    x = x1, 
    y = y1, 
    mode = 'markers',
    marker = {
        'size':sz,
        'color':colors,
        'opacity':0.6,
        'colorscale':'Viridis'
    });
iplot(fig)


# The figure above demonstrates total no. of people wounded each year due to terror attacks from 1970 to 2017.
