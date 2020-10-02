#!/usr/bin/env python
# coding: utf-8

# # Google Play Store Analysis

# **Imports**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/googleplaystore.csv')
df = data.copy()
df.head()


# In[ ]:


df.info()


# **From this we can see the null values**

# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe(include='all')


# In[ ]:


df['Reviews'] = df['Reviews'].str.replace('3.0M', '3000000')


# In[ ]:


df['Reviews'] = df['Reviews'].astype(np.float)


# In[ ]:


df['Price'].unique()


# In[ ]:


df['Price In Dollors'] = df['Price']


# In[ ]:


df['Price In Dollors'] = df['Price In Dollors'].str.replace('Everyone', '$0')


# In[ ]:


df['Price In Dollors'] = df['Price In Dollors'].str.replace('$', '')


# In[ ]:


df['Price In Dollors'] = df['Price In Dollors'].astype(np.float)


# **Row number 10472 is shifted on the left size so i moved the row one step right**

# In[ ]:


df.loc[10472]


# In[ ]:


df.loc[10472] = df.loc[10472].shift(periods=1, axis=0)


# In[ ]:


df['Rating'] = df['Rating'].astype(np.float64)


# In[ ]:


df['Last Updated'] = pd.to_datetime(df['Last Updated'])


# In[ ]:


df['Installs'].unique()


# In[ ]:


df['Installs'] = df['Installs'].str.replace('+', '')


# In[ ]:


df['Installs'] = df['Installs'].str.replace(',', '')


# In[ ]:


df['Installs'] = df['Installs'].astype(np.int)


# In[ ]:


df['Type'].unique()


# In[ ]:


df['Type'].fillna(value='Free', inplace=True)


# In[ ]:


df['Content Rating'].unique()


# In[ ]:


df['Rating'].fillna(0, inplace=True)


# In[ ]:


df['Content Rating'].unique()


# In[ ]:


df['Content Rating'].fillna('Unrated', inplace=True, axis=0)


# In[ ]:


df['Current Ver'].fillna('Unknown', inplace=True, axis=0)


# In[ ]:


df['Android Ver'].fillna('Unknown', inplace=True, axis=0)


# # Visuallization

# In[ ]:


sns.set(style="ticks", color_codes=True, font_scale=1.5)


# In[ ]:


plt.figure(figsize=(5, 4))
sns.barplot(x='Type', y='Price In Dollors', ci=None, data=df);


# In[ ]:


plt.figure(figsize=(25, 8))
sns.barplot(x='Rating', y='Price In Dollors', data=df, ci=None);


# In[ ]:


df['Rating Size'] = ''
df.loc[(df['Rating']>=0.0) & (df['Rating']<=1.0), 'Rating Size'] = '0.0 - 1.0'
df.loc[(df['Rating']>=1.0) & (df['Rating']<=2.0), 'Rating Size'] = '1.0 - 2.0'
df.loc[(df['Rating']>=2.0) & (df['Rating']<=3.0), 'Rating Size'] = '2.0 - 3.0'
df.loc[(df['Rating']>=3.0) & (df['Rating']<=4.0), 'Rating Size'] = '3.0 - 4.0'
df.loc[(df['Rating']>=4.0) & (df['Rating']<=5.0), 'Rating Size'] = '4.0 - 5.0'
df.loc[(df['Rating']>=5.0) & (df['Rating']<=6.0), 'Rating Size'] = '5.0 - 6.0'
df.loc[(df['Rating']>=6.0) & (df['Rating']<=7.0), 'Rating Size'] = '6.0 - 7.0'
df.loc[(df['Rating']>=7.0) & (df['Rating']<=9.0), 'Rating Size'] = '7.0 - 8.0'
df.loc[df['Rating']>=9.0, 'Rating Size'] = '9.0+'


# In[ ]:


plt.figure(figsize=(25, 8))
sns.barplot(x='Rating Size', y='Price In Dollors', data=df, ci=None, order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+']);


# In[ ]:


plt.figure(figsize=(20, 8))
sns.lineplot(x='Rating Size', y='Price In Dollors', data=df);


# In[ ]:


plt.figure(figsize=(20, 8))
sns.relplot(x='Price In Dollors', y='Rating Size', hue='Type', data=df);


# In[ ]:


plt.figure(figsize=(15, 8))
sns.boxplot(x='Price In Dollors', y='Rating Size', hue='Type', order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+'], data=df);


# In[ ]:


rating = df.groupby('Rating Size')['Price In Dollors'].sum().reset_index()


# In[ ]:


plt.figure(figsize=(25, 8))
sns.barplot(x='Rating Size', y='Price In Dollors', data=rating, ci=None, order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+']);


# In[ ]:


plt.figure(figsize=(20, 8))
sns.lineplot(x='Rating Size', y='Price In Dollors', data=rating);


# In[ ]:


plt.figure(figsize=(20, 8))
sns.barplot(x='Content Rating', y='Price In Dollors', data=df, ci=None);


# In[ ]:


content_rating = df.groupby('Content Rating')['Price In Dollors'].sum().reset_index()


# In[ ]:


plt.figure(figsize=(20, 8))
sns.barplot(x='Content Rating', y='Price In Dollors', data=content_rating, ci=None);


# In[ ]:


plt.figure(figsize=(35, 45))
sns.barplot(x='Price In Dollors', y='Genres', data=df, ci=None);


# In[ ]:


genres = df.groupby('Genres')['Price In Dollors'].sum().reset_index()


# In[ ]:


ten_genres = genres.sort_values(by='Price In Dollors', ascending=False).reset_index(drop=True)
ten_genres.head()


# In[ ]:


plt.figure(figsize=(20, 8))
sns.barplot(x='Price In Dollors', y='Genres', data=ten_genres, ci=None, order=ten_genres.Genres.loc[:9]);


# In[ ]:


plt.figure(figsize=(20, 8))
sns.countplot(x='Rating Size', hue='Type', order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+'], data=df);


# In[ ]:


sns.catplot(x='Type', y='Price In Dollors', col='Rating Size',col_order=['0.0 - 1.0','1.0 - 2.0','2.0 - 3.0','3.0 - 4.0','4.0 - 5.0','5.0 - 6.0','6.0 - 7.0','7.0 - 8.0','9.0+'], data=df);


# **Extracting year from last updated column**

# In[ ]:


df['Year'] = ''
df.loc[:,'Year'] = pd.DatetimeIndex(df['Last Updated']).year


# **Created a function which can show to all the important stats of the particular year**

# In[ ]:


def get_yearby_info(y, plot=False):
    
    if y > 2018 and y < 2010:
        raise ValueError('Year starts from 2010 to 2018')
    if y is None:
        raise ValueError('Please enter year')
    
    year = df[df['Year'] == y]
    installs = year.groupby(['App','Year','Type', 'Size', 'Genres', 'Content Rating','Price In Dollors'])['Installs'].sum().reset_index()
    top_installs = installs.sort_values(by='Installs', ascending=False).reset_index(drop=True)
    
    if plot == False:
        return top_installs.head(10)
    else:
        plt.figure(figsize=(20, 20))
        plt.subplot(321)
        plt.xscale('log')
        sns.barplot(x='Installs', y=top_installs.App.loc[:9], data=top_installs);

        plt.subplot(322)
        plt.xscale('log')
        sns.distplot(top_installs['Installs']);

        plt.figure(figsize=(40, 10))
        plt.subplot(323)
        plt.xscale('log')
        sns.boxplot(x='Installs', y='Type', data=top_installs);


# In[ ]:


get_yearby_info(2018)


# In[ ]:


get_yearby_info(2018, plot=True)

