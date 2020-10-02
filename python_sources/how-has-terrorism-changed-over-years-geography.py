#!/usr/bin/env python
# coding: utf-8

# # How has terrorism changed over the years? (Geographic maps)

# ## Table of contents
# * [1. Data import](#1.-Data-import)
# * [2. Full Map](#2.-Full-Map)
# * [3. Annual dynamics with map](#3.-Annual-dynamics-by-map)
# * [4. Annual dynamics by regions](#4.-Annual-dynamics-by-regions)
# * [5. Successful/unsuccessful terrorist strike](#5.-Successful/unsuccessful-terrorist-strike)
# * [6. Types of attacks](#6.-Types-of-attacks)
# * [7. Types of target](#7.-Types-of-target)
# * [8. Human freedom score due to terrorism](#8.-Human-freedom-score-due-to-terrorism)
# * [9. Personal freedom score due to terrorism](#9.-Personal-freedom-score-due-to-terrorism)
# 
# 

# ## 1. Data import

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.basemap import Basemap
get_ipython().run_line_magic('pylab', 'inline')


# According to the proposed data we consider how terrorism has changed over time.

# In[ ]:


data_ter = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')


# In[ ]:


data_ter = data_ter.dropna(thresh=160000,axis=1)
data_ter.head(5)


# Remove columns with a large number of missing values.

# In[ ]:


data_ter.dropna(thresh=160000,axis=1).shape


# ## 2. Full Map
# Descry full map of terrorist attacks. We can see attacks for all the time. Much more important to see the trend. To do this, we divide all data into ten-year segments.
# 

# In[ ]:


def mapWorld(col1,size2,label4,metr=100,colmap='hot',ds=data_ter,scat=False):
    datatt = ds
    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,            llcrnrlon=-150,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-120,91.,30.))
    m.drawmeridians(np.arange(-120,90.,60.))
    lat = datatt['latitude'].values
    lon = datatt['longitude'].values
    a_1 = datatt[col1].values
    if size2:
        a_2 = datatt[size2].values
    else: a_2 = 1
    if scat:
        m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,edgecolors='black',cmap=colmap,alpha=1)
    else:
        m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,cmap=colmap,alpha=1)
sns.set(font_scale=1.5)
plt.figure(figsize=(15,15))
plt.title('Terrorist attacks', fontsize=20)
mapWorld(col1='targtype1', size2=False,label4='',metr=10,colmap='viridis',ds=data_ter)


# ## 3. Annual dynamics by map
# In 1970-1979 terrorist concentrates more tightly in Western Europe, Carribean and Middle East. <br>
# In 1980-1989  - South America, Western Europe, Carribean. <br>
# In 1990-1999, surge of terrorist activity - Western Europe, South Asia, South America, Middle East and North Africa. <br>
# In the past 20 years, terrorists have been concentrated in South Asia, Middle Asia and Sub-Saharan Africa. <br>
# For a clearer view, see the bar charts below.

# In[ ]:


fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(3,2,1)
ax1.set_title('2010-2017')
mapWorld(col1='targtype1',ds=data_ter[data_ter['iyear']>=2010], size2=False,label4='',metr=1,colmap='viridis',scat=True)
ax2 = fig.add_subplot(3,2,2)
ax2.set_title('2000-2009')
mapWorld(col1='targtype1',ds=data_ter[(data_ter['iyear']>=2000) & (data_ter['iyear']<2010)],          size2=False,label4='',metr=10,colmap='viridis',scat=True)
ax3 = fig.add_subplot(3,2,3)
ax3.set_title('1990-1999')
mapWorld(col1='targtype1',ds=data_ter[(data_ter['iyear']>=1990) & (data_ter['iyear']<2000)],          size2=False,label4='',metr=10,colmap='viridis',scat=True)
ax4 = fig.add_subplot(3,2,4)
ax4.set_title('1980-1989')
mapWorld(col1='targtype1',ds=data_ter[(data_ter['iyear']>=1980) & (data_ter['iyear']<1990)],          size2=False,label4='',metr=10,colmap='viridis',scat=True)
ax4 = fig.add_subplot(3,2,5)
ax4.set_title('1970-1979')
mapWorld(col1='targtype1',ds=data_ter[(data_ter['iyear']>=1970) & (data_ter['iyear']<1980)],          size2=False,label4='',metr=10,colmap='viridis',scat=True)


# ## 4. Annual dynamics by regions

# In[ ]:


def plot_by_years(kind='region_txt',big=(30,20)):
    sns.set(style="white",font_scale=2.5)
    fig = plt.figure(figsize=big)
    ax1 = fig.add_subplot(3,2,1)
    ax1.set_title('2010-2017')
    ax1.set_ylabel('');
    data_ter[data_ter['iyear']>=2010]['eventid'].groupby(data_ter[kind]).count().plot(kind='barh');
    ax1.set_ylabel('');
    ax2 = fig.add_subplot(3,2,2)
    ax2.set_title('2000-2009')
    data_ter[(data_ter['iyear']>=2000) & (data_ter['iyear']<2010)]['eventid'].groupby(data_ter[kind]).count().plot(kind='barh');
    ax2.set_ylabel('');
    ax3 = fig.add_subplot(3,2,3)
    ax3.set_title('1990-1999')
    data_ter[(data_ter['iyear']>=1990) & (data_ter['iyear']<2000)]['eventid'].groupby(data_ter[kind]).count().plot(kind='barh');
    ax3.set_ylabel('');
    ax4 = fig.add_subplot(3,2,4)
    ax4.set_title('1980-1989')
    data_ter[(data_ter['iyear']>=1980) & (data_ter['iyear']<1990)]['eventid'].groupby(data_ter[kind]).count().plot(kind='barh');
    ax4.set_ylabel('');
    ax4 = fig.add_subplot(3,2,5)
    ax4.set_title('1970-1979')
    data_ter[(data_ter['iyear']>=1970) & (data_ter['iyear']<1980)]['eventid'].groupby(data_ter[kind]).count().plot(kind='barh');
    plt.tight_layout()
    plt.ylabel('');
plot_by_years(kind='region_txt')


# Consider the distribution of attacks by years. Since 2012, terrorist activity is greatly increased.

# In[ ]:


sns.set(font_scale=1.5)
fig = plt.figure(figsize=(20,10))
sns.countplot(x='iyear',data=data_ter);
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45)
plt.title('Terrorist attacks', fontsize=20)
plt.tight_layout()


# Let's see if there is any feature in the distribution by week or by day. By first view - there are no interesting in these countplots. Only in 31 - not all months contain 31 day.

# In[ ]:


fig = plt.figure(figsize=(10,5))
sns.set(font_scale=1.5)
sns.countplot(data_ter.imonth);
plt.ylabel('count');


# In[ ]:


fig = plt.figure(figsize=(20,5))
sns.set(font_scale=1.5)
sns.countplot(data_ter.iday);
plt.ylabel('count');


# ## 5. Successful/unsuccessful terrorist strike
# This bar chart also does not inspire optimism - the overwhelming majority of attacks end successfully

# In[ ]:


sns.set(font_scale=1.5)
fig = plt.figure(figsize=(20,10))
sns.countplot(x='iyear',data=data_ter,hue='success', orient='v');
plt.xlabel('')
plt.ylabel('')
plt.title('Successful/unsuccessful terrorist strike', fontsize=20)
plt.xticks(rotation=45)
plt.tight_layout()


# In[ ]:


sns.set(font_scale=1.5)
plt.figure(figsize=(20,20))
plt.title('Successful/unsuccessful terrorist strike', fontsize=20)
mapWorld(col1='success', size2=False,label4='',metr=10,colmap='viridis',ds=data_ter)


# What share of incidents was a suicide attack?

# In[ ]:


sns.set(font_scale=1.5)
fig = plt.figure(figsize=(20,10))
sns.countplot(x='iyear',data=data_ter,hue='suicide');
plt.title("The incident was/wasn't a suicide attack", fontsize=20)
plt.xlabel('')
plt.ylabel('')
plt.tight_layout()
plt.xticks(rotation=45);


# ## 6. Types of attacks
# The most frequent types of attacks have not changed much - armed assault and bombing/explosion pushed out assassination. The share of use of bombs increases - terrorist attacks began to be massive and aimed at hitting a large number of people.

# In[ ]:


plot_by_years(kind='attacktype1_txt');


# ## 7. Types of target
# Different types of targets for terrorists:

# In[ ]:


def plot_by_years(kind='region_txt',big=(30,20)):
    sns.set(style="white",font_scale=2.5)
    fig = plt.figure(figsize=big)
    ax1 = fig.add_subplot(3,2,1)
    ax1.set_title('2010-2017')
    ax1.set_ylabel('');
    data_ter[data_ter['iyear']>=2010]['eventid'].groupby(data_ter[kind]).count().sort_values(ascending=False)[:10].plot(kind='barh');
    ax1.set_ylabel('');
    ax2 = fig.add_subplot(3,2,2)
    ax2.set_title('2000-2009')
    data_ter[(data_ter['iyear']>=2000) & (data_ter['iyear']<2010)]['eventid'].groupby(data_ter[kind]).count().sort_values(ascending=False)[:10].plot(kind='barh');
    ax2.set_ylabel('');
    ax3 = fig.add_subplot(3,2,3)
    ax3.set_title('1990-1999')
    data_ter[(data_ter['iyear']>=1990) & (data_ter['iyear']<2000)]['eventid'].groupby(data_ter[kind]).count().sort_values(ascending=False)[:10].plot(kind='barh');
    ax3.set_ylabel('');
    ax4 = fig.add_subplot(3,2,4)
    ax4.set_title('1980-1989')
    data_ter[(data_ter['iyear']>=1980) & (data_ter['iyear']<1990)]['eventid'].groupby(data_ter[kind]).count().sort_values(ascending=False)[:10].plot(kind='barh');
    ax4.set_ylabel('');
    ax4 = fig.add_subplot(3,2,5)
    ax4.set_title('1970-1979')
    data_ter[(data_ter['iyear']>=1970) & (data_ter['iyear']<1980)]['eventid'].groupby(data_ter[kind]).count().sort_values(ascending=False)[:10].plot(kind='barh');
    plt.tight_layout()
    plt.ylabel('');
plot_by_years(kind='targtype1_txt',big=(30,30));


# ## 8. Human freedom score due to terrorism
# Collate this data with another kaggle dataset "The Human Freedom Index". Firstly, slightly change the data for a successful merge.

# In[ ]:


data_hap = pd.read_csv('../input/the-human-freedom-index/hfi_cc_2018.csv')
data_hap.head(3)


# In[ ]:


def reg(x):
    if x in ('South America','Central America & Caribbean'):
        res = 'Latin America & the Caribbean'
    elif x=='Central Asia':
        res = 'Caucasus & Central Asia'
    elif x=='Australasia & Oceania':
        res = 'Oceania'
    elif x in('South Asia','Southeast Asia',):
            res = 'South Asia'
    else:
        res=x
    return res
data_bx = data_ter
data_bx['region_txt'] = data_bx.region_txt.apply(reg)
data_bx = data_bx[data_bx.iyear.isin(['2016','2015','2014','2013','2012','2011','2010','2009','2008'])]


# ## 9. Personal freedom score due to terrorism
# Compare Human Freedom (score) and count of terrorist attacks:

# In[ ]:


plt.figure(figsize=(20,5))
sns.set(font_scale=1.5)
sns.boxplot(x='region',y='hf_score',data=data_hap,order=['Caucasus & Central Asia','East Asia','Eastern Europe',                                'Latin America & the Caribbean','Middle East & North Africa','North America',                                    'Oceania','South Asia','Sub-Saharan Africa','Western Europe']);
plt.title("Human Freedom (score)", fontsize=20)
plt.xlabel('')
plt.xticks(rotation=30)
plt.show()

plt.figure(figsize=(25,5))
sns.set(font_scale=1.5)
data_bx['eventid'].groupby(data_ter['region_txt']).count().plot(kind='bar');
plt.title("Count of terrorist attacks", fontsize=20)
plt.xticks(rotation=30)
plt.xlabel('')
plt.show()


# Compare Personal Freedom (score) and count of terrorist attacks. We can see by plots that number of attacks is related to freedom schedules.

# In[ ]:


plt.figure(figsize=(20,5))
sns.set(font_scale=1.5)
sns.boxplot(x='region',y='pf_score',data=data_hap,order=['Caucasus & Central Asia','East Asia','Eastern Europe',                                'Latin America & the Caribbean','Middle East & North Africa','North America',                                    'Oceania','South Asia','Sub-Saharan Africa','Western Europe']);
plt.title("Personal Freedom (score)", fontsize=20)
plt.xlabel('')
plt.xticks(rotation=30)
plt.show()

plt.figure(figsize=(25,5))
sns.set(font_scale=1.5)
data_bx['eventid'].groupby(data_ter['region_txt']).count().plot(kind='bar');
plt.title("Count of terrorist attacks", fontsize=20)
plt.xlabel('')
plt.xticks(rotation=30)
plt.show()


# Thank you for reading! <br>
# If you like same map visualization, you can see my other kernels:
# https://www.kaggle.com/nikitagrec/kernels
