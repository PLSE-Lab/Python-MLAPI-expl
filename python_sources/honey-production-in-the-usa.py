#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from  matplotlib.ticker import PercentFormatter
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/honeyproduction.csv')


# In[ ]:


df.head(10)


# In[ ]:


df['per_10k_col']=(df['numcol']/10000)


# In[ ]:


df.describe()


# # Let's look at 2012 data

# In[ ]:


a = df.groupby('year')
yr2012 = a.get_group(2012)
total = a['totalprod'].sum()
print (total)


# In[ ]:


df['percent_prod'] = (df['totalprod']/140907000)*100
yr2012['percent_prod'] = (yr2012['totalprod']/140907000)*100


# In[ ]:


yr2012['percent_prod'].sum() #check to make sure sum is actually 100%


# ## Which state has the largest number of colonies?
# ### This is per 10k colonies to keep numbers more manageable

# In[ ]:


plot = sns.barplot(x='state', y = 'per_10k_col', data=yr2012)
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Number of colonies (per 10k colonies)')
plt.title('2012: Number of colonies per state')


# ## What percentage of honey does each state contribute to total US production?
# ### I converted total production (state) into percentage of US production and called this 'percent_prod' 

# In[ ]:


plot = sns.barplot(x='state', y = 'percent_prod', data=yr2012)
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Percent of domestic production')
plt.title('2012: Domestic production of honey')


# In[ ]:


plot = sns.barplot(x='state', y = 'yieldpercol', data=yr2012)
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Yield per colony')
plt.title('2012: Yield per colony')


# ### For the year 2012:
# - North Dakota led the nation in honey production. 
# - North Dakota also had the greatest number of colonies.
# - Mississippi had the highest yield per colony.

# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="state",y="yieldpercol", hue="year", kind="swarm", data=df).set_xticklabels(rotation=90)
(g.fig.suptitle('Colony yield: USA'))


# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="state",y="priceperlb", hue='year', kind='point', data=df).set_xticklabels(rotation=90)
(g.fig.suptitle('Price per lb: 1998-2012'))


# ## Split the dataset into 5-year increments

# ### 1998-2002

# In[ ]:


years = [1998, 1999, 2000, 2001, 2002]
first5 = df[df.year.isin(years)]
first5.head()
first5.tail()


# In[ ]:


plot = sns.catplot(x='state', y='percent_prod', col='year', kind='swarm', data=first5)
plot.set_xticklabels(rotation=90)


# In[ ]:


sns.set_style("whitegrid")
sns.set_palette("summer")
g=sns.catplot(x="state",y="yieldpercol", hue="year",dodge=1, kind="swarm", data=first5).set_xticklabels(rotation=90)
(g.fig.suptitle('Colony yield: USA'))


# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="state",y="priceperlb", hue='year', kind='point', data=first5).set_xticklabels(rotation=90)
(g.fig.suptitle('Price per lb: 1998-2002'))


# ### 2003-2007

# In[ ]:


nextyears = [2003, 2004, 2005, 2006, 2007]
second5 = df[df.year.isin(nextyears)]
second5.tail()


# In[ ]:


sns.set_style("whitegrid")
sns.set_palette("winter")
g=sns.catplot(x="state",y="priceperlb", hue='year', kind='point', data=second5).set_xticklabels(rotation=90)
(g.fig.suptitle('Price per lb: 2003-2007'))


# ### 2008-2012

# In[ ]:


lastyears = [2008, 2009, 2010, 2011, 2012]
last5 = df[df.year.isin(lastyears)]
last5.head()


# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="state",y="priceperlb", hue='year', kind='point', data=last5).set_xticklabels(rotation=90)
(g.fig.suptitle('Price per lb: 2008-2012'))


# # Regional distribution
# 

# ### Regions according to Bureau of Economic Analysis 
# #### New England: Connecticut, Maine, Massasschusetts, New Hampshire, Rhode Island, Vermont
# #### Mideast: Delaware, DC, Maryland, New Jersey, New York, Pennsylvania
# #### Great Lakes: Illinois, Indiana, Michigan, Ohio, Wisconsin
# #### Plains: Iowa, Kansas, Minnesota, Missouri, Nebraska, North Dakota, South Dakota
# #### Southeast: Alabama, Arkansas, Florida, Georgia, Kentucky, Louisianna, Mississippi, North Carolina, Tennessee, Virginia, West Virgnia
# #### Rocky Mountain: Colorado, Idaho, Montana, Utah, Wyoming
# #### Southwest: Arizona, New Mexico, Texas, Oklahoma
# #### Far West: California, Alaska, Hawaii, Nevada, Oregon, Washington

# ## New England Region

# In[ ]:


newEngRegion = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']
newEng = df[df.state.isin(newEngRegion)]
newEng.tail()


# In[ ]:


sns.set_style("whitegrid")
sns.set_palette('Blues')
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=newEng).set_xticklabels(rotation=90)
(g.fig.suptitle('New England Honey production'))


# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="year",y="numcol", hue='state', kind='point', data=newEng).set_xticklabels(rotation=90)
(g.fig.suptitle('New England: Number of honey colonies'))


# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="year",y="yieldpercol", hue='state', kind='point', data=newEng).set_xticklabels(rotation=90)
(g.fig.suptitle('New England: Coloney yield, 1998-2012'))


# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=newEng).set_xticklabels(rotation=90)
(g.fig.suptitle('New England: percent of total US production, 1998-2012'))


# ## Mid East USA Region

# In[ ]:


midEastRegion = ['DE', 'DC', 'MD', 'NJ', 'NY', 'PA']
midEast = df[df.state.isin(midEastRegion)]
midEast.tail()


# In[ ]:


sns.set_style("whitegrid")
sns.set_palette("hls")
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=midEast).set_xticklabels(rotation=90)
(g.fig.suptitle('Mid East Region Honey production'))


# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="year",y="numcol", hue='state', kind='point', data=midEast).set_xticklabels(rotation=90)
(g.fig.suptitle('Mid East US: Number of honey colonies'))


# In[ ]:


sns.set_style("whitegrid")
g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=midEast).set_xticklabels(rotation=90)
(g.fig.suptitle('Mid East US: percent of total US production, 1998-2012'))


# ## Great Lakes Region:

# In[ ]:


greatLakesRegion = ['IL', 'IN', 'MI', 'OH', 'WI']
greatLakes = df[df.state.isin(greatLakesRegion)]
greatLakes.tail()


# In[ ]:


sns.set_style("white")
sns.set_palette("PRGn")
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)
(g.fig.suptitle('Great Lakes Region Honey production'))


# In[ ]:


g=sns.catplot(x="year",y="yieldpercol", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)
(g.fig.suptitle('Great Lakes Region: Coloney yield, 1998-2012'))


# In[ ]:


g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)
(g.fig.suptitle('Great Lakes Region Honey production'))


# In[ ]:


g=sns.catplot(x="year",y="numcol", hue='state', kind='point', data=midEast).set_xticklabels(rotation=90)
(g.fig.suptitle('Great Lakes US: Number of honey colonies'))


# In[ ]:


g=sns.catplot(x="year",y="yieldpercol", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)
(g.fig.suptitle('Great Lakes US: Coloney yield, 1998-2012'))


# In[ ]:


g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=greatLakes).set_xticklabels(rotation=90)
(g.fig.suptitle('Great Lakes US: percent of total US production, 1998-2012'))


# ## Plains Region, USA

# In[ ]:


plainsRegion = ['IA', 'KS', 'MN', 'MS', 'NE', 'ND', 'SD']
plains = df[df.state.isin(plainsRegion)]
plains.tail()


# In[ ]:


plains.head()


# In[ ]:


sns.set_style("white")
sns.set_palette("BrBG")
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=plains).set_xticklabels(rotation=90)
(g.fig.suptitle('Plains Region Honey production'))


# In[ ]:


g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=plains).set_xticklabels(rotation=90)
(g.fig.suptitle('Plains Region, USA: Number of honey colonies'))


# In[ ]:


g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=plains).set_xticklabels(rotation=90)
(g.fig.suptitle('Plains Region, USA: percent of total US production, 1998-2012'))


# ## Southeast USA

# In[ ]:


southEastRegion = ['AL', 'AK', 'FL', 'GA', 'KY', 'LA', 'MS', 'NC', 'TN', 'VA', 'WV']
southEast = df[df.state.isin(southEastRegion)]
southEast.tail()


# In[ ]:


sns.set_style("white")
sns.set_palette("Spectral")
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=southEast).set_xticklabels(rotation=90)
(g.fig.suptitle('Southeast Region Honey production'))


# In[ ]:


g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=southEast).set_xticklabels(rotation=90)
(g.fig.suptitle('Southeast Region, USA: Number of honey colonies'))


# In[ ]:


g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=southEast).set_xticklabels(rotation=90)
(g.fig.suptitle('Southeast Region, USA: percent of total US production, 1998-2012'))


# ## Rocky Mountain Region, USA

# In[ ]:


rockyRegion = ['CO', 'ID', 'MT', 'UT', 'WY']
rocky = df[df.state.isin(rockyRegion)]
rocky.tail()


# In[ ]:


sns.set_style("white")
sns.set_palette("dark")
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=rocky).set_xticklabels(rotation=90)
(g.fig.suptitle('Rocky Mountain Region, USA Honey production'))


# In[ ]:


g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=rocky).set_xticklabels(rotation=90)
(g.fig.suptitle('Rocky Mountain Region, USA: Number of honey colonies'))


# In[ ]:


g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=rocky).set_xticklabels(rotation=90)
(g.fig.suptitle('Rocky Mountain Region, USA: percent of total US production, 1998-2012'))


# ## Southwest Region, USA

# In[ ]:


SWRegion = ['AZ', 'NM', 'TX', 'OK']
SW = df[df.state.isin(SWRegion)]
SW.tail()


# In[ ]:


sns.set_style("white")
sns.set_palette("PuOr")
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=SW).set_xticklabels(rotation=90)
(g.fig.suptitle('Southwest Region, USA Honey production'))


# In[ ]:


g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=SW).set_xticklabels(rotation=90)
(g.fig.suptitle('Southwest Region, USA: Number of honey colonies'))


# In[ ]:


g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=SW).set_xticklabels(rotation=90)
(g.fig.suptitle('Southwest Region, USA: percent of total US production, 1998-2012'))


# ## Far West

# In[ ]:


farWRegion = ['CA', 'AK', 'HI', 'NV', 'OR', 'WA']
farW = df[df.state.isin(farWRegion)]
farW.tail()


# In[ ]:


sns.set_style("white")
sns.set_palette("PuRd")
g=sns.catplot(x="year",y="priceperlb", hue='state', kind='point', data=farW).set_xticklabels(rotation=90)
(g.fig.suptitle('Far West region, USA Honey production'))


# In[ ]:


g=sns.catplot(x="year",y="per_10k_col", hue='state', kind='point', data=farW).set_xticklabels(rotation=90)
(g.fig.suptitle('Far West Region, USA: Number of honey colonies'))


# In[ ]:


g=sns.catplot(x="year",y="percent_prod", hue='state', kind='point', data=farW).set_xticklabels(rotation=90)
(g.fig.suptitle('Far West Region, USA: percent of total US production, 1998-2012'))

