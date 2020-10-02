#!/usr/bin/env python
# coding: utf-8

# *Aim* : The EDA on Indian Trade is to present a simple insight into the dataset. I would be glad of you to review the work. 
# 
# **Objectives**
# 1. Understand the insights of trade over the period
# 2. Highlight the most valuable trade (export and import) for each country
# 3. Represent the trend of value of trade for a commodity with different countries
# 4. Trend in trade over the years
# 5. Understand the trade relations with different countries
# 
# The entire notebook will primarily comprise of 3 sections.
# > The first section will highlight the trend for exports, second the trend for imports and the third section will be a merger of exports and imports for overall trading with the countries.

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


filenames


# **Exports**

# In[ ]:


exports = pd.read_csv(os.path.join(dirname,'2018-2010_export.csv'))
exports.shape


# In[ ]:


exports.columns


# > Each commodity has one and only one unique HSCode

# In[ ]:


total_value_over_year = exports.groupby(by = ['country','year'])['value'].sum()
total_value_over_year = pd.DataFrame(total_value_over_year)
total_value_over_year.columns


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


total_value_over_year['value'][5:10].index


# In[ ]:


df = pd.DataFrame(total_value_over_year['value'])


# In[ ]:


country = 'U S A'
sns.lineplot(x = df.xs(country).index, y = df.xs(country)['value'], data = df.xs(country))
plt.title('Exports to USA over 2010-18')


# In[ ]:


countries = exports['country'].unique()


# In[ ]:


total_value = total_value_over_year.groupby('country')['value'].sum().sort_values()
type(total_value)


# In[ ]:


total_value = pd.DataFrame(total_value)
print(total_value.index,total_value.columns)


# In[ ]:


def plot_year_trend(country,df):
    sns.lineplot(x = df.index, y = df['value'], data = df)


# **Which country have maximum exports from India in terms of **
# a. Value
# b. Different commodities

# In[ ]:


countries = total_value.tail(5).index
i = 1
plt.figure(figsize=(12,12))
for country in countries:
    d = df.xs(country)
    plot_year_trend(country,d)
plt.legend(countries)
plt.title('Total value of exports for top 5 countries with maximum value over the period of 2010-18')


# > Trade with U ARAB EMTS declined.
# Trade with China declined during the period 2013-15.
# Trade with USA increased with the slight fall during 2014-15

# In[ ]:


total_commodities = exports.groupby('country')['Commodity'].describe().sort_values(by='count',ascending=False)
total_commodities.head(20)


# > 'COFFEE, TEA, MATE AND SPICES' is the most frequent item among the top countries exporting from India.
# The major countries also get maximum varities of commodities exported from India

# **Most Valuable item exported to each country**

# In[ ]:


idx_max = exports.groupby(['country'])['value'].transform(max)== exports['value']
most_valuable_export = exports[idx_max]


# In[ ]:


mvs = most_valuable_export.sort_values(by=['value'],ascending=[False])


# In[ ]:


mvs.head()


# most valuable export has taken place to U ARAB EMTS of value 19805.17 in million US dollars in 2010 for the commodity Natural or cultured pearls,...

# In[ ]:


sns.barplot(x = 'country', y = 'value', data = mvs.head(), hue = 'Commodity')
plt.legend()


# In[ ]:


commodity_export_value = exports.groupby('Commodity')['value'].sum().sort_values(ascending=[False])
commodity_export_value.head()


# Most valuable exported Items

# In[ ]:


plt.figure(figsize=(7,7))
color_palette = sns.color_palette(n_colors = 9)
sns.scatterplot(x = 'HSCode', y = 'value', data = mvs, hue = 'year', palette = color_palette)


# In[ ]:


x = mvs[mvs['value']<1]
x.sample(5)


# In[ ]:


len(x)


# *39 countries have their most valuable export from India worth less than 1 million US dollars*

# In[ ]:


most_valuable_export.groupby('Commodity').describe().sort_values(by=[('HSCode','count')],ascending=[False])


# > Mineral fuels, mineral oils are the most valuable frequently exported items to most of the countries
# > > next to Mineral fuels, vehicles other than railway or tramway rolling stock, parts and accessories have been the most valuable exports with most countries

# **Is there any relationship between the commodity exported and its value?**

# *Total value in exports for each year*

# In[ ]:


total_value_year = exports.groupby('year')['value'].sum()
total_value_year = pd.DataFrame(total_value_year)


# In[ ]:


exports.groupby('year').describe()


# In[ ]:


sns.lineplot(x = total_value_year.index, y = 'value', data = total_value_year)
plt.title('trend of exports for India over the period 2010-18')


# > There is a recession in exports for the year 2015. **What coule be the possible reason?**

# In[ ]:


total_exports_year = exports.groupby('year')['HSCode'].describe()
sns.lineplot(x = total_exports_year.index, y = 'count', data = total_exports_year)
plt.title('Number of Commodities exported vs year')


# > total number of exports have risen over the years with a slight drop around 2014-16, which is one of the reasons for drop in exports around the period.

# In[ ]:


cond = (exports['year']==2013) | (exports['year']==2014) | (exports['year']==2015)
exports_2013_15 = exports.loc[cond,:]
exports_2013_15.sample(5)


# In[ ]:


exports_2013_15.groupby('year').describe()['HSCode']


# Mean value of exports in 2015 have dropped significantly along with the maximum value of commodity exported.

# **There is a rapid jump in exports around 2011**

# In[ ]:


exports_2010_11 = exports.loc[(exports['year']==2010)|(exports['year']==2011),:]
exports_2010_11.sample(7)


# In[ ]:


exports_2010_11.groupby('year').describe()


# Mean value of commodities exported have increased in 2011

# **This is the end of first section of the EDA.** Remaining sections on Import and combination of Import and Export will be made available in further commits.
# Thank You for reading till the end.
