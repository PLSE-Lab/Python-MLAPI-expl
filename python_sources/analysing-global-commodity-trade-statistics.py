#!/usr/bin/env python
# coding: utf-8

# # Balance of trade
# 
# Hello everyone,
# 
# Today, I will try to analyse the  [Global commodity trade statistics] (https://www.kaggle.com/unitednations/global-commodity-trade-statistics) dataset. I will focus on the balance of trade. **I know that the balance of trade should not be considered as an indicator to judge the economy of a country but I just wanted something to try my hands on**.
# I will also try to see more in details how France is doing, since I'm french. You can just replace France with the country you want to analyse in this notebook if you're interested in a country in particular.
# 
# I know some analysis have been done on this dataset. I don't know exactly everything that is in already existing kernels, please excuse me if it already exists.

# We will start with the loading of libraries and data.

# In[54]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv("../input/commodity_trade_statistics_data.csv")
df.head()


# Let's check for missing values.

# In[81]:


df.count()


# In[4]:


df.isnull().sum()


# There is not that much missing values in comparison with the quantity of data available, so we will just drop lines with missing values.

# In[5]:


df = df.dropna(how='any').reset_index(drop=True)  


# How much commodity are included in this dataset ?

# In[82]:


df['commodity'].unique().shape


# That's a lot, I would like to make an analysis of the overall balance for each country first.
# We will start by grouping the data by country, year and flow and getting rid of everything else.

# In[7]:


gr_country_year_flow = pd.DataFrame({'trade_usd' : df.groupby( ["country_or_area","year","flow"] )["trade_usd"].sum()}).reset_index()
gr_country_year_flow.head()


# Then let's go for the trade of balance (OMG I used the french name *balance import export*), I'm sorry !

# In[8]:


def balance_imp_exp(gr_country_year_flow):
    gr_export = gr_country_year_flow[gr_country_year_flow['flow'] == 'Export'].reset_index(drop = True)
    gr_import = gr_country_year_flow[gr_country_year_flow['flow'] == 'Import'].reset_index(drop = True)
    del gr_export['flow']
    gr_export.rename(columns = {'trade_usd':'export_usd'}, inplace = True)
    gr_export['import_usd'] = gr_import['trade_usd']
    import_export_country_year = gr_export
    import_export_country_year['balance_import_export'] = import_export_country_year['export_usd'] - import_export_country_year['import_usd']
    return import_export_country_year


# In[9]:


import_export_country_year = balance_imp_exp(gr_country_year_flow)
balance_imp_exp(gr_country_year_flow).head()


# I should have checked earlier, how many countries are included in this dataset ?

# In[10]:


import_export_country_year['country_or_area'].unique().shape


# Strange, I thought there were not that many countries...
# Let's see what's in the list.

# In[100]:


countries = import_export_country_year['country_or_area'].unique()
countries


# Oh, ok. There are some groupe of country such as *EU-28*, *Belgium-Luxembourg* or the different *SAR of China*. We can even see the germany before the fall of the wall. 
# We won't take this into account yet.
# 
# We want to see the top countries in term of balance for a given year, let's do it.

# In[11]:


def sorted_balance_year(year, import_export_country_year):
    sorted_balance = import_export_country_year[import_export_country_year['year'] == year].sort_values(by=['balance_import_export'], ascending=False)
    return sorted_balance


# In[111]:


sorted_balance_2016 = sorted_balance_year(2016, import_export_country_year)
plot = sorted_balance_2016[:20].plot(x='country_or_area' , y='balance_import_export', kind='bar', legend = False, figsize=(20, 10))


# In 2016, Belgium was first. To be honest, I'm surprised.

# Now, let's see how the *baguettes* have been doing over the years.

# In[85]:


def plot_country_revenue(country, import_export_country_year):
    data_country = import_export_country_year[import_export_country_year['country_or_area'] == country].sort_values(by=['year'], ascending=True)
    plot = data_country.plot(x='year' , y='balance_import_export', kind='bar', legend = False, color = np.where(data_country['balance_import_export']<0, 'red', 'black'), figsize=(20, 12))


# In[86]:


plot_country_revenue('France', import_export_country_year)


# Well, I would say the 2008 crisis is responsible for that drop in 2009 but I'm no economist. I should investigate this one day.

# Now, let's see the France more in details. First, we extract the data for France. We will forget French Guiana and French Polynesia. Sorry guys, I still like you !

# In[16]:


def info_country(country, df):
    info = df[df['country_or_area'] == country].reset_index(drop = True)
    return info


# In[87]:


info_France = info_country('France', df)
info_France.head()


# We get rid of a few things and then, we calculate the balance of trade for each commodity.

# In[24]:


info_France.drop(columns = ['comm_code', 'commodity', 'weight_kg', 'quantity_name', 'quantity'], inplace = True)
info_France_category = pd.DataFrame({'trade_usd' : info_France.groupby( ["country_or_area","year","flow", "category"] )["trade_usd"].sum()}).reset_index()


# In[89]:


info_France_category.head()


# In[90]:


balance_France = balance_imp_exp(info_France_category)
balance_France.head()


# What was the year and commodity that is responsible for the biggest trade of balance ?

# In[91]:


balance_France[balance_France['balance_import_export'] == max(balance_France['balance_import_export'])]


# Well, thanks Airbus I guess. I know it's not the only contributor but it's probably the biggest. I don't think weapons such as war planes and rockets are included. I will check later.

# Let's see how the business have been going these years.

# In[92]:


plot = balance_France[balance_France['category'] == '88_aircraft_spacecraft_and_parts_thereof'].plot(x='year' , y='balance_import_export', kind='bar', legend = False, color = 'black', figsize=(20, 12))


# Quite good apparently.

# Let's see the top ten commodities in term of trade of balance during 2016.

# In[95]:


plot = plt.pie(balance_France[balance_France['year'] == 2016]['balance_import_export'].sort_values(ascending = False)[0:11], labels = balance_France[balance_France['year'] == 2016].sort_values(by = 'balance_import_export', ascending = False)['category'][0:11], autopct='%1.1f%%')


# Pharmaceuticals ? I did not now.
# I guess everyone knew France were exporting some alcool as well as perfumes and cosmetics.

# **To be continued.** 
# This kernel is a way for me to try my hands on visualisation, please give me your opinion, I'm deeply interested.

# In[68]:




