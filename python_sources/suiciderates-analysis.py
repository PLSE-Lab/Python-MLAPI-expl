#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/master.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# **Just interested in a few basic things about the dataset like it's correctness which I'll explore next.**

# 1. Is suicides/100K pop correctly calculated?

# In[ ]:


df['my_suicides_per_100K'] = (df['suicides_no'] * 100000/df['population']).round(2)
df[df['my_suicides_per_100K'] != df['suicides/100k pop']][['my_suicides_per_100K', 'suicides/100k pop']]


# There are 4 rows that just differ by 0.1; will let them pass. If it was a much bigger discrepancy, it might indicate some issues with data-gathering & we would have had to consider some options (including dropping them)

# In[ ]:


df = df.drop('my_suicides_per_100K', axis=1)
df.shape


# 2. How many categories do age & generation have?

# In[ ]:


df['age'].value_counts()


# In[ ]:


df['generation'].value_counts()


# Ok, so it's 6 categories for each which is manageable for visualization.
# 
# At first, I thought that they need to match, but there might be overlap in which case they might not.
# 
# Also, age is an object right now & hence, it is not ordered (25 is younger than 35); we might have to fix that later for better analysis/viz.

# **Let's do some time-based analysis for the suicide rates with other categorical or numeric variables**

# First, let's just plot the mean suicide rates over time

# In[ ]:


sns.lineplot(data=df.groupby('year')['suicides/100k pop'].mean())


# Let's get the countries with the most & least suicide rates

# In[ ]:


countries_suicide_rates = df.groupby('country')['suicides/100k pop'].mean().sort_values()
print("Countries with the least suicide rates")
countries_least = countries_suicide_rates[:10]
print(countries_least)
print("Countries with the most suicide rates")
countries_most = countries_suicide_rates[-10:]
print(countries_most)


# Let's take a look at these countries over time

# In[ ]:


def plot_countries_by_year(df, countries):
    countries_by_year = df[df['country'].isin(countries)].groupby(['country', 'year'])['suicides/100k pop'].mean()
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=countries_by_year.to_frame().reset_index(), x='year', y='suicides/100k pop', hue='country')


# In[ ]:


plot_countries_by_year(df, countries_least.index)


# In[ ]:


plot_countries_by_year(df, countries_most.index)


# Well, there is a trend for all countries with the highest suicide rates to increase in the 1990s but, then decrease significantly thereafter (hopefully, due to better help made available)
# 
# The countries with the least suicide rates seem steady over the years with a few spikes, but no general upward or downward trend. The spikes dont seem to happen at the same time which would indicate some global phenomenon

# Let's look at countries with the highest average annual rate change

# In[ ]:


country_year_mean = df.groupby(['country', 'year'])['suicides/100k pop'].mean().to_frame()
country_year_mean['pct_change'] = country_year_mean.sort_index().groupby('country')['suicides/100k pop']. transform(lambda x : x.pct_change())
country_year_mean.head()


# In[ ]:


country_pct_change = country_year_mean['pct_change'].abs().groupby('country').mean()
country_pct_change = country_pct_change.replace([np.inf, -np.inf], np.nan).dropna()
highest_countries = country_pct_change.sort_values()[-10:]
highest_countries


# In[ ]:


plot_countries_by_year(df, highest_countries.index)


# So, we see some relatively big spikes there which might be due to government or health instability

# Let's see how the rates varied over time for the age & sex categories

# In[ ]:


for col in ['age', 'sex']:
    plt.figure(figsize=(12, 8))
    by_category = df.groupby(['year', col])['suicides/100k pop'].mean()
    sns.lineplot(data=by_category.to_frame().reset_index(), x='year', y='suicides/100k pop', hue=col)
    plt.show()


# Some interesting observations :
# 1. The oldest age group has the most suicide rates throughout history
# 1. The suicide rates are pretty uniformly increasing/decreasing across the age-groups; so good or bad mental health environment has affected all age-groups similarly (although more for the oldest age-group)
# 1. Males have suicide rate increases/decreases over time, but that's less pronounced for female suicide rates & definitely doesn't match up when male suicide rates increase/decrease. Maybe, male & female suicides are driven by different underlying factors

# Next, let's compare the GDP with the suicide rates

# In[ ]:


mean_gdp_rates = df.groupby('country')[['suicides/100k pop', 'gdp_per_capita ($)', 'population']].mean()
mean_gdp_rates.head()


# In[ ]:


plt.figure(figsize=(12, 8))
sns.scatterplot(data=mean_gdp_rates, x='gdp_per_capita ($)', y='suicides/100k pop', size='population', )


# In[ ]:


mean_gdp_rates.corr()


# There doesn't seem to be a lot of correlation between mean GDP & mean suicides over all the years.
# 1 thing to note is that higher suicide rates are dominated by lower-gdp countries.

# Let's do a time-series plot of GDP vs Suicide rates for 1 country. The scale of the 2 is very different, so we need to scale them to a uniform scale (0-1 using MinMaxScaler)

# In[ ]:


def plot_countries__with_gdp(countries):
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 20), squeeze=False)
    gdp_rates_country_year = df.groupby(['country', 'year'])[['gdp_per_capita ($)', 'suicides/100k pop']].mean()
    cols = ['gdp_per_capita ($)', 'suicides/100k pop']
    i, j = 0, 0
    for country in countries:
        country_gdp_rates = gdp_rates_country_year.loc[country, :]
        for col in cols:
            country_gdp_rates[col] = (country_gdp_rates[col] - country_gdp_rates[col].min())/(country_gdp_rates[col].max() - country_gdp_rates[col].min())
        country_gdp_rates.sort_index().plot(ax=ax[i][j], title=country)
        if j == 1:
            i = i + 1
        j = 1 - j
    plt.tight_layout()
    plt.show()


# In[ ]:


plot_countries__with_gdp(countries_least.index)


# GDP is related to suicide rates differently for different countries. For eg. for South Africa & Kuwait, even though the GDP has increased, the suicide rates have not decreased. But, they have decreased for Oman & Maldives for eg.

# In[ ]:


plot_countries__with_gdp(countries_most.index)


# Wow! For all the countries here, the suicide rates have decreased as the GDP has increased (which makes sense)

# In[ ]:




