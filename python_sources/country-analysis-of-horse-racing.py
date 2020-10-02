#!/usr/bin/env python
# coding: utf-8

# I would like to conduct an analysis on the country of origin of horses
# 
# 1) The distribution of horse by country
# 2) The winning rate of each country in all races
# 3) If we bet on particular country in each race, is it a good strategy? What is the expected return?

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt

# load race and horse data into dataframe
df_races = pd.read_csv('../input/races.csv', parse_dates=['date']).set_index('race_id')
df_runs = pd.read_csv('../input/runs.csv')


# In[5]:


# create a dataframe of horse country
df_horse_country = df_runs.groupby(by='horse_id')['horse_country'].first().to_frame()

# get the no of horse from each country
series_country = df_horse_country.groupby(by='horse_country')['horse_country'].count()
series_country = series_country.sort_values(ascending=False)

ax = series_country.plot(kind='bar')
ax.set_title('The no of horse from each country')

# annotate bar chart with the no of horse from each country
for idx, p in enumerate(ax.patches):
    ax.annotate(str(series_country[idx]), (p.get_x() * 1.005, p.get_height() * 1.02))

plt.show()

series_country.head(10)


# Most of the horses are from Australia, New Zealand and Ireland.
# 
# You can find a full list of country from the HKJC website
# http://www.hkjc.com/english/racinginfo/pps_import_critieria.htm

# In[7]:


# calculate the winning rate of the horses for each country
# winning rate = no of won / no of race
df_winning_rate = df_runs.groupby(by=['horse_country']).agg({'won':'mean', 'race_id':'count'})
df_winning_rate.columns = ['winning rate', 'no of race']
df_winning_rate = df_winning_rate.sort_values(by='winning rate', ascending=False)
df_winning_rate.head(10)


# Next, we examine the winning rate for each country
# The top five are Japan, Australia, Zimbabwe, New Zealand and South Africa respectively.

# In[9]:


def get_dividend(run):
    """delegate function which re"""
    # if the horse won in the race, return the win dividend, otherwise return 0
    if df_races.loc[run.race_id, 'win_combination1'] == run.horse_no:
        return df_races.loc[run.race_id, 'win_dividend1']
    else:
        return 0
    
# calculate the win dividend for each run:
# if won => return win dividend
# else return 0
df_runs['win_dividend1'] = df_runs.apply(get_dividend, axis=1)


# In[11]:


def cost(races):
    """delegate which calculate the """
    return len(races) * 10

# calculate the outcome: win dividend - cost ($10)
df_runs['outcome'] = df_runs['win_dividend1'] - 10

# group all runs by country, sum all outcomes and count total no of race
df_outcome = df_runs.groupby(by='horse_country')['outcome', 'race_id'].agg({'outcome':'sum', 'race_id':cost})

# calculate gain/loss in percentage
df_outcome['gain/loss (%)'] = (df_outcome['outcome'] / df_outcome['race_id']) * 100

# rename columns and sort data by gain/loss
df_outcome.columns = ['outcome', 'cost', 'gain/loss (%)']
df_outcome = df_outcome.sort_values(by='gain/loss (%)', ascending=False)
df_outcome.head(10)


# If we bet on particular country in each race, we will not have positive return except Zimbabwe. However, the sample size of Zimbabwe is too small, the result may not be statistically significant.
