#!/usr/bin/env python
# coding: utf-8

# ## Daily usage
# 
# The daily_dataset folder contains measurements by day. We are certainly interested in:
# * Daily/weekly/monthly/annual usage
# * Usage by customer segment
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import glob

data_dir = "/kaggle/input/smart-meters-in-london/daily_dataset/"


# There are 112 files in this folder, with each file containing data for 50 homes

# In[ ]:


print(len(os.listdir(data_dir+'daily_dataset')))
os.listdir(data_dir+'daily_dataset')[:5]


# Read data for a single home

# In[ ]:


daily_df = pd.read_csv(data_dir+'daily_dataset/block_71.csv')
daily_df['day'] = pd.to_datetime(daily_df['day'])
daily_df = daily_df.set_index('day')
daily_df.head()


# Each home is referenced by LCLid

# In[ ]:


len(daily_df["LCLid"].unique())


# Plot usage for a single home, noting the data mostly covers 2012-2014

# In[ ]:


daily_df[daily_df["LCLid"]=="MAC000027"]["energy_sum"].plot(figsize=(20,6));


# What we discover is that there is not data for all homes for all time, but it appears more homes are coming online during 2012

# In[ ]:


daily_df.reset_index().groupby("day").nunique()["LCLid"].plot(figsize=(20,6));


# In order to account for the variation in number of measurements per day we could take an idea from another notebook here (todo add ref) and plot a normalised measure of energy use, energy use per household

# In[ ]:



num_households_df = daily_df.reset_index().groupby("day").nunique()["LCLid"] # get the number of households on each day
energy_df = daily_df.reset_index().groupby("day").sum()["energy_sum"] # get the total energy usage per day

# normalise the energy usage to the number of households and plot
energy_per_household_df = pd.concat([num_households_df, energy_df], axis=1)
energy_per_household_df["normalised"] = energy_per_household_df["energy_sum"] / energy_per_household_df["LCLid"]
energy_per_household_df["normalised"].plot(figsize=(20,6));


# Trend of increased energy usage over winter months. Note the curve is more noisy in early 2012 as there are fewer homes contributing measurements.
# 
# We can get the household grouping (Acorn_grouped) from the info file, and later we will merge this with the daily energy data.

# In[ ]:


info_df = pd.read_csv('/kaggle/input/smart-meters-in-london/informations_households.csv')
info_df.head()


# 
# ## Get all data from 112 files
# 
# Merge data from all files to single df
# 

# In[ ]:


# Helper to load a single file
def daily_to_df(file_path : str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['day'] = pd.to_datetime(df['day'])
    df["year"] = df["day"].apply(lambda x : x.year)
    df["month"] = df["day"].apply(lambda x : x.month)
    df["dayofweek"] = df["day"].apply(lambda x : x.dayofweek)
    df["day_name"] = df["day"].apply(lambda x : x.day_name())
    df = df.merge(info_df, on="LCLid")
    df = df[df["year"].isin([2012, 2013])]
    return df[["LCLid", "day", "year", "month", "day_name", "Acorn_grouped", "energy_sum"]]

df = daily_to_df(data_dir+'daily_dataset/block_71.csv')
df.head()


# Loop over all 122 files and place in single dataframe. Note that we are just keeping data for 2012 & 2013 as these are mostly complete

# In[ ]:


all_daily_df = pd.DataFrame()

for i, file_path in enumerate(glob.glob(data_dir+'daily_dataset/*.csv')):
    all_daily_df = all_daily_df.append(daily_to_df(file_path))
    print(all_daily_df.shape)


# Do some basic prep

# In[ ]:


all_daily_df = all_daily_df.drop_duplicates()
all_daily_df = all_daily_df.dropna()
all_daily_df.head()


# check that number of measurements is consistent over time, there are actually more in 2013 so we will use 2013 data in subsequent analysis
# 
# ## 2013

# In[ ]:


y2013_df = all_daily_df[all_daily_df['year']==2013]


# In[ ]:


y2013_df.groupby("Acorn_grouped").count()["LCLid"]


# Drop the two smallest groupings

# In[ ]:


y2013_df = y2013_df[y2013_df["Acorn_grouped"].isin(["Adversity", "Affluent", "Comfortable"])]


# 
# ## Investigate usage by group
# 
# To account for the variation in measurements over time we will just use 2013 data and we will normalise to the number of LCLid, using the normalisation approach described earlier
# 

# In[ ]:


sum_y2013_df = pd.concat([y2013_df.groupby("Acorn_grouped").sum()["energy_sum"], y2013_df.groupby("Acorn_grouped").count()["LCLid"]], axis=1)
sum_y2013_df["normalised"] = sum_y2013_df["energy_sum"] / sum_y2013_df["LCLid"]
sum_y2013_df


# In[ ]:


sum_y2013_df["normalised"].plot.bar();


# 
# Appears affluent use most energy whilst adversity the least, perhaps affluent have larger homes, more gadgets etc
# 
# ## Usage by day

# In[ ]:


y2013_df.groupby("day_name").sum()["energy_sum"].sort_values().plot.bar();


# 
# Usage is slightly higher at weekends
# 
# ## Usage by month (Seasonal Decomposition)

# In[ ]:


y2013_df.groupby("month").sum()["energy_sum"].sort_values().plot.bar()


# usage is higher over winter.
# 
# ## Summary
# This notebook illustrates some basic EDA of the daily data and highlights the issue of varying number of data points over time, and uses normalisation to address this. The analysis shows that affluent households consume more energy, which aligns with our intuition. Analysis of daily and seasonal trends also tallies with common sense, showing higher consumption at weekends (when people are home) and over the winter.

# In[ ]:




