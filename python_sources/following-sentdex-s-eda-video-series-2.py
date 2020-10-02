#!/usr/bin/env python
# coding: utf-8

# ###  Following sentdex's video 3, 4 and 5 of [Data Analysis w/ Python](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfSfqQuee6K8opKtZsh7sA9)

# In[ ]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("../input/us-minimum-wage-by-state-from-1968-to-2017/Minimum Wage Data.csv", encoding="latin")
df.head()


# Transforming the dataframe to have states as columns and "Low.2018" as the values in the column, indexed by the year. Using pandas' groupby feature, create a new dataframe with that structure

# In[ ]:


min_wage_df = pd.DataFrame()

for name, group in df.groupby('State'):
    if min_wage_df.empty:
        min_wage_df = group.set_index('Year')[['Low.2018']].rename(columns={'Low.2018': name})
    else:
        min_wage_df = min_wage_df.join(group.set_index('Year')[['Low.2018']].rename(columns={'Low.2018': name}))
        
min_wage_df.head()


# A quick run down of distribution of data in all the states using dataframe's describe()

# In[ ]:


min_wage_df.describe()


# Next, seeing the correlation and covariance of the data between states

# In[ ]:


min_wage_df.corr().head()


# Seems like Alabama does not have any data and hence it is doing an NaN, let us check all other states

# In[ ]:


min_wage_df.corr()


# Quite a lot of states have all data missing, then possibly some states also have partial data! Checking the dataframe...

# In[ ]:


df[df['Low.2018']==0]['State'].unique()


# Moving On, we can get rid of that data that does not exist. And then see the correlation

# In[ ]:


import numpy as np

min_wage_corr = min_wage_df.replace(0, np.NaN).dropna(axis=1).corr()
min_wage_corr


# Graphing the correlation table!

# In[ ]:


import matplotlib.pyplot as plt
plt.matshow(min_wage_corr)


# That is not so useful, I mean, what are those labels? when we have states in the correlation! Use the swiss-knife of matplotlib to some extent to tidy things up

# In[ ]:


labels = [c[:2] for c in min_wage_corr.columns]

# Figure is the base canvas in the plot
fig = plt.figure(figsize=(12, 12))

# Add one subplot with 2 dimensions
ax = fig.add_subplot(111)

# Fill the data from correlation table and change the color map to be a good one
# green = 1.0, red = 0.0
ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)

# Set labels to states
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

# Ask to display all labels than what is the asthetics
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

# Show
plt.show()


# Fixing the labels by using external data, because these labels here are sloppy!

# In[ ]:


import requests

web = requests.get("https://www.infoplease.com/state-abbreviations-and-state-postal-codes")
dfs = pd.read_html(web.text)


# In[ ]:


# Checking the data frames downloaded
for df in dfs:
    print(df.head())


# In[ ]:


postal_code_df = dfs[0].copy()
postal_code_df.set_index('State/District', inplace=True)
postal_code_df.head()


# In[ ]:


postal_code_dict = postal_code_df[["Postal Code"]].to_dict()['Postal Code']
postal_code_dict['Federal (FLSA)'] = "FLSA"
postal_code_dict['Guam'] = 'GU'
postal_code_dict['Puerto Rico'] = 'PR'

postal_code_dict


# In[ ]:


labels = [postal_code_dict[state] for state in min_wage_corr.columns]


# Plotting again with good labels

# In[ ]:


# Figure is the base canvas in the plot
fig = plt.figure(figsize=(12, 12))

# Add one subplot with 2 dimensions
ax = fig.add_subplot(111)

# Fill the data from correlation table and change the color map to be a good one, 
# green = 1.0, red = 0.0
ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)

# Set labels to states
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)

# Ask to display all labels than what is the asthetics
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))

# Show
plt.show()


# ## BAM!

# ### VIDEO 5 - Combining new datasets with minimum wage dataset 

# ### Load the unemployment rate by county dataset to into a pandas dataframe

# In[ ]:


unemp_county = pd.read_csv('../input/unemployment-by-county-us/output.csv')
unemp_county.head()


# Dropping zeros from min wage dataframe on the column axis

# In[ ]:


min_wage_df = min_wage_df.replace(0, np.NaN).dropna(axis=1)
min_wage_df.head()


# Taking this minimum wage and adding it to the unemployment county dataset as a column, worth noting is the fact that the range of time on unemployement dataset is smaller compared to minimum wage dataset.
# 
# Also note that minimum wage data does not exist for counties, so for all counties in a state assume the same min wage rates 

# Method 1 of doing this - raw and descriptive, is slow too

# In[ ]:


def get_min_wage(year, state):
    try:
        return min_wage_df.loc[year, state]
    except:
        return np.NaN

# Get the min wage for a state for an year, fail with a NaN if does not exist 
get_min_wage(2012, 'Colorado')


# Using the familiar map() of python to produce an ordered list of values which correspond to each row in the unemployment rate by county dataframe, it takes time, because it is O(n\*\*2)

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nunemp_county['min_wage'] = list(map(get_min_wage, unemp_county['Year'], unemp_county['State']))")


# In[ ]:


unemp_county.head()


# Checking the correlation between minimum wage and unemployment rate

# In[ ]:


unemp_county[["Rate", "min_wage"]].corr()


# Seems there is not much correlation! There may be other factors which influence either or both of these, but they among themselves do not

# In[ ]:


unemp_county[["Rate", "min_wage"]].cov()


# They seem to vary together for sure

# ### Reading in the presidential dataset

# In[ ]:


pres16 = pd.read_csv('../input/2016uspresidentialvotebycounty/pres16results.csv')
pres16.head()


# In[ ]:


len(unemp_county)


# Unemployment county dataframe is huge, probably better off taking only data just running up until the elections 

# In[ ]:


unemp_county_2015 = unemp_county.copy()[(unemp_county['Year'] == 2015) & (unemp_county['Month'] == 'February')]
unemp_county_2015.head()


# Look! There are NaN values for min_wage, maybe after brushing out all the zeros in min_wage dataframe it cannot map for certain combinations of year and state, hence the NaNs exist

# Cursory look at the election results data - what all states are included? Because, here, dealing with state level data

# In[ ]:


pres16['st'].unique()


# Need to map `unemp_county_2015` state column to postal codes to later mash it up against election data

# In[ ]:


unemp_county_2015['State'] = unemp_county_2015['State'].map(postal_code_dict)


# In[ ]:


unemp_county_2015.head()


#    Checking the size of `unemp_county_2015` and `pres16`

# In[ ]:


print(len(unemp_county_2015))
print(len(pres16))


# Mashing `pres16` data into `unemp_county_2015` base as it is a smaller set
# 
# Join will be on state and county columns
# 
# Renaming these columns on `pres16` dataframe to be consistent with `unemp_county_2015` 

# In[ ]:


pres16.rename(columns={'county': 'County', 'st': 'State'}, inplace=True)


# In[ ]:


pres16.head()


# Setting index of both the dataframes as a double index [State, County]

# In[ ]:


for df in [unemp_county_2015, pres16]:
    df.set_index(['County', 'State'], inplace=True)


# In[ ]:


pres16.head()


# #### Now, we are interested in what is dependence between minimum wages, unemployement rate and vote percentage to Trump

# In[ ]:


pres16 = pres16.copy()[pres16['cand'] == 'Donald Trump'][["pct"]]
pres16.dropna(inplace=True)
pres16.head()


# #### Mashing!

# In[ ]:


all_together = unemp_county_2015.merge(pres16, on=['County', 'State'])
all_together.dropna(inplace=True)


# In[ ]:


all_together.head()


# Dropping "Year" and "Month" columns

# In[ ]:


all_together.drop("Year", axis=1, inplace=True)
all_together.drop("Month", axis=1, inplace=True)
all_together.head()


# Now the `all_together` dataframe has all the interesting columns, computing the correlation and covariance for the dataframe!

# In[ ]:


all_together.corr()


# In[ ]:


all_together.cov()


# Minimum wage is negatively correlated to percent of votes to Trump - lesser the minimum wage, higher the vote percent to Trump, which is kind of expected.
# 
# Unemployement Rate does not seem to have had a great impact on vote share to Trump. Hmm! _Politics in complicated_
# 
# But on the other hand, these things seem to be not varying together.

# ### DOUBLE BAM!
