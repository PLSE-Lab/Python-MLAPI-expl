#!/usr/bin/env python
# coding: utf-8

# <h1>Novel Coronavirus 2019-nCoV Data Preprocessing</h1>
# 
# 
# # Introduction
# 
# This Notebook only performs data preprocessing on the dataset with daily updates of coronavirus information.
# 
# We will perform the following operations:
# * Check missing data; perform missing data imputation, if needed;
# * Check last update of the daily data;
# * Check multiple country names; fix the multiple country names, where needed;
# * Check multiple province/state;
# * Check if a country/region appears as well as province/state; 
# * Deep-dive in the case of US states, cities, counties and unidentified places.
# * Export curated data.

# ## Load packages & data

# In[ ]:


import numpy as np
import pandas as pd
import datetime as dt
data_df = pd.read_csv("..//input//novel-corona-virus-2019-dataset//covid_19_data.csv")


# ## Glimpse the data

# In[ ]:


print(f"Data: rows: {data_df.shape[0]}, cols: {data_df.shape[1]}")
print(f"Data columns: {list(data_df.columns)}")

print(f"Days: {data_df.ObservationDate.nunique()} ({data_df.ObservationDate.min()} : {data_df.ObservationDate.max()})")
print(f"Country/Region: {data_df['Country/Region'].nunique()}")
print(f"Province/State: {data_df['Province/State'].nunique()}")
print(f"Confirmed all: {sum(data_df.groupby(['Province/State'])['Confirmed'].max())}")
print(f"Recovered all: {sum(data_df.loc[~data_df.Recovered.isna()].groupby(['Province/State'])['Recovered'].max())}")
print(f"Deaths all: {sum(data_df.loc[~data_df.Deaths.isna()].groupby(['Province/State'])['Deaths'].max())}")

print(f"Diagnosis: days since last update: {(dt.datetime.now() - dt.datetime.strptime(data_df.ObservationDate.max(), '%m/%d/%y')).days} ")


# Comment: the dataset was not updates since few days ago (time to last run this Notebook).

# In[ ]:


data_df.head()


# In[ ]:


data_df.info()


# There are no missing data other than `Province/Region` - which makes sense, since for some of the Countries/Regions there is only Country/Region level data available.

# ## Check multiple countries names

# In[ ]:


country_sorted = list(data_df['Country/Region'].unique())
country_sorted.sort()
print(country_sorted)


# <font color='red'>Comment</font>: we can observe that there are few countries with duplicate name, as following:
# 
# * ` Azerbaijan` & `Azerbaijan`;
# * `Holly See` & `Vatican City`;
# * `Ireland` & `Republic of Ireland`;
# * `St. Martin` & `('St. Martin',)`.
# 
# For `UK` & `North Ireland` we will need a clarification, since theoretically `North Ireland` is a part of `UK`. 

# ## Fix duplicated countries names

# In[ ]:


data_df.loc[data_df['Country/Region']=='Holy See', 'Country/Region'] = 'Vatican City'
data_df.loc[data_df['Country/Region']==' Azerbaijan', 'Country/Region'] = 'Azerbaijan'
data_df.loc[data_df['Country/Region']=='Republic of Ireland', 'Country/Region'] = 'Ireland'
data_df.loc[data_df['Country/Region']=="('St. Martin',)", 'Country/Region'] = 'St. Martin'


# ## Check duplicate Province/State names

# In[ ]:


province_sorted = list(data_df.loc[~data_df['Province/State'].isna(), 'Province/State'].unique())
province_sorted.sort()
print(province_sorted)


# <font color='red'>Comment</font>: we can observe that there are few provinces with duplicate name or, for US - data at both county level and at state level & China with both province and independent territories. Here we show just few examples:
# 
# * ' Norfolk County, MA' & 'Norfolk County, MA' - duplicate county name;
# *  'Providence County, RI' &  'Providence, RI' - duplicate county name from US;
# * 'France' - country name;
# * 'Washington' & 'Washington D.C.' & 'District of Columbia' - duplicate state name?;
# * 'Clark County, W' & 'Washington' (state)?
# * 'New York', 'New York City, NY', 'New York County, NY - possible duplicate for NYC?  
# * 'King County, WA', 'Kittitas County, WA but also Washington (state)?
# 
# There are multiple attributions of `None` or `Unassigned Location`: `Unassigned Location (From Diamond Princess)`, `Unassigned Location, VT`, `Unassigned Location, WA`, `Unknown Location, MA`.
# 
# There are multiple mentions of `from Diamond Princess`. Let's list them as well:
# 

# In[ ]:


diamond_list = list(data_df.loc[data_df['Province/State'].str.contains("Diamond", na=False), 'Province/State'].unique())
diamond_list.sort()
print(diamond_list)


# ## Check Country/Region & Province/State intersection
# 
# 
# We check now if a territory is marked both as a Country/Region and as a Province/State.

# In[ ]:


province_ = list(data_df.loc[~data_df['Province/State'].isna(), 'Province/State'].unique())
country_ = list(data_df['Country/Region'].unique())

common_province_country = set(province_) & set(country_)
print(common_province_country)


# Let's check now the name of the country when the province is in the common list of provinces and countries.

# In[ ]:


for province in list(common_province_country):
    country_list = list(data_df.loc[data_df['Province/State']==province, 'Country/Region'].unique())
    print(province, country_list)


# The analysis of the provinces and countries list should be interpreted as folllowing:
# 
# 
# * Macau, Hong Kong appears both as independent Countries and as part of Mainland China; this is not correct.
# * France & Saint Barthelemy appears as provinces of France. This is not correct because Saint Barthelemy appears as well as an independent state. It must probably be fixed by replacing Saint Barthelemy as part of France, where appears as independent Country.
# * UK, Gibraltar & Channel Islands appears both as countries and as part from UK. It should be corrected by setting Gibraltar * Channel Islands as part of UK where appears as independent state;
# * Faroe Islands appears both as a state and as a part of Denmark. Should be corrected by setting only as a Province/State;
# * Georgia is both a state in US and an independent country. This is not an error.

# ## Check US states & counties
# 
# In US we have both data at state level and at county level.   
# This might mislead when building statistics since we do not know for example if the statistic for Washington (State) includes also the data from King County, WA (a county from Washington state where is also Seattle).  
# 
# 
# Let's check first the list of counties in US.

# In[ ]:


counties_us = list(data_df.loc[(~data_df['Province/State'].isna()) &                                data_df['Province/State'].str.contains("County,", na=False) &                               (data_df['Country/Region']=='US'), 'Province/State'].unique())
counties_us.sort()
print(counties_us)


# Let's check now also the list of locations that are not counties but are not states names.

# In[ ]:


cities_places_us = list(data_df.loc[(~data_df['Province/State'].isna()) &                                (~data_df['Province/State'].str.contains("County,", na=False)) &                               (data_df['Province/State'].str.contains(",", na=False)) &                               (data_df['Country/Region']=='US'), 'Province/State'].unique())
cities_places_us.sort()
print(cities_places_us)


# Few entries are not actual places, as following: `Lackland, TX (From Diamond Princess)`, `Omaha, NE (From Diamond Princess)` `Unassigned Location, VT`, `Unassigned Location, WA`, `Unknown Location, MA`.
# 
# Let's check now the states names.

# In[ ]:


states_us = list(data_df.loc[(~data_df['Province/State'].isna()) &                                (~data_df['Province/State'].str.contains("County,", na=False)) &                               (~data_df['Province/State'].str.contains(",", na=False)) &                               (data_df['Country/Region']=='US'), 'Province/State'].unique())
states_us.sort()
print(states_us)
print(len(states_us))


# There are few items here that are not states: Chicago (city in Illinois), Grand Princess (Diamond Princess?), Unassigned Location (From Diamond Princess).

# # Export the data
# 
# We will export the curated data.

# In[ ]:


data_df.to_csv("covid_19_data.csv", index=False)

