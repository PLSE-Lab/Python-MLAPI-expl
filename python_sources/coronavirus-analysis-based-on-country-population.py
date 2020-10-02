#!/usr/bin/env python
# coding: utf-8

# ### Update (3rd Mar) : 
# * The code in this notebook has been updated to include population of States or Countries to the analysis
# * I made the file **population_convid.csv** to use it for further analysis
# * The source of database used from [here](https://www.kaggle.com/francescomandru/coronavirus-covid-19-analysis-updated)

# # World Coronavirus : Analysis Based on Country/State Population
# 
# There is a huge discussion about coronavirus cases and increasing numbers of confirmed cases. Although the numbers may cause terror, a more careful analysis must be done, based on the percentage of Population. You will see that the population of China parts consists of dozens of millions, many of them much more that a European country.
# 
# At the end of the end of the Notebook there is:
# 
# * Table **df_city** gives top 20 cases by ascending the NUMBER of Confirmed Cases
# 
# * Table **df_percentage** gives top 20 cases by ascending the PERCENTAGE of Confirmed Cases based on POPULATION
# 

# In[ ]:


import pandas as pd
import numpy as np


# Adjustments of countries names

# In[ ]:


virus_data = pd.read_csv('../input/virus_data.csv')
virus_data.rename(columns={'Country/Region': 'Country', 'ObservationDate': 'Date'}, inplace=True)
virus_data = virus_data.fillna('unknow')
virus_data['Country'] = virus_data['Country'].str.replace('US','United States')
virus_data['Country'] = virus_data['Country'].str.replace('UK','United Kingdom') 
virus_data['Country'] = virus_data['Country'].str.replace('Mainland China','China')
virus_data['Country'] = virus_data['Country'].str.replace('South Korea','Korea, South')
virus_data['Country'] = virus_data['Country'].str.replace('North Korea','Korea, North')
virus_data['Country'] = virus_data['Country'].str.replace('Macau','China')
virus_data['Country'] = virus_data['Country'].str.replace('Ivory Coast','Cote d\'Ivoire')
virus_data.head()


# In[ ]:


# Group data by Province/State as a first priority or in case there is no Province/State group by Country.
df1=virus_data.groupby(['Province/State',"Country"]).max()

# Make a table with only Confirmed, Deaths and Recovered cases
df2=df1[['Confirmed','Deaths','Recovered']]

# Make a table by ascending the confirmed cases
df_city=df2.sort_values(by=['Confirmed'],ascending=False).reset_index(drop=None) 

# Print the top 20 cases by ascending the confirmed cases
df_city.head(20)


# # Table including the Population of Province/State. All values of population comes from wikipedia.

# In[ ]:


# Use this dataframe with population to map it to State or Countries
population_convid = pd.read_csv('../input/population_convid.csv')
population_convid.head(10)


# In[ ]:


# Insert Population data from "population_convid" to "df_city". 
df_city['Population']=population_convid["Population"]

# Change order of Population Column
df_city = df_city[['Province/State', 'Population','Country', 'Confirmed', 'Deaths', 'Recovered']]
df_city.head()


# # Table of Confirmed cases in pure numbers, but also by Percentage of Province/State the Population.

# In[ ]:


# Percentage of "Confirmed" cases to Population
per= (df_city['Confirmed']/df_city['Population'])*100

# Insert new column of the percentage
df_city['Confirmed (%)']=per

# Change the order of 'Confirmed (%)'
df_city = df_city[['Province/State', 'Population','Country', 'Confirmed','Confirmed (%)', 'Deaths', 'Recovered']]
df_city.head()


# # Table 1: by ascending the pure numbers of Confirmed Cases.

# In[ ]:


# Multiple the percetange with 1000 to make a clear "picture" of comparison between different states/countries
df_city['Confirmed (%)*10^-3']=(per*1000).round(2)

# Change the oredr of ''Confirmed (%)*10^-3'
df_city = df_city[['Province/State', 'Population','Country', 'Confirmed','Confirmed (%)', 'Confirmed (%)*10^-3','Deaths', 'Recovered']]

# Check top 20 cases according the NUMBER of Confirmed cases
df_city.head(20)


# # Table 2: by ascending Confirmed Cases based on Province/State the Population percentage.

# In[ ]:


# Check top 20 cases according the PERCENTAGE of Confirmed cases
df_percentage=df_city.sort_values(by=['Confirmed (%)'],ascending=False).reset_index(drop=None) 

# Check top 20 cases according the PERCENTAGE of Confirmed cases BASES ON POPULATION
df_percentage.head(20)


# # In pure numbers in table 1, China, South Korea and Iran have the majority of confirmed cases. Based on population percentage from table 2 the order changes. 

# 
