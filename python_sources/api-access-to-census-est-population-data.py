#!/usr/bin/env python
# coding: utf-8

# # Accessing US Census API to get updated populations by county and state
# 
# This is a very simple notebook showing how to access data on population estimates form the US Census API
# 
# ## API Basics
# 
# [Census.gov](https://www.census.gov/en.html) --> [Data](https://www.census.gov/data) --> [Developers](https://www.census.gov/developers/) --> [Available APIs](https://www.census.gov/data/developers/data-sets.html) --> [Populaiton API](https://www.census.gov/data/developers/data-sets/popest-popproj.html)
# 
# > Each year, the Census Bureau's Population Estimates Program uses current data on births, deaths, and migration to calculate population change since the most recent decennial census and produces a time series of estimates of population, demographic components of change, and housing units. The annual time series of estimates begins with the most recent decennial census data and extends to the vintage year.
# 
# ## Population Estimates
# * API Call: https://api.census.gov/data/2019/pep/population
# * Examples and Supported Geographies: https://api.census.gov/data/2019/pep/population.html
# * Variables: https://api.census.gov/data/2019/pep/population/variables.html
# * Example Call: https://api.census.gov/data/2019/pep/population?get=COUNTY,DATE_CODE,DATE_DESC,DENSITY,POP,NAME,STATE&for=region:*&key=YOUR_KEY
# 

# ## API Key Request
# This takes only a few minuets and is free: https://api.census.gov/data/key_signup.html
# 
# Then can load API Key into *Kaggle Secret* at top of screen. 
# 
# >Add-ons --> Secrets

# In[ ]:


#Load API key from Kaggle Secret
#Can find at add-on secret at top of notebook
#Can apply for a free API key here: https://msr-apis.portal.azure-api.net/products

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
census_api_key = user_secrets.get_secret("census_api_key")


# In[ ]:


import pandas as pd
import requests

r= requests.get("https://api.census.gov/data/2019/pep/population?get=POP,NAME,DENSITY&for=state:*&key={}".format(census_api_key))
results = r.json()
columns = results.pop(0)
pd.DataFrame(results, columns=columns)


# ## There are lots more endpoints at the Census website so please let me know if there is other demographic/census information that would be useful!
# 
# They also hae informaiton of healthcare statistics which I think will be my next effort
