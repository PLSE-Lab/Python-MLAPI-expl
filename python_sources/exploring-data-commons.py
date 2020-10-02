#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#install the data commons package
get_ipython().system('pip install -U git+https://github.com/google/datacommons.git@stable-1.x')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datacommons as dc


# ## Get your API key
# 
# Slightly clunky. Need to sign up for a Google Cloud account. Here are instructions for getting an API key: http://docs.datacommons.org/api/setup.html
# 
# You can use insert the key using Add-on > Secrets in the file menu of this notebook

# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
dcKey = user_secrets.get_secret("dcKey")
dc.set_api_key(dcKey)


# ## Pull a list of all cities in the US

# In[ ]:



city_dcids = dc.get_places_in(['country/USA'], 'City')['country/USA']
#city_dcids = city_dcids[0:400]


# ## Get population by city
# 
# I'm only look at cities with 100K+ populations to make processing faster

# In[ ]:


data = pd.DataFrame({'CityId': city_dcids})

data['PopId'] = dc.get_populations(data['CityId'],'Person')
data['population'] = dc.get_observations(data['PopId'], 'count','measuredValue','2017', measurement_method='CensusACS5yrSurvey')

#only look at cities with populations of 100K+
data = data[data['population'] > 100000]

data['population'] = data['population'].astype(int)


# ## Show data from Census Bureau, BLS, CDC and FBI
# 
# This section joins data from Census Bureau, Bureau of Labor Statistics, Center for Disease Control and Federal Bureau of investigations. Joining a dataset from any on of these additional sources took 1-2 lines of code.
# 
# You need a line you need to map city ID to the data source. The second line specifies the specific metric, frequency, date, aggregation method. 
# 

# In[ ]:


# Create the Pandas DataFrame 

data['median_age'] = dc.get_observations(data['PopId'], 'age','medianValue','2017', measurement_method='CensusACS5yrSurvey')
data['unemployment_rate'] = dc.get_observations(data['PopId'],'unemploymentRate','measuredValue','2017',observation_period='P1Y',measurement_method='BLSSeasonallyUnadjusted')

# Get the name of each city
data['City'] = dc.get_property_values(data['CityId'], 'name')
data = data.explode('City')

#Get the name of each state
data['State'] = dc.get_property_values(data['CityId'].str[:8], 'name')
data = data.explode('State')

data.index = data[['City','State']]

data['ObesityId'] = dc.get_populations(data['CityId'],'Person',constraining_properties={'age': 'Years18Onwards','healthBehavior': 'Obesity'})
data['obesity_rate'] = dc.get_observations(data['ObesityId'],'percent','measuredValue','2015','P1Y',measurement_method='CrudePrevalence')

data['ViolentCrimeId'] = dc.get_populations(data['CityId'],'CriminalActivities',constraining_properties={'crimeType': 'ViolentCrime'})
data['violent_crime_rate'] = round(dc.get_observations(data['ViolentCrimeId'],'count','measuredValue','2017','P1Y')/data['population']*100,2)


# ## Show data from Census, BLS, CDC and FBI

# In[ ]:


CitySample = ['New York','Los Angeles','Chicago','Houston','Philadelphia','Austin','Denver','Seattle','San Francisco', 'Washington DC','Boston','Miami','Charleston']
data[data['City'].isin(CitySample)].sort_values(by='population',ascending=False)[['population','unemployment_rate','obesity_rate','violent_crime_rate']][:11]


# In[ ]:




