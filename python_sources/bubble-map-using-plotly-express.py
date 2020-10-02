#!/usr/bin/env python
# coding: utf-8

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


df = pd.read_csv('/kaggle/input/data-jobs-listings-glassdoor/glassdoor.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


for column in df.columns:
    print(column)


# Wow, that's a lot of information to parse through...
# 
# For now I'm just interested in making a simple bubble map for this data set as we have 'geo' data.

# Counting the number of nan values for the countries for curiosity

# In[ ]:


df['map.country'].isnull().sum()


# Analysis this feature column with nan values dropped.

# In[ ]:


countriesData = df['map.country'].dropna()
countriesData


# Noticing there is not a proper convention for the countries names:

# In[ ]:


countriesData.unique()


# Will format those to conventional names as best as I can using iso3166 get function:

# In[ ]:


#What is great about this function is that it returns always the same output for all inputs corresponding to identifiers of the country
from iso3166 import countries
print(countries.get('us'))
print(countries.get('USA'))
print(countries.get('United States of America'))


# The problem is that it is not a perfect dictionary and doesnt identify all possible ways to reference to a country:
# 
# for example: countries.get('U.S.A.') would give us an error

# I will convert the 'map.country' values to standardized ones using countries.get but I need to handle exceptions:

# In[ ]:


def rename(country):
    try:
        return countries.get(country).alpha3
    except:
        return (np.nan)


# In[ ]:


old_sample_number = countriesData.shape[0]

countriesData = countriesData.apply(rename)
countriesData = countriesData.dropna()

new_sample_number = countriesData.shape[0]
print('we lost', old_sample_number-new_sample_number, 'samples after converting')


# That's a lot of samples lost but I don't know better dictionaries that can map those conventionless country names to ISO convention names

# In[ ]:


countriesData


# Now all names follow the same convention! 

# It's time to plot those count: first with a simple barplot using seaborn and next in a bubble world map using plotly express

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns 
plt.figure(figsize=(24, 6))
sns.barplot(countriesData.value_counts()[countriesData.value_counts()>150].index, countriesData.value_counts()[countriesData.value_counts()>150].values)


# Creating a dataframe to help plotly express understand the data.
# 
# Read: https://plot.ly/python/px-arguments/

# In[ ]:


#Creating a DataFrame that stores the ID of the countries and their count
country_df = pd.DataFrame(data=[countriesData.value_counts().index, countriesData.value_counts().values],index=['country','count']).T


# In[ ]:


#Converting count values to int because this will be important for plotly
country_df['count']=pd.to_numeric(country_df['count'])


# In[ ]:


country_df.head()


# Plotly express can make bubble maps really fast and very high level 
# 
# See the doc here: https://plot.ly/python/bubble-maps/

# In[ ]:


import plotly.express as px
fig = px.scatter_geo(country_df, locations="country", size='count',
                     hover_name="country", color='country',
                     projection="natural earth")
fig.show()


# Obviously this approach had problems for exemple I dropped a third of the samples when renaming the countries to the ISO standard.
# 
# At least this is a functional and pretty bubble map but I'm really not sure it is relevent...

# Thank you for reading!
