#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# First, let's get the codes for the countries from another database

# In[ ]:


world_pop = pd.read_csv('../input/wikipedia-iso-country-codes.csv')
world_pop_codes = world_pop.filter(items=['English short name lower case', 'Alpha-2 code'])
world_pop_codes.columns = ['country_name', 'country']
world_pop_codes.head()


# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")


# I'm interested to see which are the most polluted countries, so I will take the top 5

# In[ ]:


# query to pass to 
query = """SELECT country, SUM(value)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country
            ORDER BY SUM(value) DESC
            LIMIT 5
            
        """
pollution_by_country = open_aq.query_to_pandas_safe(query)


# In[ ]:


pol_per_country = pd.merge(world_pop_codes, pollution_by_country, on='country', how='inner')
print(pol_per_country)


# We can see that the US is not on the top polluted countries and the most air pollution comes from Spain.

# Now, I want to check the most polluted cities:

# In[ ]:


# query to pass to 
query = """SELECT city, SUM(value)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY city
            ORDER BY SUM(value) DESC
            LIMIT 5 
        """
most_polluted = open_aq.query_to_pandas_safe(query)


# In[ ]:


print(most_polluted)


# MK005A is in Macedonia according to "https://openaq.org/#/location/Rektorat?_k=7jixeb"
# and it's interesting to see even though MK005A is the top polluted city, Macedonia didn't appear
# on the most polluted countries.

# Now, let's see in which day is the most polluted day:

# In[ ]:


# query to find out the max pollution which 
# happen on each day of the week
query = """SELECT MAX(value), 
                  EXTRACT(DAYOFWEEK FROM timestamp)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp)
            ORDER BY MAX(value) DESC
        """
max_pol_per_day = open_aq.query_to_pandas_safe(query)


# In[ ]:


print(max_pol_per_day)


# We can see that Wednesday is the most polluted day, and after it Monday. Now, what is the most polluted hour?

# In[ ]:


# query to find out the max pollution which 
# happen on each hour of the day
query = """SELECT MAX(value), 
                  EXTRACT(HOUR FROM timestamp)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY EXTRACT(HOUR FROM timestamp)
            ORDER BY MAX(value) DESC
        """
max_pol_per_hour = open_aq.query_to_pandas_safe(query)


# In[ ]:


print(max_pol_per_hour)


# We can see that we should probably stay inside at 16:00 and at 13:00

# In[ ]:


sns.set(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))

# Plot the orbital period with horizontal boxes
sns.boxplot(x="f1_", y= "f0_", data=max_pol_per_hour,
            whis=np.inf, palette="vlag")

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="Pollution")
ax.set(xlabel="Hour of Day")
sns.despine(trim=True, left=True)

